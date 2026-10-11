"""Oobleck VAE (vocoder) for the YuE2 torch worker.

Exact port of the Oobleck classes in ComfyUI
``comfy/ldm/audio/autoencoder.py`` (adapted from Stability-AI
stable-audio-tools; post-PR-#16250 ``master``, revision pinned in
README), instantiated with the YuE2 VAE config
(``sample_rate`` 48000, ``downsampling_ratio`` 1920, ``latent_dim`` 64,
decoder ``channels`` 64 / ``c_mults`` [1,2,4,8,16,32] /
``strides`` [2,2,4,4,5,6], snake activations, no final tanh).

Only the decoder runs in generation (plus the identity bottleneck);
the encoder is constructed so the full ``vae.*`` section loads with
``strict=True``, then deleted before the GPU move (VRAM staging).

Native vocoder rate is 48 kHz at 1920x upsampling (latent rate exactly
25 Hz = the semantic frame rate, so DiT frames map 1:1 to audio).
"""

import math

import torch
import torch.nn as nn

# YuE2 VAE operating point (from yue2-vae-config.json).
NATIVE_SR = 48000
UPSAMPLE = 1920
LATENT_DIM = 64


def _snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 1e-9)) * pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1).to(x.device))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1).to(x.device))
        return _snake_beta(x, alpha, beta)


def WNConv1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(
        nn.ConvTranspose1d(*args, **kwargs))


def _activation(name, channels):
    if name == "elu":
        return nn.ELU()
    if name == "snake":
        return SnakeBeta(channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {name}")


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False):
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2
        act = "snake" if use_snake else "elu"
        self.layers = nn.Sequential(
            _activation(act, out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=7, dilation=dilation, padding=padding),
            _activation(act, out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                     kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False):
        super().__init__()
        act = "snake" if use_snake else "elu"
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, dilation=1,
                         use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, dilation=3,
                         use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, dilation=9,
                         use_snake=use_snake),
            _activation(act, in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=2 * stride, stride=stride,
                     padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False):
        super().__init__()
        act = "snake" if use_snake else "elu"
        self.layers = nn.Sequential(
            _activation(act, in_channels),
            WNConvTranspose1d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=2 * stride, stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2),
            ResidualUnit(out_channels, out_channels, dilation=1,
                         use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, dilation=3,
                         use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, dilation=9,
                         use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(self, in_channels=2, channels=64, latent_dim=128,
                 c_mults=(1, 2, 4, 8, 16, 32),
                 strides=(2, 2, 4, 4, 5, 6), use_snake=True):
        super().__init__()
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [WNConv1d(in_channels=in_channels,
                           out_channels=c_mults[0] * channels,
                           kernel_size=7, padding=3)]
        for i in range(self.depth - 1):
            layers.append(EncoderBlock(c_mults[i] * channels,
                                       c_mults[i + 1] * channels,
                                       stride=strides[i],
                                       use_snake=use_snake))
        layers += [_activation("snake" if use_snake else "elu",
                               c_mults[-1] * channels),
                   WNConv1d(in_channels=c_mults[-1] * channels,
                            out_channels=latent_dim,
                            kernel_size=3, padding=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, out_channels=2, channels=64, latent_dim=64,
                 c_mults=(1, 2, 4, 8, 16, 32),
                 strides=(2, 2, 4, 4, 5, 6), use_snake=True,
                 final_tanh=False):
        super().__init__()
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [WNConv1d(in_channels=latent_dim,
                           out_channels=c_mults[-1] * channels,
                           kernel_size=7, padding=3)]
        for i in range(self.depth - 1, 0, -1):
            layers.append(DecoderBlock(
                c_mults[i] * channels, c_mults[i - 1] * channels,
                stride=strides[i - 1], use_snake=use_snake))
        layers += [_activation("snake" if use_snake else "elu",
                               c_mults[0] * channels),
                   WNConv1d(in_channels=c_mults[0] * channels,
                            out_channels=out_channels,
                            kernel_size=7, padding=3, bias=False),
                   nn.Tanh() if final_tanh else nn.Identity()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VAEBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_discrete = False

    def decode(self, x):
        return x


class AudioOobleckVAE(nn.Module):
    """YuE2 VAE (decoder path used for generation)."""

    def __init__(self):
        super().__init__()
        self.encoder = OobleckEncoder()
        self.decoder = OobleckDecoder()
        self.bottleneck = VAEBottleneck()

    def decode(self, x):
        """Full-song decode: (B, 64, T) -> (B, 2, T*1920)."""
        if x.dim() != 3 or x.shape[1] != LATENT_DIM:
            raise ValueError(
                f"latents must be (B, {LATENT_DIM}, T), got {tuple(x.shape)}")
        if x.shape[-1] < 1:
            raise ValueError("empty latent sequence")
        return self.decoder(self.bottleneck.decode(x))

    @staticmethod
    def load_decoder(checkpoint, device="cpu", dtype=None):
        """Strict-load the ``vae.*`` section, drop the encoder, move to
        device. ``checkpoint`` maps full names (``vae.encoder...``)."""
        vae = AudioOobleckVAE()
        state = {k[4:]: v for k, v in checkpoint.items()
                 if k.startswith("vae.")}
        vae.load_state_dict(state, strict=True)
        del vae.encoder
        if dtype is not None:
            vae.to(dtype=dtype)
        return vae.to(device=device).eval()


# Tiling grid from yue2-vae-config.json (decode_core_frames/decode_halo_frames).
DECODE_CORE_FRAMES = 1024
DECODE_HALO_FRAMES = 16


def decode_tiled(vae, latents, core=DECODE_CORE_FRAMES,
                 halo=DECODE_HALO_FRAMES):
    """Decode long latent sequences tile by tile (VRAM staging).

    Full-song fp32 decode materializes multi-GB feature maps at high-res
    levels (~21 GB transient for a 240 s song — fatal on 8 GB). Tiles of
    ``core`` frames plus ``halo`` overlap on each side are decoded
    independently, halo outputs trimmed, cores concatenated. Output length
    is exactly ``frames * 1920``. Deterministic for fixed weights.
    """
    import torch
    frames = latents.shape[-1]
    if frames <= core:
        return vae.decode(latents)
    parts = []
    for c0 in range(0, frames, core):
        c1 = min(c0 + core, frames)
        h0 = max(0, c0 - halo)
        h1 = min(frames, c1 + halo)
        with torch.no_grad():
            out = vae.decode(latents[..., h0:h1])
        trim_left = (c0 - h0) * UPSAMPLE
        trim_right = (h1 - c1) * UPSAMPLE
        end = out.shape[-1] - trim_right if trim_right else out.shape[-1]
        parts.append(out[..., trim_left:end])
    return torch.cat(parts, dim=-1)
