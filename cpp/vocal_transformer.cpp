// VocalTransformer native processor (ANode standard C-ABI).
//
// Studio-grade vocal pitch / formant / gender transformer:
//   - resampled-analysis-frame pitch shifting (frame reads at stride ratio)
//   - 1024-pt True-Envelope estimation (Roebel-Rodet, SYMMETRIC quefrency
//     lifter with formant-bandwidth-adaptive cutoff)
//   - same-grid peak-locked phase vocoder (Laroche-Dolson), unvoiced fallback
//   - formant-preserving envelope replacement + asymmetric multi-band VTLN warp
//     (F1 decoupled; precomputed spectral-tilt & H1-harmonic shaping)
//   - raised-cosine sibilant bypass (3.5-5.5 kHz) blending COMPLEX bins
//   - tract-shaped 1.5-7 kHz xorshift32 aspiration noise (deterministic, RT-safe)
//   - ring-buffer OLA with FIXED emission latency (kLatency = 4608 samples = 96 ms)
//
// mix = 0 is a bit-exact memcpy bypass; set_param(mix) clears transient state on
// bypass-boundary transitions (anti-ghosting). All ring indices are wrapped into
// [0, kRingSize) via & kRingMask AFTER reducing positions into int range — the
// double->int cast must never receive an out-of-range value (UB).
//
// Zero steady-state heap allocation: every buffer is a fixed member array.

#include <cmath>
#include <cstring>
#include <algorithm>
#include <new>

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

namespace {

constexpr int   kFFT = 1024;
constexpr int   kHop = 256;
constexpr int   kHalf = 513;              // real frequency bins incl. DC + Nyquist
constexpr int   kLifter = 32;
constexpr int   kIters = 3;
constexpr int   kMaxChannels = 2;
constexpr int   kRingSize = 8192;         // input history + OLA ring (power of 2)
constexpr int   kRingMask = kRingSize - 1;
// Fixed emission latency: output block k reads ring positions L behind the
// input stream. Frame m spans 1024*ratio input samples (ratio <= 4 at +-24 st)
// and output sample o is complete once frame floor(o/256) has been processed,
// so L >= 512 + 1024*4 = 4608 keeps every emitted sample fully accumulated.
constexpr long long kLatency = 4608;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kXoverLowHz = 3500.0f;
constexpr float kXoverHighHz = 5500.0f;
constexpr unsigned int kRngSeed = 0x1D872B41u;

inline float wrap_phase(float x) {
    while (x > kPi) x -= kTwoPi;
    while (x <= -kPi) x += kTwoPi;
    return x;
}

class VocalTransformerProcessor {
public:
    VocalTransformerProcessor()
        : sr_(48000.0f),
          pitch_st_(0.0f), formant_st_(0.0f), gender_morph_(0.0f),
          breathiness_(0.0f), sibilant_mix_(0.8f), mix_(1.0f), prev_mix_(1.0f),
          rng_state_(kRngSeed) {
        build_tables();
        recompute_tables();
        reset();
    }

    void set_samplerate(float sr) {
        // Windows are normalized-frequency; sr scales the Hz mapping of the
        // VTLN band boundary, the tilt/H1 shaping bands, and the breath band
        // limits — every table below must be rebuilt when the rate changes.
        if (sr > 1.0f && sr != sr_) {
            sr_ = sr;
            recompute_tables();
        }
    }

    void set_param(int id, float v) {
        switch (id) {
            case 0: pitch_st_ = std::max(-24.0f, std::min(24.0f, v)); break;
            case 1: formant_st_ = std::max(-24.0f, std::min(24.0f, v)); break;
            case 2:
                gender_morph_ = std::max(-1.0f, std::min(1.0f, v));
                // VTLN warp buckets, spectral tilt, and H1 emphasis all depend
                // only on gender_morph; rebuild every table here (param-change
                // path) instead of recomputing 513 atan/sin/cos/pow per frame
                // inside replace_envelope()/synthesize().
                recompute_tables();
                break;
            case 3: breathiness_ = std::max(0.0f, std::min(1.0f, v)); break;
            case 4: sibilant_mix_ = std::max(0.0f, std::min(1.0f, v)); break;
            case 5:
                mix_ = std::max(0.0f, std::min(1.0f, v));
                // Bypass-boundary anti-ghosting: crossing mix = 0 invalidates the
                // transient pipeline (stale rings / OLA / phase history).
                if ((prev_mix_ <= 0.0f) != (mix_ <= 0.0f)) reset_transient();
                prev_mix_ = mix_;
                break;
            default: break;
        }
    }

    void reset() {
        reset_transient();
        prev_mix_ = mix_;
        rng_state_ = kRngSeed;
    }

    void process(const float* in, float* out, int channels, int frames) {
        if (channels < 1) channels = 1;
        if (channels > kMaxChannels) channels = kMaxChannels;
        if (frames <= 0) return;

        // Dry bypass: bit-exact memcpy early-out (zero latency, zero CPU).
        // Must honor the mono->stereo duplication contract of the normal path,
        // otherwise a raw channels==1 call would leave output channel 1 stale.
        if (mix_ <= 0.0f) {
            const int chs = (channels == 1) ? kMaxChannels : channels;
            for (int c = 0; c < chs; ++c) {
                const float* src = (channels == 1) ? in : in + c * frames;
                std::memcpy(out + c * frames, src,
                            static_cast<size_t>(frames) * sizeof(float));
            }
            return;
        }

        // Mono input: duplicate internally to both output channels.
        const int chs = (channels == 1) ? kMaxChannels : channels;
        const float ratio = std::pow(2.0f, pitch_st_ / 12.0f);

        for (int c = 0; c < chs; ++c) {
            const float* in_ch = (channels == 1) ? in : in + c * frames;
            float* out_ch = out + c * frames;

            // Push this block's input into the history ring.
            for (int i = 0; i < frames; ++i)
                in_ring_[c][(total_received_[c] + i) & kRingMask] = in_ch[i];
            total_received_[c] += frames;

            // Process every frame whose input span is fully available.
            // Frame m starts at input position 256*m and spans 1024*ratio
            // input samples — the pitch shift happens in the frame read.
            while (256.0 * (double)frame_index_[c] + 1024.0 * (double)ratio
                   <= (double)total_received_[c] + 1e-6) {
                process_frame(c, ratio);
            }

            // Emit this block's output from the fixed latency L behind the
            // input stream (every emitted sample is fully accumulated; the
            // first L samples of the stream are silence).
            const long long read_base = total_received_[c] - kLatency;
            for (int i = 0; i < frames; ++i) {
                const long long o = read_base + i;
                if (o >= 0) {
                    out_ch[i] = ola_ring_[c][o & kRingMask];
                    ola_ring_[c][o & kRingMask] = 0.0f;
                } else {
                    out_ch[i] = 0.0f;
                }
            }
        }

        // Zero any unused output channels (anti-ghosting). Defensive only:
        // chs is always kMaxChannels today (mono is internally duplicated).
        for (int c = chs; c < kMaxChannels; ++c)
            std::memset(out + c * frames, 0, static_cast<size_t>(frames) * sizeof(float));
    }

private:
    // ---- Parameters -------------------------------------------------------
    float sr_;
    float pitch_st_;       // [-24, +24] semitones
    float formant_st_;     // [-24, +24] semitones
    float gender_morph_;   // [-1, +1] VTLN warping
    float breathiness_;    // [0, 1]
    float sibilant_mix_;   // [0, 1]
    float mix_;            // [0, 1]
    float prev_mix_;       // bypass-transition detection

    // ---- Per-channel persistent state --------------------------------------
    float in_ring_[kMaxChannels][kRingSize];   // raw input history
    float ola_ring_[kMaxChannels][kRingSize];  // output OLA ring (trailing read)
    long long total_received_[kMaxChannels];   // input samples received
    long long frame_index_[kMaxChannels];      // frames processed (start = 256*m)
    float prev_phase_in_[kMaxChannels][kHalf];
    float prev_phase_out_[kMaxChannels][kHalf];

    // ---- Pre-computed tables (constructor only) -----------------------------
    float window_[kFFT];        // analysis Hann
    float synth_window_[kFFT];  // Hann / 1.5 (exact COLA unity at 75% overlap)
    int   rev_[kFFT];           // FFT bit-reversal permutation
    float cos_tab_[kFFT / 2];   // twiddle tables
    float sin_tab_[kFFT / 2];
    float vtln_warp_bin_[kHalf]; // VTLN warp buckets: source bin -> dest pos
                                // (rebuilt in recompute_tables() on gender/sr change)
    float excitation_shaper_[kHalf]; // spectral tilt * H1 emphasis, multiplied
                                // into the fine excitation in synthesize()
                                // (rebuilt in recompute_tables() too)

    // ---- Per-hop scratch (no allocation) ------------------------------------
    float scratch_time_[kFFT];     // synthesized output frame (pre-window)
    float fft_re_[kFFT];           // complex FFT work buffers (planar layout)
    float fft_im_[kFFT];
    float spec_re_[kHalf];         // analysis half-spectrum (original, for blending)
    float spec_im_[kHalf];
    float mag_[kHalf];
    float phase_[kHalf];
    float log_mag_[kHalf];         // running log-envelope A_i(k)
    float envelope_[kHalf];        // original envelope -> final warped envelope
    float warped_envelope_[kHalf]; // formant-resampled stage
    float excitation_[kHalf];      // fine excitation H(k), from ORIGINAL envelope
    float synth_mag_[kHalf];       // destination grid
    float synth_phase_[kHalf];     // destination grid
    int   peak_bins_[kHalf];
    int   peak_owner_[kHalf];      // source bin -> nearest peak index
    unsigned int rng_state_;

    void reset_transient() {
        for (int c = 0; c < kMaxChannels; ++c) {
            std::memset(in_ring_[c], 0, sizeof(float) * kRingSize);
            std::memset(ola_ring_[c], 0, sizeof(float) * kRingSize);
            total_received_[c] = 0;
            frame_index_[c] = 0;
            std::memset(prev_phase_in_[c], 0, sizeof(float) * kHalf);
            std::memset(prev_phase_out_[c], 0, sizeof(float) * kHalf);
        }
    }

    void build_tables() {
        for (int i = 0; i < kFFT; ++i) {
            const float w = 0.5f - 0.5f * std::cos(kTwoPi * static_cast<float>(i) / kFFT);
            window_[i] = w;
            synth_window_[i] = w / 1.5f;
        }
        int bits = 0;
        while ((1 << bits) < kFFT) ++bits;
        for (int i = 0; i < kFFT; ++i) {
            int r = 0, x = i;
            for (int b = 0; b < bits; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
            rev_[i] = r;
        }
        for (int j = 0; j < kFFT / 2; ++j) {
            cos_tab_[j] = std::cos(kTwoPi * static_cast<float>(j) / kFFT);
            sin_tab_[j] = std::sin(kTwoPi * static_cast<float>(j) / kFFT);
        }
    }

    // Precompute every parameter-dependent spectral table:
    //   vtln_warp_bin_[k]   — asymmetric multi-band VTLN allpass warp bucket
    //                         (source bin -> destination position). The F1
    //                         region (< 1 kHz) warps at reduced intensity so
    //                         the disproportionately long adult-male pharynx
    //                         shifts F2/F3 more than F1; identity at neutral.
    //   excitation_shaper_[k] — spectral tilt (clamped to +-8 dB) times H1
    //                         harmonic emphasis (gated strictly off DC). Kills
    //                         the buzzy/pinched quality on upward shifts and
    //                         dullness on downward shifts.
    // Depends only on gender_morph_ / sr_; rebuilt on param or rate change.
    void recompute_tables() {
        const float alpha_base = gender_morph_ * 0.25f;
        const float bin_hz = sr_ / static_cast<float>(kFFT);

        for (int k = 0; k < kHalf; ++k) {
            const float f_hz = static_cast<float>(k) * bin_hz;

            // 1. Asymmetric multi-band VTLN warp table.
            if (alpha_base == 0.0f) {
                vtln_warp_bin_[k] = static_cast<float>(k);
            } else {
                float alpha = alpha_base;
                if (f_hz < 1000.0f) {
                    alpha *= (0.6f + 0.4f * (f_hz / 1000.0f));
                }
                const float w = kPi * static_cast<float>(k) / static_cast<float>(kHalf - 1);
                const float wp = w + 2.0f * std::atan(
                    -alpha * std::sin(w) / (1.0f + alpha * std::cos(w)));
                float b = wp / kPi * static_cast<float>(kHalf - 1);
                if (b < 0.0f) b = 0.0f;
                if (b > static_cast<float>(kHalf - 1)) b = static_cast<float>(kHalf - 1);
                vtln_warp_bin_[k] = b;
            }

            // 2. Precomputed spectral tilt & H1 harmonic emphasis.
            float tilt_db = -gender_morph_ * 6.0f * (f_hz / 8000.0f);
            if (tilt_db > 8.0f) tilt_db = 8.0f;
            if (tilt_db < -8.0f) tilt_db = -8.0f;
            const float tilt_gain = std::pow(10.0f, tilt_db / 20.0f);

            float h1_gain = 1.0f;
            if (gender_morph_ > 0.0f && k > 0 && f_hz < 350.0f) {
                h1_gain += 0.75f * gender_morph_ * (1.0f - f_hz / 350.0f);
            }

            excitation_shaper_[k] = tilt_gain * h1_gain;
        }
    }

    // ---- Per-hop pipeline ---------------------------------------------------

    void process_frame(int c, float ratio) {
        // 1. Analysis frame: read from the input ring at stride `ratio`
        //    (linear interpolation). Frame m starts at input position 256*m;
        //    the pitch shift happens right here in the frame read.
        //    fpos grows without bound over a long session, so wrap it into
        //    ring range BEFORE the double->int cast — an out-of-range cast is
        //    UB and (on x86 cvttsd2si) silently reads ring position 0.
        const double fpos = std::fmod(256.0 * (double)frame_index_[c], (double)kRingSize);
        for (int i = 0; i < kFFT; ++i) {
            const double pos = fpos + (double)i * (double)ratio;
            const int i0 = (int)std::floor(pos) & kRingMask;
            const float frac = (float)(pos - std::floor(pos));
            const float v = in_ring_[c][i0] * (1.0f - frac)
                          + in_ring_[c][(i0 + 1) & kRingMask] * frac;
            scratch_time_[i] = v * window_[i];
        }
        forward_real(scratch_time_, spec_re_, spec_im_);

        // 2. Magnitude / phase.
        for (int k = 0; k < kHalf; ++k) {
            mag_[k] = std::sqrt(spec_re_[k] * spec_re_[k] + spec_im_[k] * spec_im_[k]);
            phase_[k] = std::atan2(spec_im_[k], spec_re_[k]);
        }

        // 3. True envelope (Roebel-Rodet, symmetric quefrency lifter).
        true_envelope();

        // 4. Fine excitation from the analysis envelope (before replacement),
        //    clamped to prevent explosion in spectral troughs.
        for (int k = 0; k < kHalf; ++k) {
            float h = mag_[k] / (envelope_[k] + 1e-9f);
            excitation_[k] = h < 0.0f ? 0.0f : (h > 100.0f ? 100.0f : h);
        }

        // 5. Formant-preserving envelope replacement + formant/VTLN warp.
        replace_envelope(ratio);

        // 6. Spectral peaks (same grid as synthesis — no bin mapping).
        const int num_peaks = find_peaks();

        // 7. Peak-locked phases (time-stretch propagation) + magnitudes.
        synthesize(c, num_peaks, ratio);

        // 8. Sibilant complex-bin blend + reconstruction into scratch_time_.
        reconstruct();

        // 9. Synthesis window, accumulate into the OLA ring at this frame's
        //     output position (256*m). Emission is handled by the trailing
        //     read pointer in process() once the frame is the last contributor.
        const long long base = 256LL * frame_index_[c];
        for (int i = 0; i < kFFT; ++i)
            ola_ring_[c][(base + i) & kRingMask] += scratch_time_[i] * synth_window_[i];
        frame_index_[c]++;
    }

    void true_envelope() {
        float* A = log_mag_;
        for (int k = 0; k < kHalf; ++k)
            A[k] = std::log(mag_[k] + 1e-9f);

        for (int it = 0; it < kIters; ++it) {
            // Real, EVEN 1024-pt spectrum from A -> IFFT -> real cepstrum.
            for (int k = 0; k < kHalf; ++k) { fft_re_[k] = A[k]; fft_im_[k] = 0.0f; }
            for (int k = 1; k < kHalf - 1; ++k) {
                fft_re_[kFFT - k] = A[k];
                fft_im_[kFFT - k] = 0.0f;
            }
            fft_run(true);

            // Symmetric liftering on BOTH quefrency halves: lifting only the
            // positive half would destroy c(n) = c(N - n) symmetry and make the
            // forward transform complex (imaginary leakage into the envelope).
            //
            // Adaptive lifter cutoff: feminine morphs (gender_morph_ > 0) have
            // higher acoustic wall/radiation losses relative to vocal-tract
            // volume, broadening formant bandwidths (lower Q). Shorten the
            // cutoff toward 18 (broader peaks) as gender_morph_ -> 1; retain
            // the full 32-quefrency resolution (sharp peaks) at =< 0.
            int eff_lifter = kLifter;
            if (gender_morph_ > 0.0f) {
                eff_lifter = static_cast<int>(kLifter - gender_morph_ * 12.0f);
                if (eff_lifter < 18) eff_lifter = 18;
            }
            apply_symmetric_lifter(fft_re_, kFFT, eff_lifter);

            // Forward FFT of the liftered (real, even) cepstrum.
            for (int i = 0; i < kFFT; ++i) fft_im_[i] = 0.0f;
            fft_run(false);
            for (int k = 0; k < kHalf; ++k)
                A[k] = std::max(A[k], fft_re_[k]);   // imag ~ 0 by evenness
        }

        // Clamp before exponentiation: the running max() can only grow.
        const float a_max = std::log(1.0e6f);
        for (int k = 0; k < kHalf; ++k)
            envelope_[k] = std::exp(std::min(A[k], a_max));
    }

    static void apply_symmetric_lifter(float* cep, int n, int cutoff) {
        for (int q = 0; q < cutoff; ++q) {
            const float w = std::cos(0.5f * kPi * static_cast<float>(q) / cutoff);
            cep[q] *= w;
            if (q > 0) cep[n - q] *= w;   // mirror quefrency (q = 0 applied once)
        }
        std::memset(cep + cutoff, 0,
                    static_cast<size_t>(n - 2 * cutoff + 1) * sizeof(float));
    }

    static float interp(const float* buf, float pos) {
        // Clamp the FRACTION too: clamping only the integer index would leave a
        // negative fraction when pos < 0, extrapolating past the array's lower
        // edge and potentially producing a negative envelope.
        if (pos < 0.0f) pos = 0.0f;
        if (pos > static_cast<float>(kHalf - 1)) pos = static_cast<float>(kHalf - 1);
        const int i = static_cast<int>(pos);
        const int j = (i < kHalf - 1) ? i + 1 : i;
        return buf[i] + (buf[j] - buf[i]) * (pos - static_cast<float>(i));
    }

    // Formant-preserving envelope replacement + formant/VTLN warp.
    // The frame read at stride `ratio` moved every formant up by `ratio`;
    // V_base(k) = V_ana(k*ratio) undoes that, then formant_shift/VTLN apply.
    void replace_envelope(float ratio) {
        // (a) Undo the resampler's formant shift (edge-clamped interpolation).
        for (int k = 0; k < kHalf; ++k) {
            float pos = static_cast<float>(k) * ratio;
            if (pos > static_cast<float>(kHalf - 1))
                pos = static_cast<float>(kHalf - 1);
            warped_envelope_[k] = interp(envelope_, pos);
        }

        // (b) Formant scaling: evaluate V_base at k / R_F (edge-clamped,
        //     no zero fill — broadband gain must be preserved).
        const float rf = std::pow(2.0f, formant_st_ / 12.0f);
        for (int k = 0; k < kHalf; ++k) {
            float pos = static_cast<float>(k) / rf;
            if (pos < 0.0f) pos = 0.0f;
            if (pos > static_cast<float>(kHalf - 1))
                pos = static_cast<float>(kHalf - 1);
            envelope_[k] = interp(warped_envelope_, pos);
        }

        // (c) VTLN bilinear allpass warp (omega'(0)=0, omega'(pi)=pi preserved),
        //     via the precomputed warp-bucket table (rebuilt on gender_morph
        //     change). Table is identity when gender_morph_ = 0.
        for (int k = 0; k < kHalf; ++k)
            warped_envelope_[k] = interp(envelope_, vtln_warp_bin_[k]);
        std::memcpy(envelope_, warped_envelope_, kHalf * sizeof(float));
    }

    int find_peaks() {
        int n = 0;
        for (int k = 1; k < kHalf - 1; ++k) {
            if (mag_[k] > mag_[k - 1] && mag_[k] >= mag_[k + 1] && mag_[k] > 1e-6f)
                peak_bins_[n++] = k;
        }
        // Assign each source bin to its nearest peak (-1 when unvoiced).
        int pi = 0;
        for (int k = 0; k < kHalf; ++k) {
            while (pi + 1 < n &&
                   std::abs(k - peak_bins_[pi + 1]) < std::abs(k - peak_bins_[pi]))
                ++pi;
            peak_owner_[k] = (n > 0) ? pi : -1;
        }
        return n;
    }

    void synthesize(int c, int num_peaks, float ratio) {
        // Classic same-grid peak-locked time-stretch propagation:
        //   analysis hop  Ha' = 256/ratio  (frame starts advance 256 input
        //                  samples = Ha' analysis samples)
        //   synthesis hop Hs  = 256
        //   phi_syn(k) += Omega_k*Hs + dphi(k)*(Hs/Ha') = Omega_k*256 + dphi*ratio
        const float ha = kHop / ratio;
        for (int k = 0; k < kHalf; ++k) {
            const float omega_k = kTwoPi * static_cast<float>(k) / kFFT;
            const float dphi = wrap_phase(phase_[k] - prev_phase_in_[c][k]
                                          - omega_k * ha);
            synth_phase_[k] = wrap_phase(prev_phase_out_[c][k]
                                         + omega_k * kHop + dphi * ratio);
        }

        // Rigid peak-locking: every bin inherits its nearest peak's propagated
        // phase plus the (stationary) source phase offsets. With no peaks
        // (unvoiced frames) the propagated values stand.
        if (num_peaks > 0) {
            for (int k = 0; k < kHalf; ++k) {
                const int kp = peak_bins_[peak_owner_[k]];
                synth_phase_[k] = wrap_phase(synth_phase_[kp]
                                             + phase_[k] - phase_[kp]);
            }
        }

        // Magnitudes: replaced envelope x (fine excitation x precomputed
        // excitation shaper — spectral tilt and H1 harmonic emphasis).
        for (int k = 0; k < kHalf; ++k)
            synth_mag_[k] = envelope_[k] * (excitation_[k] * excitation_shaper_[k]);

        // Update phase history for the next frame.
        for (int k = 0; k < kHalf; ++k) {
            prev_phase_in_[c][k] = phase_[k];
            prev_phase_out_[c][k] = synth_phase_[k];
        }
    }

    void reconstruct() {
        // Sibilant crossfade gain + complex-bin blending of the BAND-LIMITED,
        // tract-shaped aspiration noise, then rebuild the full conjugate-
        // symmetric spectrum and invert. NEVER interpolate phase angles:
        // (1-g)*Y_voc + g*X_orig avoids +-pi branch-cut cancellation.
        const float bin_hz = sr_ / static_cast<float>(kFFT);

        // Tract-shaped aspiration: normalize the final envelope peak so the
        // breath gain follows the frame's formant shape, then inject the
        // deterministic xorshift32 noise into complex bins ONLY in the
        // 1.5-7 kHz band where real glottal aspiration energy lives
        // (DC and Nyquist strictly skipped).
        float max_env = 1e-9f;
        for (int k = 0; k < kHalf; ++k) {
            if (envelope_[k] > max_env) max_env = envelope_[k];
        }
        const int k_low = static_cast<int>(1500.0f * (kFFT / sr_));
        const int k_high = static_cast<int>(7000.0f * (kFFT / sr_));
        const float breath_scale = breathiness_ * 0.08f;

        for (int kd = 0; kd < kHalf; ++kd) {
            float vr = synth_mag_[kd] * std::cos(synth_phase_[kd]);
            float vi = synth_mag_[kd] * std::sin(synth_phase_[kd]);

            // Complex tract-filtered breath injection (1.5-7 kHz band).
            if (breathiness_ > 0.0f && kd >= k_low && kd <= k_high
                && kd < kHalf - 1) {
                rng_state_ ^= rng_state_ << 13;
                rng_state_ ^= rng_state_ >> 17;
                rng_state_ ^= rng_state_ << 5;
                const float n_re =
                    (static_cast<float>(rng_state_ & 0xFFFFu) / 32768.0f) - 1.0f;

                rng_state_ ^= rng_state_ << 13;
                rng_state_ ^= rng_state_ >> 17;
                rng_state_ ^= rng_state_ << 5;
                const float n_im =
                    (static_cast<float>(rng_state_ & 0xFFFFu) / 32768.0f) - 1.0f;

                const float env_norm = envelope_[kd] / max_env;
                const float noise_gain = breath_scale * env_norm * synth_mag_[kd];
                vr += n_re * noise_gain;
                vi += n_im * noise_gain;
            }

            float band_gain = 0.0f;
            const float f_hz = static_cast<float>(kd) * bin_hz;
            if (f_hz >= kXoverHighHz) {
                band_gain = 1.0f;
            } else if (f_hz > kXoverLowHz) {
                const float mu = (f_hz - kXoverLowHz) / (kXoverHighHz - kXoverLowHz);
                band_gain = 0.5f * (1.0f - std::cos(kPi * mu));
            }
            const float g = sibilant_mix_ * band_gain;

            float yr = (1.0f - g) * vr + g * spec_re_[kd];
            float yi = (1.0f - g) * vi + g * spec_im_[kd];
            if (!std::isfinite(yr)) yr = 0.0f;   // NaN/Inf -> 0 guard
            if (!std::isfinite(yi)) yi = 0.0f;
            fft_re_[kd] = yr;
            fft_im_[kd] = yi;
        }

        // Real DC / Nyquist + conjugate-symmetric upper half (planar layout).
        fft_im_[0] = 0.0f;
        fft_im_[kHalf - 1] = 0.0f;
        for (int k = 1; k < kHalf - 1; ++k) {
            fft_re_[kFFT - k] = fft_re_[k];
            fft_im_[kFFT - k] = -fft_im_[k];
        }
        fft_run(true);

        // Safety clamp on the synthesized frame.
        for (int i = 0; i < kFFT; ++i) {
            float v = fft_re_[i];
            if (!std::isfinite(v)) v = 0.0f;
            if (v > 4.0f) v = 4.0f;
            if (v < -4.0f) v = -4.0f;
            scratch_time_[i] = v;
        }
    }

    void forward_real(const float* time, float* re513, float* im513) {
        for (int i = 0; i < kFFT; ++i) { fft_re_[i] = time[i]; fft_im_[i] = 0.0f; }
        fft_run(false);
        for (int k = 0; k < kHalf; ++k) {
            re513[k] = fft_re_[k];
            im513[k] = fft_im_[k];
        }
        // DC and Nyquist are purely real.
        im513[0] = 0.0f;
        im513[kHalf - 1] = 0.0f;
    }

    // Iterative radix-2 complex FFT, n = kFFT fixed. Inverse includes 1/N.
    void fft_run(bool inverse) {
        for (int i = 0; i < kFFT; ++i) {
            const int j = rev_[i];
            if (j > i) {
                std::swap(fft_re_[i], fft_re_[j]);
                std::swap(fft_im_[i], fft_im_[j]);
            }
        }
        for (int len = 2; len <= kFFT; len <<= 1) {
            const int half = len >> 1;
            const int step = kFFT / len;
            for (int i = 0; i < kFFT; i += len) {
                for (int j = 0; j < half; ++j) {
                    const int t = j * step;
                    const float wr = cos_tab_[t];
                    const float wi = inverse ? sin_tab_[t] : -sin_tab_[t];
                    const int a = i + j, b = a + half;
                    const float xr = fft_re_[b] * wr - fft_im_[b] * wi;
                    const float xi = fft_re_[b] * wi + fft_im_[b] * wr;
                    fft_re_[b] = fft_re_[a] - xr;
                    fft_im_[b] = fft_im_[a] - xi;
                    fft_re_[a] += xr;
                    fft_im_[a] += xi;
                }
            }
        }
        if (inverse) {
            const float s = 1.0f / kFFT;
            for (int i = 0; i < kFFT; ++i) { fft_re_[i] *= s; fft_im_[i] *= s; }
        }
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Standard ANode C-ABI (bound by ffi_base._bind_functions; no extended exports)
// ---------------------------------------------------------------------------

EXPORT void* create(void) {
    return static_cast<void*>(new (std::nothrow) VocalTransformerProcessor());
}

EXPORT void destroy(void* h) {
    delete static_cast<VocalTransformerProcessor*>(h);
}

EXPORT void set_samplerate(void* h, float samplerate) {
    if (h) static_cast<VocalTransformerProcessor*>(h)->set_samplerate(samplerate);
}

EXPORT void set_param(void* h, int id, float value) {
    if (h) static_cast<VocalTransformerProcessor*>(h)->set_param(id, value);
}

EXPORT void reset(void* h) {
    if (h) static_cast<VocalTransformerProcessor*>(h)->reset();
}

EXPORT void process(void* h, const float* in, float* out, int channels, int frames) {
    if (h) static_cast<VocalTransformerProcessor*>(h)->process(in, out, channels, frames);
}
