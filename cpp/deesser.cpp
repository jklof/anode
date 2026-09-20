// DeEsser native processor (ANode standard C-ABI).
//
// Split-band dynamic EQ (dbx-902 topology), zero added latency:
//
//   in ──┬──→ LR4 crossover ──┬──→ LF ──────────→ [+] ──→ out
//        │                    └──→ HF ──→ [+] ──→ [× g] ──┘
//        │                         ▲
//        └──→ detector = HF band ──┘ (peak follower → gain computer)
//
// The Linkwitz-Riley 4 crossover (two cascaded Butterworth biquads per
// side, RBJ cookbook) sums flat, so with g = 1 the output is bit-near the
// input. Only the HF band is ever attenuated, leaving vowel formants and
// low harmonics untouched. Detection runs on an independent bandpass
// (Q = 0.5, ~2.5 octaves) centered at the same frequency, so ess energy
// below the split point still triggers; it is linked across channels (max of the two
// bands) so the stereo image never shifts under reduction.
//
// `listen` solos an audition signal to the output for tuning: listen_mode 0
// (Delta) plays only the removed sibilance hf*(1-g) — silence when idle — and
// listen_mode 1 (Sibilant Band) solos the HF crossover band (the legacy
// behavior). `auto_threshold` switches the gain computer to level-independent
// relative contrast: detector energy vs. a vowel-body reference tracked by a
// dedicated 3 kHz lowpass follower (a 120 ms release keeps the reference from
// collapsing between glottal pulses). A noise-floor gate disables the
// relative mode when the body reference is below ~-60 dBFS, so silence never
// triggers. `mix <= 0` is a bit-exact memcpy bypass. All state is fixed
// member data; reset() clears biquad delay lines, both followers, and the
// GR estimate.

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstddef>
#include <new>

#include "anode_export.h"

namespace {

constexpr int kMaxChannels = 2;
constexpr float kPi = 3.14159265358979323846f;
// Vowel-body reference lowpass corner and auto-threshold noise-floor gate
// (~-60 dBFS peak): below this the relative contrast mode never reduces.
constexpr float kRefLpHz = 3000.0f;
constexpr float kRefGateLin = 1.0e-3f;

// Param IDs (must match plugins/deesser.py PARAM_MAP).
enum ParamId {
    kFrequency = 0,
    kThresholdDb = 1,
    kDepthDb = 2,
    kAttackMs = 3,
    kReleaseMs = 4,
    kListen = 5,
    kMix = 6,
    kListenMode = 7,      // 0: Delta (removed sibilance), 1: Sibilant Band
    kAutoThreshold = 8,   // 0: Absolute dBFS threshold, 1: Relative contrast
};

struct Biquad {
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
    float a1 = 0.0f, a2 = 0.0f;
    float z1 = 0.0f, z2 = 0.0f;

    void reset() { z1 = z2 = 0.0f; }

    void set_butterworth_lowpass(float fc, float sr) {
        fc = std::max(20.0f, std::min(0.45f * sr, fc));
        const float w = 2.0f * kPi * fc / sr;
        const float c = std::cos(w);
        const float s = std::sin(w);
        const float alpha = s * 0.70710678118654752440f;  // Q = 1/sqrt(2)
        const float a0 = 1.0f + alpha;
        b0 = ((1.0f - c) * 0.5f) / a0;
        b1 = (1.0f - c) / a0;
        b2 = b0;
        a1 = (-2.0f * c) / a0;
        a2 = (1.0f - alpha) / a0;
    }

    void set_butterworth_highpass(float fc, float sr) {
        fc = std::max(20.0f, std::min(0.45f * sr, fc));
        const float w = 2.0f * kPi * fc / sr;
        const float c = std::cos(w);
        const float s = std::sin(w);
        const float alpha = s * 0.70710678118654752440f;
        const float a0 = 1.0f + alpha;
        b0 = ((1.0f + c) * 0.5f) / a0;
        b1 = (-(1.0f + c)) / a0;
        b2 = b0;
        a1 = (-2.0f * c) / a0;
        a2 = (1.0f - alpha) / a0;
    }

    // Constant-peak-gain bandpass for the ess detector (Q = 0.5 spans the
    // full sibilant range around the center frequency). Kept independent
    // of the crossover so ess energy below the split point still triggers.
    void set_detector_bandpass(float fc, float sr) {
        fc = std::max(20.0f, std::min(0.45f * sr, fc));
        const float w = 2.0f * kPi * fc / sr;
        const float c = std::cos(w);
        const float s = std::sin(w);
        const float alpha = s;  // Q = 0.5 (~2.5 octaves: wide ess coverage)
        const float a0 = 1.0f + alpha;
        b0 = alpha / a0;
        b1 = 0.0f;
        b2 = -alpha / a0;
        a1 = (-2.0f * c) / a0;
        a2 = (1.0f - alpha) / a0;
    }

    inline float process(float x) {
        const float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

class DeEsserProcessor {
public:
    DeEsserProcessor()
        : sr_(48000.0f), freq_(6500.0f), thresh_db_(-18.0f), depth_db_(6.0f),
          att_ms_(1.0f), rel_ms_(60.0f), listen_(false), mix_(1.0f),
          listen_mode_(0), auto_threshold_(false),
          env_(0.0f), ref_env_(0.0f), gr_db_(0.0f), gr_block_min_(0.0f) {
        recalc();
    }

    void set_samplerate(float sr) {
        if (sr > 1.0f && sr != sr_) {
            sr_ = sr;
            recalc();
        }
    }

    void set_param(int id, float v) {
        switch (id) {
            case kFrequency:
                freq_ = std::max(3000.0f, std::min(9000.0f, v));
                recalc_filters();
                break;
            case kThresholdDb:
                thresh_db_ = std::max(-40.0f, std::min(0.0f, v));
                break;
            case kDepthDb:
                depth_db_ = std::max(0.0f, std::min(12.0f, v));
                break;
            case kAttackMs:
                att_ms_ = std::max(0.1f, std::min(10.0f, v));
                recalc_ballistics();
                break;
            case kReleaseMs:
                rel_ms_ = std::max(10.0f, std::min(300.0f, v));
                recalc_ballistics();
                break;
            case kListen:
                listen_ = v > 0.5f;
                break;
            case kMix:
                mix_ = std::max(0.0f, std::min(1.0f, v));
                break;
            case kListenMode:
                listen_mode_ = (v > 0.5f) ? 1 : 0;
                break;
            case kAutoThreshold:
                auto_threshold_ = v > 0.5f;
                break;
            default: break;
        }
    }

    float gr_db() const { return gr_block_min_; }

    void reset() {
        for (int c = 0; c < kMaxChannels; ++c) {
            lp1_[c].reset();
            lp2_[c].reset();
            hp1_[c].reset();
            hp2_[c].reset();
            det_[c].reset();
        }
        ref_lp_[0].reset();
        ref_lp_[1].reset();
        env_ = 0.0f;
        ref_env_ = 0.0f;
        gr_db_ = 0.0f;
        gr_block_min_ = 0.0f;
    }

    void process(const float* in, float* out, int channels, int frames) {
        if (!in || !out || frames <= 0) return;
        if (channels < 1) channels = 1;
        if (channels > kMaxChannels) channels = kMaxChannels;

        // Dry bypass: bit-exact memcpy early-out (zero latency).
        // Honors the mono->stereo duplication contract of the normal path.
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

        // Metering reports the block minimum (most negative) reduction so a
        // brief mid-block spike is not masked by end-of-block release.
        gr_block_min_ = 0.0f;

        for (int i = 0; i < frames; ++i) {
            // Linked detection: hottest detector-band energy across channels.
            float det = 0.0f;
            float hf[kMaxChannels];
            float lf[kMaxChannels];
            for (int c = 0; c < chs; ++c) {
                const float x = (channels == 1) ? in[i] : in[c * frames + i];
                const float lo = lp2_[c].process(lp1_[c].process(x));
                const float hi = hp2_[c].process(hp1_[c].process(x));
                lf[c] = lo;
                hf[c] = hi;
                const float a = std::fabs(det_[c].process(x));
                if (a > det) det = a;
            }

            // Peak follower ballistics on the detector.
            const float alpha = (det > env_) ? att_ : rel_;
            env_ += alpha * (det - env_);

            // Vowel-body reference: peak of the 3 kHz-lowpassed signal across
            // channels. Fast attack (shared with the detector) tracks onsets;
            // a slow 120 ms release keeps the reference from decaying to zero
            // between glottal pulses so relative contrast stays meaningful.
            float ref = 0.0f;
            for (int c = 0; c < chs; ++c) {
                const float a = std::fabs(ref_lp_[c].process(
                    (channels == 1) ? in[i] : in[c * frames + i]));
                if (a > ref) ref = a;
            }
            const float ref_alpha = (ref > ref_env_) ? att_ : ref_rel_;
            ref_env_ += ref_alpha * (ref - ref_env_);

            // Gain computer: fixed 3:1 ratio above threshold, clamped to depth.
            float target_gr = 0.0f;
            if (depth_db_ > 0.0f) {
                float over = 0.0f;
                if (auto_threshold_) {
                    // Relative contrast: detector energy vs. vowel-body
                    // reference. Level-independent — whisper-to-scream
                    // dynamics trigger the same. The user threshold acts as
                    // a contrast offset around parity (default -18 dB => 0 dB
                    // offset => reduce once ess exceeds the vowel body).
                    // Noise-floor gate: no reduction when the body reference
                    // is essentially silent (prevents idle self-triggering).
                    if (ref_env_ > kRefGateLin) {
                        const float ess_db = 20.0f * std::log10(env_ + 1e-9f);
                        const float body_db = 20.0f * std::log10(ref_env_ + 1e-9f);
                        const float rel_thresh = thresh_db_ + 18.0f;
                        over = (ess_db - body_db) - rel_thresh;
                    }
                } else {
                    const float env_db = 20.0f * std::log10(env_ + 1e-9f);
                    over = env_db - thresh_db_;
                }
                if (over > 0.0f) {
                    target_gr = -std::min(depth_db_, over * (2.0f / 3.0f));
                }
            }
            gr_db_ = target_gr;
            if (target_gr < gr_block_min_) gr_block_min_ = target_gr;
            const float g = std::pow(10.0f, gr_db_ / 20.0f);

            for (int c = 0; c < chs; ++c) {
                // Reconstructed dry carries the crossover phase; blending
                // dry/wet inside this domain keeps every mix value
                // comb-free (raw dry + phase-shifted wet would notch at fc).
                // mix = 0 is therefore magnitude-transparent rather than
                // bit-exact; bit-exact bypass is the mix <= 0 path above.
                float wet0, wet;
                if (listen_) {
                    if (listen_mode_ == 0) {
                        // DELTA AUDITION: only the removed HF energy —
                        // exactly silence when no reduction is applied.
                        wet0 = hf[c] * (1.0f - g);
                        wet = wet0;
                    } else {
                        // BAND AUDITION: solo the HF crossover band.
                        wet0 = hf[c];
                        wet = hf[c];
                    }
                } else {
                    wet0 = lf[c] + hf[c];
                    wet = lf[c] + hf[c] * g;
                }
                float y = (1.0f - mix_) * wet0 + mix_ * wet;
                out[c * frames + i] = std::isfinite(y) ? y : 0.0f;
            }
        }

        // Zero any unused output channels (anti-ghosting, defensive).
        for (int c = chs; c < kMaxChannels; ++c)
            std::memset(out + c * frames, 0, static_cast<size_t>(frames) * sizeof(float));
    }

private:
    void recalc() {
        recalc_filters();
        recalc_ballistics();
    }

    void recalc_filters() {
        for (int c = 0; c < kMaxChannels; ++c) {
            lp1_[c].set_butterworth_lowpass(freq_, sr_);
            lp2_[c].set_butterworth_lowpass(freq_, sr_);
            hp1_[c].set_butterworth_highpass(freq_, sr_);
            hp2_[c].set_butterworth_highpass(freq_, sr_);
            det_[c].set_detector_bandpass(freq_, sr_);
            // Vowel-body reference band: fixed 3 kHz lowpass (formants and
            // body), independent of the split frequency so the sibilant's own
            // low tail near the crossover never inflates the reference.
            ref_lp_[c].set_butterworth_lowpass(kRefLpHz, sr_);
        }
    }

    void recalc_ballistics() {
        att_ = 1.0f - std::exp(-1000.0f / (sr_ * att_ms_));
        rel_ = 1.0f - std::exp(-1000.0f / (sr_ * rel_ms_));
        // Vowel-body reference release: smooth 120 ms decay.
        ref_rel_ = 1.0f - std::exp(-1000.0f / (sr_ * 120.0f));
    }

    float sr_;
    float freq_, thresh_db_, depth_db_, att_ms_, rel_ms_;
    bool listen_;
    int listen_mode_;
    bool auto_threshold_;
    float mix_;
    float att_, rel_, ref_rel_;
    float env_, ref_env_, gr_db_, gr_block_min_;
    Biquad lp1_[kMaxChannels], lp2_[kMaxChannels];
    Biquad hp1_[kMaxChannels], hp2_[kMaxChannels];
    Biquad det_[kMaxChannels];
    Biquad ref_lp_[kMaxChannels];
};

DeEsserProcessor* self_of(void* h) {
    return static_cast<DeEsserProcessor*>(h);
}

}  // namespace

EXPORT void* create() {
    try {
        return new DeEsserProcessor();
    } catch (...) {
        return nullptr;
    }
}

EXPORT void destroy(void* handle) {
    delete self_of(handle);
}

EXPORT void set_samplerate(void* handle, float sr) {
    if (handle) self_of(handle)->set_samplerate(sr);
}

EXPORT void set_param(void* handle, int id, float value) {
    if (handle) self_of(handle)->set_param(id, value);
}

EXPORT void reset(void* handle) {
    if (handle) self_of(handle)->reset();
}

EXPORT void process(void* handle, float* in, float* out, int channels, int frames) {
    if (handle) self_of(handle)->process(in, out, channels, frames);
}

EXPORT float get_gr_db(void* handle) {
    if (!handle) return 0.0f;
    return self_of(handle)->gr_db();
}
