// SignalsmithVocal native processor (ANode standard C-ABI).
//
// Sub-band pitch/formant shifter built on Signalsmith Stretch v1.3.2
// (MIT, header-only C++11; fetched via CMake FetchContent, which also pulls
// signalsmith-linear for FFTs):
//   input ring -> SignalsmithStretch::process (equal in/out block sizes,
//                 i.e. pitch-shift at 1x rate, no time-stretch)
//         -> output FIFO position -> latency-aligned dry delay + mix
//
// Division of responsibilities:
//   Signalsmith (upstream header): sub-band transient-aware phase locking,
//     pitch transpose, formant shift/compensation, internal latency.
//   This wrapper: planar pointer arrays, dry-delay ring for comb-free mix,
//     mono handling, bypass early-out, reset/latency C-ABI contract.
//
// Allocation policy (AGENTS.md §4, relaxed rule): C++ may allocate in bounded
// fashion. The stretcher owns its pre-sized vectors after presetDefault() /
// configure() (create/set_samplerate only); per-block process() reuses them.
// The wrapper's dry ring and wet scratch are sized once in configure().
// No disk/network/UI work happens here.
//
// Latency: L = inputLatency() + outputLatency() after presetDefault()
// (~0.12 s block + 0.03 s interval at 48 kHz, queried at runtime, not
// hardcoded). mix <= 0 is a bit-exact memcpy bypass (zero latency).

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstddef>

#include "signalsmith-stretch.h"

#include "anode_export.h"

namespace {

constexpr int   kMaxChannels = 2;
constexpr int   kRingSize = 32768;     // dry-delay history (power of 2)
constexpr int   kRingMask = kRingSize - 1;
constexpr float kFormantBaseHz = 160.0f;  // vocal mid-register guess (cf.
                                          // upstream 200/sampleRate example)
constexpr float kGenderStPerUnit = 3.5f;  // formant semitones per gender unit

// Param IDs (must match plugins/signalsmith_vocal.py PARAM_MAP).
enum ParamId {
    kPitchShift = 0,
    kFormantShift = 1,
    kGenderMorph = 2,
    kTonalityLimitHz = 3,
    kMix = 4,
    kLatencyMode = 5
};

// Operating modes: Studio keeps the default preset; Live uses a shorter
// block/interval for monitor-friendly latency (see configure()).
enum SigLatencyMode {
    kSigStudio = 0,
    kSigLive = 1
};

class SignalsmithVocalProcessor {
public:
    SignalsmithVocalProcessor()
        : sr_(48000.0f), pitch_st_(0.0f), formant_st_(0.0f),
          gender_morph_(0.0f), tonality_hz_(0.0f),
          mix_(1.0f), prev_mix_(1.0f), latency_mode_(kSigStudio), latency_(0) {
        configure();
        apply_params();
    }

    void set_samplerate(float sr) {
        if (sr <= 0.0f || sr == sr_) return;
        sr_ = sr;
        configure();
        apply_params();
    }

    void set_param(int id, float v) {
        switch (id) {
            case kPitchShift:
                pitch_st_ = std::max(-24.0f, std::min(24.0f, v));
                push_transpose();
                break;
            case kFormantShift:
                formant_st_ = std::max(-24.0f, std::min(24.0f, v));
                push_formant();
                break;
            case kGenderMorph:
                gender_morph_ = std::max(-1.0f, std::min(1.0f, v));
                push_formant();
                break;
            case kTonalityLimitHz:
                tonality_hz_ = std::max(0.0f, std::min(8000.0f, v));
                push_transpose();
                break;
            case kMix: {
                mix_ = std::max(0.0f, std::min(1.0f, v));
                // Bypass early-out is zero-latency raw audio while the wet
                // path is latency-delayed: crossing the boundary would pop,
                // so restart both paths cleanly (mirrors world_transformer).
                if ((prev_mix_ <= 0.0f) != (mix_ <= 0.0f)) reset_transient();
                prev_mix_ = mix_;
                break;
            }
            case kLatencyMode: {
                const int m = (v > 0.5f) ? kSigLive : kSigStudio;
                if (m != latency_mode_) {
                    latency_mode_ = m;
                    // New block/interval geometry: rebuild the stretcher and
                    // clear both paths so the dry-read step cannot pop.
                    configure();
                    apply_params();
                }
                break;
            }
            default: break;
        }
    }

    int latency_samples() const { return latency_; }

    void reset() {
        reset_transient();
        prev_mix_ = mix_;
    }

    void process(float* in, float* out, int channels, int frames) {
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

        const int chs = (channels == 1) ? kMaxChannels : channels;

        // Wet scratch: sized for the engine block; grown once if a larger
        // block ever arrives (never steady-state).
        if (static_cast<int>(wet_.size()) < kMaxChannels * frames)
            wet_.assign(static_cast<size_t>(kMaxChannels * frames), 0.0f);

        // Planar pointer arrays for the stretcher. Mono input is read by
        // both channels from the same array (read-only sharing is safe;
        // only in==out aliasing is forbidden, and those never alias here).
        float* in_ptrs[kMaxChannels] = {
            in, (channels == 1) ? in : in + frames
        };
        float* out_ptrs[kMaxChannels] = { wet_.data(), wet_.data() + frames };

        // Push this block into the dry-delay history (absolute-indexed).
        for (int i = 0; i < frames; ++i) {
            float l, r;
            if (channels == 1) {
                l = r = in[i];
            } else {
                l = in[i];
                r = in[frames + i];
            }
            dry_[0][total_in_ & kRingMask] = l;
            dry_[1][total_in_ & kRingMask] = r;
            ++total_in_;
        }

        stretch_.process(in_ptrs, frames, out_ptrs, frames);

        // Emit latency-aligned output: wet block N represents input stream
        // time N*frames - latency_ (the stretcher's output lags its input
        // by inputLatency()+outputLatency()), so the dry sample paired with
        // wet sample i is the input captured latency_ samples ago.
        for (int c = 0; c < chs; ++c) {
            float* out_ch = out + c * frames;
            const float* wet_ch = (c == 0) ? out_ptrs[0] : out_ptrs[1];
            for (int i = 0; i < frames; ++i) {
                const long long p = total_out_ + i - latency_;
                float dry = 0.0f;
                if (p >= 0) dry = dry_[c][p & kRingMask];
                const float wet = wet_ch[i];
                const float y = (1.0f - mix_) * dry + mix_ * wet;
                float s = std::isfinite(y) ? y : 0.0f;
                if (s > 4.0f) s = 4.0f;
                if (s < -4.0f) s = -4.0f;
                out_ch[i] = s;
            }
        }
        total_out_ += frames;

        // Mono-duplicated inputs must produce bit-identical stereo outputs.
        if (channels == 2 &&
            std::memcmp(in, in + frames, static_cast<size_t>(frames) * sizeof(float)) == 0) {
            std::memcpy(out + frames, out,
                        static_cast<size_t>(frames) * sizeof(float));
        }
        // Zero any unused output channels (anti-ghosting, defensive).
        for (int c = chs; c < kMaxChannels; ++c)
            std::memset(out + c * frames, 0, static_cast<size_t>(frames) * sizeof(float));
    }

private:
    void configure() {
        if (latency_mode_ == kSigLive) {
            // 40 ms block / 10 ms interval for monitor-friendly latency;
            // the reported latency is queried below, not assumed.
            stretch_.configure(kMaxChannels, static_cast<int>(sr_ * 0.04),
                               static_cast<int>(sr_ * 0.01));
        } else {
            stretch_.presetDefault(kMaxChannels, sr_);
        }
        stretch_.setFormantBase(kFormantBaseHz / sr_);
        latency_ = stretch_.inputLatency() + stretch_.outputLatency();
        if (latency_ < 0) latency_ = 0;
        dry_[0].assign(kRingSize, 0.0f);
        dry_[1].assign(kRingSize, 0.0f);
        wet_.assign(static_cast<size_t>(kMaxChannels * 512), 0.0f);
        total_in_ = 0;
        total_out_ = 0;
    }

    void apply_params() {
        push_transpose();
        push_formant();
    }

    void push_transpose() {
        // Tonality limit is normalized against the sample rate per the
        // upstream API (e.g. 8000/48000); 0 disables it.
        const float tl = (tonality_hz_ > 0.0f) ? tonality_hz_ / sr_ : 0.0f;
        stretch_.setTransposeSemitones(pitch_st_, tl);
    }

    void push_formant() {
        const float eff = formant_st_ + kGenderStPerUnit * gender_morph_;
        // compensatePitch=true keeps the formant shift absolute instead of
        // letting the pitch shift drag formants along (no chipmunk effect).
        stretch_.setFormantSemitones(eff, true);
    }

    void reset_transient() {
        stretch_.reset();
        std::fill(dry_[0].begin(), dry_[0].end(), 0.0f);
        std::fill(dry_[1].begin(), dry_[1].end(), 0.0f);
        std::fill(wet_.begin(), wet_.end(), 0.0f);
        total_in_ = 0;
        total_out_ = 0;
    }

    float sr_;
    float pitch_st_, formant_st_, gender_morph_, tonality_hz_;
    float mix_, prev_mix_;
    int latency_mode_;
    int latency_;
    signalsmith::stretch::SignalsmithStretch<float> stretch_;
    std::vector<float> dry_[kMaxChannels];
    std::vector<float> wet_;
    long long total_in_ = 0;
    long long total_out_ = 0;
};

SignalsmithVocalProcessor* self_of(void* h) {
    return static_cast<SignalsmithVocalProcessor*>(h);
}

}  // namespace

EXPORT void* create() {
    try {
        return new SignalsmithVocalProcessor();
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

EXPORT int get_latency_samples(void* handle) {
    if (!handle) return 0;
    return self_of(handle)->latency_samples();
}
