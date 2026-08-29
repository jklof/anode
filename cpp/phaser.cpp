// 6-Stage Stereo Allpass Phaser (ANode standard C-ABI).
//
// Cascade of 6 first-order allpass stages per channel with exponential LFO
// sweep, quadrature stereo spread, and tanh-saturated feedback.
//
// Allpass coefficient: a = (1 - tan(pi*fc/fs)) / (1 + tan(pi*fc/fs))
// Allpass stage: y[n] = a*x[n] + x[n-1] - a*y[n-1]
//
// set_param ids:
//   0: rate      (Hz, clamped [0.05, 8.0])
//   1: depth     (clamped [0, 1])
//   2: base_freq (Hz, clamped [50, 3000])
//   3: feedback  (clamped [0, 0.95])
//   4: spread    (clamped [0, 1])
//   5: mix       (clamped [0, 1])

#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

namespace {

constexpr int MAX_CHANNELS = 2;
constexpr int NUM_STAGES = 6;
constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 6.28318530717958647692f;

// tanh approximation for speed
inline float fast_tanh(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace

class PhaserProcessor {
public:
    PhaserProcessor()
        : sr_(48000.0f),
          rate_(0.5f), depth_(0.7f), base_freq_(400.0f),
          feedback_(0.5f), spread_(0.5f), mix_(0.5f),
          lfo_phase_(0.0f) {
        std::fill(&ap_state_[0][0], &ap_state_[0][0] + MAX_CHANNELS * NUM_STAGES, 0.0f);
        std::fill(&ap_xprev_[0][0], &ap_xprev_[0][0] + MAX_CHANNELS * NUM_STAGES, 0.0f);
        std::fill(&ap_yprev_[0][0], &ap_yprev_[0][0] + MAX_CHANNELS * NUM_STAGES, 0.0f);
    }

    void set_samplerate(float sr) {
        if (sr > 1.0f) sr_ = sr;
    }

    void set_param(int id, float v) {
        switch (id) {
            case 0: rate_ = clampf(v, 0.05f, 8.0f); break;
            case 1: depth_ = clampf(v, 0.0f, 1.0f); break;
            case 2: base_freq_ = clampf(v, 50.0f, 3000.0f); break;
            case 3: feedback_ = clampf(v, 0.0f, 0.95f); break;
            case 4: spread_ = clampf(v, 0.0f, 1.0f); break;
            case 5: mix_ = clampf(v, 0.0f, 1.0f); break;
            default: break;
        }
    }

    void reset() {
        std::fill(&ap_state_[0][0], &ap_state_[0][0] + MAX_CHANNELS * NUM_STAGES, 0.0f);
        std::fill(&ap_xprev_[0][0], &ap_xprev_[0][0] + MAX_CHANNELS * NUM_STAGES, 0.0f);
        std::fill(&ap_yprev_[0][0], &ap_yprev_[0][0] + MAX_CHANNELS * NUM_STAGES, 0.0f);
        lfo_phase_ = 0.0f;
    }

    void process(const float* in, float* out, int channels, int frames) {
        if (!in || !out || channels <= 0 || frames <= 0) return;
        if (channels > MAX_CHANNELS) channels = MAX_CHANNELS;

        const float lfo_step = rate_ / sr_;
        const float fc_min = 20.0f;
        const float fc_max = 0.45f * sr_;

        for (int i = 0; i < frames; ++i) {
            for (int c = 0; c < channels; ++c) {
                // Exponential LFO sweep with quadrature stereo spread
                float lfo_phase = lfo_phase_ + ((c & 1) ? spread_ * 0.25f : 0.0f);
                float lfo_val = std::sin(TWO_PI * lfo_phase);
                float fc = base_freq_ * std::pow(2.0f, depth_ * 3.0f * lfo_val);
                if (fc < fc_min) fc = fc_min;
                if (fc > fc_max) fc = fc_max;

                // Allpass coefficient
                float tan_val = std::tan(PI * fc / sr_);
                float a = (1.0f - tan_val) / (1.0f + tan_val);

                float x = 0.0f;
                if (channels == 1) {
                    x = in[i];
                } else {
                    x = in[static_cast<size_t>(c) * frames + i];
                }

                // Feedback injection with tanh saturation
                float fb_in = ap_yprev_[c][NUM_STAGES - 1];
                x = x + fast_tanh(feedback_ * fb_in);

                // Cascade 6 allpass stages
                // y[n] = a*x[n] - x[n-1] + a*y[n-1]  (allpass, -90 deg at fc per stage)
                float y_stage = x;
                for (int k = 0; k < NUM_STAGES; ++k) {
                    float xn = y_stage;
                    float yn = a * xn - ap_xprev_[c][k] + a * ap_yprev_[c][k];
                    ap_xprev_[c][k] = xn;
                    ap_yprev_[c][k] = yn;
                    y_stage = yn;
                }

                // Mix output
                float dry = (channels == 1) ? in[i] : in[static_cast<size_t>(c) * frames + i];
                out[static_cast<size_t>(c) * frames + i] = (1.0f - mix_) * dry + mix_ * y_stage;
            }
            lfo_phase_ += lfo_step;
            if (lfo_phase_ >= 1.0f) lfo_phase_ -= 1.0f;
        }
    }

private:
    float sr_;
    float rate_, depth_, base_freq_, feedback_, spread_, mix_;
    float lfo_phase_;

    // Allpass state: [channel][stage]
    float ap_xprev_[MAX_CHANNELS][NUM_STAGES];
    float ap_yprev_[MAX_CHANNELS][NUM_STAGES];
    float ap_state_[MAX_CHANNELS][NUM_STAGES];
};

extern "C" {

EXPORT void* create() {
    return new (std::nothrow) PhaserProcessor();
}

EXPORT void destroy(void* handle) {
    if (handle) delete static_cast<PhaserProcessor*>(handle);
}

EXPORT void set_samplerate(void* handle, float sr) {
    if (handle) static_cast<PhaserProcessor*>(handle)->set_samplerate(sr);
}

EXPORT void set_param(void* handle, int id, float val) {
    if (handle) static_cast<PhaserProcessor*>(handle)->set_param(id, val);
}

EXPORT void reset(void* handle) {
    if (handle) static_cast<PhaserProcessor*>(handle)->reset();
}

EXPORT void process(void* handle, const float* in, float* out, int channels, int frames) {
    if (handle) static_cast<PhaserProcessor*>(handle)->process(in, out, channels, frames);
}

} // extern "C"