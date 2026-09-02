// ChorusFlanger native processor (ANode standard C-ABI).
//
// Stereo modulated delay line with dual quadrature LFOs (phi_L = 0 deg,
// phi_R = 90 deg * spread). Fractional delay via 4-point Catmull-Rom
// Hermite interpolation on a per-channel ring buffer. Feedback path is
// soft-saturated with tanh so any setting stays bounded.
//
// Buffer capacity covers MAX_DELAY_MS (base 20 + depth 8 + margin);
// allocated once in set_samplerate (called once at construction time,
// off the audio thread) and never resized afterwards.

#include <cmath>
#include <vector>

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

namespace {

constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kMaxDelayMs = 30.0f;

inline float hermite_read(const std::vector<float>& buf, int cap,
                          float read_pos) {
    // read_pos in [0, cap); Catmull-Rom over buf[floor-1]..buf[floor+2],
    // centered on buf[floor] so an integer read position returns buf[floor]
    // exactly (no 1-sample offset).
    int i0 = static_cast<int>(read_pos);
    float frac = read_pos - static_cast<float>(i0);
    const int im1 = (i0 - 1 + cap) % cap;
    const int i1 = (i0 + 1) % cap;
    const int i2 = (i0 + 2) % cap;
    const float ym1 = buf[im1], y0 = buf[i0], y1 = buf[i1], y2 = buf[i2];
    const float c1 = 0.5f * (y1 - ym1);
    const float c2 = ym1 - 2.5f * y0 + 2.0f * y1 - 0.5f * y2;
    const float c3 = 0.5f * (y2 - ym1) + 1.5f * (y0 - y1);
    return ((c3 * frac + c2) * frac + c1) * frac + y0;
}

} // namespace

class ChorusProcessor {
public:
    ChorusProcessor()
        : sr_(48000.0f), cap_(0), write_pos_(0), phase_(0.0f),
          rate_(0.6f), depth_ms_(3.0f), base_ms_(5.0f), feedback_(0.3f),
          spread_(1.0f), mix_(0.5f), mode_(0) {}

    void set_samplerate(float sr) {
        sr_ = sr;
        cap_ = static_cast<int>(std::ceil(sr * kMaxDelayMs / 1000.0f)) + 4;
        ring_[0].assign(static_cast<size_t>(cap_), 0.0f);
        ring_[1].assign(static_cast<size_t>(cap_), 0.0f);
        write_pos_ = 0;
        phase_ = 0.0f;
    }

    void set_param(int id, float v) {
        switch (id) {
            case 0: rate_ = v; break;
            case 1: depth_ms_ = v; break;
            case 2: base_ms_ = v; break;
            case 3: feedback_ = v; break;
            case 4: spread_ = v; break;
            case 5: mix_ = v; break;
            case 6: mode_ = static_cast<int>(v); break;
        }
    }

    void reset() {
        if (cap_ > 0) {
            ring_[0].assign(static_cast<size_t>(cap_), 0.0f);
            ring_[1].assign(static_cast<size_t>(cap_), 0.0f);
        }
        write_pos_ = 0;
        phase_ = 0.0f;
    }

    void process(const float* in, float* out, int channels, int frames) {
        if (cap_ <= 0) return;
        const int chs = channels < 2 ? 2 : channels;
        const float d_min = 1.0f;   // keep the read head behind the write head
        const float lfo_step = rate_ / sr_;

        for (int i = 0; i < frames; ++i) {
            for (int c = 0; c < chs; ++c) {
                float x = 0.0f;
                if (channels == 1) {
                    x = in[i];                      // mono input duplicated
                } else if (c < channels) {
                    x = in[c * frames + i];
                }
                auto& ring = ring_[c & 1];

                // Modulated delay in samples (quadrature on channel 1)
                const float lfo_phase =
                    phase_ + ((c & 1) ? spread_ * 0.25f : 0.0f);   // cycles
                const float lfo = std::sinf(kTwoPi * lfo_phase);
                float d_ms = base_ms_ + depth_ms_ * (0.5f + 0.5f * lfo);
                if (d_ms < 0.0f) d_ms = 0.0f;
                float d = d_ms * 0.001f * sr_;
                if (d < d_min) d = d_min;
                if (d > static_cast<float>(cap_ - 4)) d = static_cast<float>(cap_ - 4);

                // Read BEFORE pushing the feedback-injected sample
                float read_pos = static_cast<float>(write_pos_) - d;
                while (read_pos < 0.0f) read_pos += static_cast<float>(cap_);
                // Float rounding can land exactly on cap (e.g. -1e-5f + cap
                // rounds up to cap), which would read buf[cap] out of bounds;
                // the true position is then ~0 (frac ~0), so clamp to 0.
                if (read_pos >= static_cast<float>(cap_)) read_pos = 0.0f;
                const float wet = hermite_read(ring, cap_, read_pos);

                float feed_in;
                if (mode_ == 1) {
                    // Vibrato mode: no feedback, 100% wet
                    feed_in = x;
                    ring[static_cast<size_t>(write_pos_)] = feed_in;
                    out[c * frames + i] = wet;
                } else {
                    // Normal chorus/flanger mode
                    feed_in = x + std::tanh(feedback_ * wet);
                    ring[static_cast<size_t>(write_pos_)] = feed_in;
                    out[c * frames + i] = (1.0f - mix_) * x + mix_ * wet;
                }
            }
            write_pos_ = (write_pos_ + 1) % cap_;
            phase_ += lfo_step;
            if (phase_ >= 1.0f) phase_ -= 1.0f;
        }
    }

private:
    float sr_;
    int cap_;
    int write_pos_;
    float phase_;                 // LFO phase in cycles [0, 1)
    float rate_, depth_ms_, base_ms_, feedback_, spread_, mix_;
    int mode_;                    // 0 = normal, 1 = vibrato
    std::vector<float> ring_[2];
};

extern "C" {

EXPORT void* create() {
    return new ChorusProcessor();
}

EXPORT void destroy(void* h) {
    delete static_cast<ChorusProcessor*>(h);
}

EXPORT void set_samplerate(void* h, float sr) {
    static_cast<ChorusProcessor*>(h)->set_samplerate(sr);
}

EXPORT void set_param(void* h, int id, float v) {
    static_cast<ChorusProcessor*>(h)->set_param(id, v);
}

EXPORT void reset(void* h) {
    static_cast<ChorusProcessor*>(h)->reset();
}

EXPORT void process(void* h, const float* in, float* out, int channels, int frames) {
    static_cast<ChorusProcessor*>(h)->process(in, out, channels, frames);
}

} // extern "C"
