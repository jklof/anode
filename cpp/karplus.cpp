// Karplus-Strong physical modeling string synthesizer (ANode standard C-ABI).
//
// Excites a delay line with a filtered white-noise burst on rising-edge
// trigger. Fractional delay is read with linear interpolation and a 1-pole
// lowpass in the feedback loop gives natural high-frequency decay. Delay
// length is compensated for the filter phase lag and interpolation delay so
// the musical pitch tracks f0 accurately.
//
// set_param ids:
//   0: freq       (Hz, clamped [20, 2000])
//   1: damping    (clamped [0, 0.99])
//   2: brightness (clamped [0, 1])
//   3: decay      (clamped [0.8, 1])

#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

namespace {

constexpr int MAX_CHANNELS = 2;
constexpr int DELAY_CAPACITY = 4800;

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace

class KarplusProcessor {
public:
    KarplusProcessor()
        : sr_(48000.0f),
          freq_(220.0f), damping_(0.5f), brightness_(0.8f), decay_(0.99f),
          last_trig_val_(0.0f) {
        rng_state_ = 0x9E3779B9;
        std::fill(&delay_[0][0], &delay_[0][0] + MAX_CHANNELS * DELAY_CAPACITY, 0.0f);
        reset();
    }

    void set_samplerate(float sr) {
        if (sr > 1.0f) sr_ = sr;
    }

    void set_param(int id, float v) {
        switch (id) {
            case 0: freq_ = clampf(v, 20.0f, 2000.0f); break;
            case 1: damping_ = clampf(v, 0.0f, 0.99f); break;
            case 2: brightness_ = clampf(v, 0.0f, 1.0f); break;
            case 3: decay_ = clampf(v, 0.8f, 1.0f); break;
            default: break;
        }
    }

    void reset() {
        std::fill(&delay_[0][0], &delay_[0][0] + MAX_CHANNELS * DELAY_CAPACITY, 0.0f);
        for (int c = 0; c < MAX_CHANNELS; ++c) {
            state_[c] = 0.0f;
            write_pos_[c] = 0;
            excite_remaining_[c] = 0;
            lp_state_[c] = 0.0f;
        }
        last_trig_val_ = 0.0f;
    }

    void process(const float* trigger_in, float* out, int channels, int frames) {
        if (!trigger_in || !out || channels <= 0 || frames <= 0) return;
        if (channels > MAX_CHANNELS) channels = MAX_CHANNELS;

        // Compensated tuning: effective_delay = fs/f0 - D_filter - 0.5
        const float d_filter = damping_ / (1.0f - damping_ + 0.05f);
        const float eff_delay = (sr_ / freq_) - d_filter - 0.5f;
        const int pluck_len = std::max(1, std::min(static_cast<int>(eff_delay), DELAY_CAPACITY));
        // One-pole coefficient for the excitation lowpass: brightness 1 -> very
        // bright (a ~ 0.05), brightness 0 -> very dark (a ~ 0.95).
        const float excite_a = 0.95f - 0.9f * brightness_;

        for (int c = 0; c < channels; ++c) {
            const float* trig = trigger_in + static_cast<size_t>(c) * frames;
            float* y = out + static_cast<size_t>(c) * frames;
            float local_last = last_trig_val_;
            int excite_remaining = excite_remaining_[c];
            float lp = lp_state_[c];

            for (int i = 0; i < frames; ++i) {
                const float trig_val = trig[i];
                const bool edge = (trig_val > 0.0f) && (local_last <= 0.0f);
                if (edge) {
                    // Start a new pluck: excite the next L samples through the
                    // write path with a filtered noise burst scaled by velocity.
                    excite_remaining = pluck_len;
                    lp = 0.0f;
                    // Reset the noise seed per pluck so identical trigger
                    // signals produce identical channels (mono->stereo dup).
                    rng_state_ = 0x9E3779B9;
                }
                local_last = trig_val;

                // Fractional delay read with linear interpolation
                float read_pos = static_cast<float>(write_pos_[c]) - eff_delay;
                while (read_pos < 0.0f) read_pos += static_cast<float>(DELAY_CAPACITY);
                int idx0 = static_cast<int>(read_pos) % DELAY_CAPACITY;
                int idx1 = (idx0 + 1) % DELAY_CAPACITY;
                float frac = read_pos - static_cast<float>(idx0);
                float delayed = delay_[c][idx0] * (1.0f - frac) + delay_[c][idx1] * frac;

                // 1-pole lowpass in feedback path
                state_[c] = (1.0f - damping_) * delayed + damping_ * state_[c];
                float fb = state_[c] * decay_;

                // Excitation writes through the same write position the feedback
                // uses, so the burst is read L samples later and never overwritten.
                float write_val;
                if (excite_remaining > 0) {
                    const float noise = noise_gen_();
                    lp = lp * excite_a + noise * (1.0f - excite_a);
                    write_val = lp * trig_val;
                    excite_remaining--;
                } else {
                    write_val = fb;
                }
                delay_[c][write_pos_[c]] = write_val;
                write_pos_[c] = (write_pos_[c] + 1) % DELAY_CAPACITY;

                y[i] = delayed;
            }
            excite_remaining_[c] = excite_remaining;
            lp_state_[c] = lp;
        }
        if (frames > 0) {
            last_trig_val_ = trigger_in[static_cast<size_t>(channels - 1) * frames + (frames - 1)];
        }
    }

private:
    float noise_gen_() {
        uint32_t x = rng_state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rng_state_ = x;
        return static_cast<float>(static_cast<int32_t>(x)) / 2147483648.0f;
    }

    float sr_;
    float freq_, damping_, brightness_, decay_;
    float last_trig_val_;

    float delay_[MAX_CHANNELS][DELAY_CAPACITY];
    float state_[MAX_CHANNELS];
    int write_pos_[MAX_CHANNELS];
    int excite_remaining_[MAX_CHANNELS];
    float lp_state_[MAX_CHANNELS];
    uint32_t rng_state_;
};

extern "C" {

EXPORT void* create() {
    return new (std::nothrow) KarplusProcessor();
}

EXPORT void destroy(void* handle) {
    if (handle) delete static_cast<KarplusProcessor*>(handle);
}

EXPORT void set_samplerate(void* handle, float sr) {
    if (handle) static_cast<KarplusProcessor*>(handle)->set_samplerate(sr);
}

EXPORT void set_param(void* handle, int id, float val) {
    if (handle) static_cast<KarplusProcessor*>(handle)->set_param(id, val);
}

EXPORT void reset(void* handle) {
    if (handle) static_cast<KarplusProcessor*>(handle)->reset();
}

EXPORT void process(void* handle, const float* trigger_in, float* out, int channels, int frames) {
    if (handle) static_cast<KarplusProcessor*>(handle)->process(trigger_in, out, channels, frames);
}

} // extern "C"