// EnvelopeFollower native processor (ANode standard C-ABI).
//
// One-pole ballistics with attack/release coefficient switching per sample.
// Detection runs over the ACTUAL channel count passed in (planar layout,
// in[c * frames + i]) — never a hardcoded channel count.
// Gate output has hysteresis: opens at thresh, closes below thresh * 0.5.

#include <cmath>

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

class EnvelopeProcessor {
public:
    EnvelopeProcessor()
        : sr_(48000.0f), env_(0.0f), gate_open_(false), mode_(0),
          gain_(1.0f), thresh_(0.1f), att_ms_(10.0f), rel_ms_(100.0f) {
        recalc();
    }

    void set_samplerate(float sr) { sr_ = sr; recalc(); }

    void set_param(int id, float v) {
        switch (id) {
            case 0: mode_ = (v > 0.5f) ? 1 : 0; break;            // Peak / RMS
            case 1: att_ms_ = v < 0.1f ? 0.1f : v; recalc(); break;
            case 2: rel_ms_ = v < 1.0f ? 1.0f : v; recalc(); break;
            case 3: gain_ = v; break;
            case 4: thresh_ = v; break;
        }
    }

    void reset() { env_ = 0.0f; gate_open_ = false; }

    void process(const float* in, float* cv, float* gate, int channels, int frames) {
        for (int i = 0; i < frames; ++i) {
            float det = 0.0f;
            if (mode_ == 0) {                       // Peak
                for (int c = 0; c < channels; ++c) {
                    float a = std::fabs(in[c * frames + i]);
                    if (a > det) det = a;
                }
            } else {                                // RMS
                float s = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    float x = in[c * frames + i];
                    s += x * x;
                }
                det = std::sqrt(s / static_cast<float>(channels));
            }
            det *= gain_;

            const float alpha = (det > env_) ? att_ : rel_;
            env_ += alpha * (det - env_);

            // Gate with hysteresis (chatter suppression)
            if (!gate_open_) {
                if (env_ >= thresh_) gate_open_ = true;
            } else if (env_ < thresh_ * 0.5f) {
                gate_open_ = false;
            }

            cv[i] = env_;                           // range [0, gain], unclamped
            gate[i] = gate_open_ ? 1.0f : 0.0f;
        }
    }

private:
    void recalc() {
        att_ = 1.0f - std::exp(-1000.0f / (sr_ * att_ms_));
        rel_ = 1.0f - std::exp(-1000.0f / (sr_ * rel_ms_));
    }

    float sr_, env_, gain_, thresh_, att_ms_, rel_ms_, att_, rel_;
    int mode_;
    bool gate_open_;
};

extern "C" {

EXPORT void* create() {
    return new EnvelopeProcessor();
}

EXPORT void destroy(void* h) {
    delete static_cast<EnvelopeProcessor*>(h);
}

EXPORT void set_samplerate(void* h, float sr) {
    static_cast<EnvelopeProcessor*>(h)->set_samplerate(sr);
}

EXPORT void set_param(void* h, int id, float v) {
    static_cast<EnvelopeProcessor*>(h)->set_param(id, v);
}

EXPORT void reset(void* h) {
    static_cast<EnvelopeProcessor*>(h)->reset();
}

// Extended export — Python side MUST annotate restype/argtypes explicitly.
EXPORT void process(void* h, const float* in, float* cv, float* gate,
             int channels, int frames) {
    static_cast<EnvelopeProcessor*>(h)->process(in, cv, gate, channels, frames);
}

} // extern "C"
