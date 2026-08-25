#include <cmath>
#include <cstring>
#include <algorithm>
#include <new>

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

// ANode Biquad Filter (IIR) — Robert Bristow-Johnson Audio EQ Cookbook
// topologies, Direct Form II Transposed, double-precision state.
//
// set_param ids:
//   0: type (menu index 0..6)
//   1: cutoff (Hz, clamped to [20, min(20000, 0.45*sr)])
//   2: q     (clamped to >= 0.05)
//   3: gain_db (peaking/shelving)

namespace {

constexpr int MAX_CHANNELS = 2;
constexpr double PI = 3.14159265358979323846;

inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace

class BiquadProcessor {
public:
    BiquadProcessor()
        : sr_(48000.0), type_(0), cutoff_(1000.0), q_(0.70707), gain_db_(0.0),
          b0_(1.0), b1_(0.0), b2_(0.0), a1_(0.0), a2_(0.0),
          last_channels_(-1) {
        reset();
        design();
    }

    void set_param(int id, float value) {
        switch (id) {
            case 0: type_ = static_cast<int>(value); break;
            case 1: cutoff_ = value; break;
            case 2: q_ = value; break;
            case 3: gain_db_ = value; break;
            default: return;
        }
        design();
    }

    void set_samplerate(float samplerate) {
        if (samplerate > 1.0f) sr_ = samplerate;
        design();
    }

    void reset() {
        for (int c = 0; c < MAX_CHANNELS; ++c) {
            z_[c][0] = 0.0;
            z_[c][1] = 0.0;
        }
        last_channels_ = -1;
    }

    // Planar buffers: in/out are [Ch0 frames..., Ch1 frames...].
    void process(const float* in, float* out, int channels, int frames) {
        if (!in || !out || channels <= 0 || frames <= 0) return;
        if (channels > MAX_CHANNELS) channels = MAX_CHANNELS;

        // Channel-count change: stale DF2T state would ring old content.
        if (channels != last_channels_) {
            for (int c = 0; c < channels; ++c) { z_[c][0] = 0.0; z_[c][1] = 0.0; }
            last_channels_ = channels;
        }

        for (int c = 0; c < channels; ++c) {
            const float* x = in + static_cast<size_t>(c) * frames;
            float* y = out + static_cast<size_t>(c) * frames;
            double z1 = z_[c][0];
            double z2 = z_[c][1];
            for (int i = 0; i < frames; ++i) {
                const double xn = x[i];
                const double yn = b0_ * xn + z1;
                z1 = b1_ * xn - a1_ * yn + z2;
                z2 = b2_ * xn - a2_ * yn;
                y[i] = static_cast<float>(yn);
            }
            z_[c][0] = z1;
            z_[c][1] = z2;
        }
    }

private:
    void design() {
        const double f0 = clampd(cutoff_, 20.0,
                                 std::min(20000.0, 0.45 * sr_));
        const double q = std::max(static_cast<double>(q_), 0.05);
        const double w0 = 2.0 * PI * f0 / sr_;
        const double cosw = std::cos(w0);
        const double sinw = std::sin(w0);
        const double alpha = sinw / (2.0 * q);
        const double A = std::pow(10.0, gain_db_ / 40.0);

        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a0 = 1.0, a1 = 0.0, a2 = 0.0;

        switch (type_) {
            case 0:  // Low Pass
                b0 = (1.0 - cosw) / 2.0; b1 = 1.0 - cosw; b2 = b0;
                a0 = 1.0 + alpha; a1 = -2.0 * cosw; a2 = 1.0 - alpha;
                break;
            case 1:  // High Pass
                b0 = (1.0 + cosw) / 2.0; b1 = -(1.0 + cosw); b2 = b0;
                a0 = 1.0 + alpha; a1 = -2.0 * cosw; a2 = 1.0 - alpha;
                break;
            case 2:  // Band Pass (constant 0 dB peak gain)
                b0 = alpha; b1 = 0.0; b2 = -alpha;
                a0 = 1.0 + alpha; a1 = -2.0 * cosw; a2 = 1.0 - alpha;
                break;
            case 3:  // Notch
                b0 = 1.0; b1 = -2.0 * cosw; b2 = 1.0;
                a0 = 1.0 + alpha; a1 = -2.0 * cosw; a2 = 1.0 - alpha;
                break;
            case 4:  // Peaking EQ
                b0 = 1.0 + alpha * A; b1 = -2.0 * cosw; b2 = 1.0 - alpha * A;
                a0 = 1.0 + alpha / A; a1 = -2.0 * cosw; a2 = 1.0 - alpha / A;
                break;
            case 5: {  // Low Shelf
                const double beta = 2.0 * std::sqrt(A) * alpha;
                b0 = A * ((A + 1.0) - (A - 1.0) * cosw + beta);
                b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosw);
                b2 = A * ((A + 1.0) - (A - 1.0) * cosw - beta);
                a0 = (A + 1.0) + (A - 1.0) * cosw + beta;
                a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosw);
                a2 = (A + 1.0) + (A - 1.0) * cosw - beta;
                break;
            }
            default: {  // High Shelf
                const double beta = 2.0 * std::sqrt(A) * alpha;
                b0 = A * ((A + 1.0) + (A - 1.0) * cosw + beta);
                b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw);
                b2 = A * ((A + 1.0) + (A - 1.0) * cosw - beta);
                a0 = (A + 1.0) - (A - 1.0) * cosw + beta;
                a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw);
                a2 = (A + 1.0) - (A - 1.0) * cosw - beta;
                break;
            }
        }

        b0_ = b0 / a0;
        b1_ = b1 / a0;
        b2_ = b2 / a0;
        a1_ = a1 / a0;
        a2_ = a2 / a0;
    }

    double sr_;
    int type_;
    double cutoff_;
    double q_;
    double gain_db_;

    double b0_, b1_, b2_, a1_, a2_;
    double z_[MAX_CHANNELS][2];
    int last_channels_;
};

extern "C" {

EXPORT void* create() {
    return new (std::nothrow) BiquadProcessor();
}

EXPORT void destroy(void* handle) {
    if (handle) delete static_cast<BiquadProcessor*>(handle);
}

EXPORT void process(void* handle, float* in, float* out, int channels, int frames) {
    if (handle) static_cast<BiquadProcessor*>(handle)->process(in, out, channels, frames);
}

EXPORT void set_param(void* handle, int param_id, float value) {
    if (handle) static_cast<BiquadProcessor*>(handle)->set_param(param_id, value);
}

EXPORT void set_samplerate(void* handle, float samplerate) {
    if (handle) static_cast<BiquadProcessor*>(handle)->set_samplerate(samplerate);
}

EXPORT void reset(void* handle) {
    if (handle) static_cast<BiquadProcessor*>(handle)->reset();
}

} // extern "C"
