#include <cmath>
#include <cstring>
#include <algorithm>
#include <new>

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

// ANode Linear Phase EQ (FIR) — Hann-windowed sinc designs, overlap-save
// block convolution. Odd tap count (Type I symmetric) => integer group delay.
//
// set_param ids:
//   0: type (menu index: 0 Low Pass, 1 High Pass, 2 Band Pass, 3 Notch)
//   1: cutoff (Hz)
//   2: q     (bandwidth control for BP / Notch)
//
// The coefficient design runs here (not in Python) so param changes never
// cross the FFI boundary with a 255-float payload.

namespace {

constexpr int TAPS = 255;
constexpr int HIST = TAPS - 1;      // samples carried between blocks
constexpr int MAX_CHANNELS = 2;
constexpr int MAX_BLOCK = 4096;     // padded-row capacity (engine uses 512)
constexpr double PI = 3.14159265358979323846;

inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline double sinc(double x) {
    if (std::fabs(x) < 1e-12) return 1.0;
    return std::sin(PI * x) / (PI * x);
}

} // namespace

class FirEqProcessor {
public:
    FirEqProcessor()
        : sr_(48000.0), type_(0), cutoff_(1000.0), q_(1.0), last_channels_(-1) {
        reset();
        design();
    }

    void set_param(int id, float value) {
        switch (id) {
            case 0: type_ = static_cast<int>(value); break;
            case 1: cutoff_ = value; break;
            case 2: q_ = value; break;
            default: return;
        }
        design();
    }

    void set_samplerate(float samplerate) {
        if (samplerate > 1.0f && samplerate != sr_) {
            sr_ = samplerate;
            reset();
            design();
        }
    }

    void reset() {
        std::memset(hist_, 0, sizeof(hist_));
        last_channels_ = -1;
    }

    // Planar buffers: [Ch0 frames..., Ch1 frames...].
    //
    // Per block, each channel's [history | block] sequence is converted once
    // into a contiguous double row u. Because HIST == TAPS-1 the convolution
    // becomes y[i] = sum_p kernel[p] * u[i+p]; since the designed kernel is
    // symmetric (kernel[p] == kernel[M-p]), this folds to half the MACs:
    //   y[i] = sum_{p<127} kfold[p]*(u[i+p]+u[M-p offset]) + center*u[i+127]
    // 4-way unrolled accumulators break the FP dependency chain.
    void process(const float* in, float* out, int channels, int frames) {
        if (!in || !out || channels <= 0 || frames <= 0) return;
        if (channels > MAX_CHANNELS) channels = MAX_CHANNELS;
        if (frames > MAX_BLOCK) frames = MAX_BLOCK;

        if (channels != last_channels_) {
            for (int c = 0; c < channels; ++c)
                std::memset(hist_[c], 0, sizeof(float) * HIST);
            last_channels_ = channels;
        }

        constexpr int HALF = TAPS / 2;  // 127

        for (int c = 0; c < channels; ++c) {
            const float* x = in + static_cast<size_t>(c) * frames;
            float* y = out + static_cast<size_t>(c) * frames;
            double* u = u_[c];

            for (int j = 0; j < HIST; ++j) u[j] = hist_[c][j];
            for (int j = 0; j < frames; ++j) u[HIST + j] = x[j];

            for (int i = 0; i < frames; ++i) {
                const double* wc = u + i + HALF;  // window center element
                // Symmetry fold: pair samples equidistant from the window
                // center. y[i] = kf[0]*w[c] + sum_{d>=1} kf[d]*(w[c-d]+w[c+d])
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                int d = 1;
                for (; d + 4 <= HALF + 1; d += 4) {
                    s0 += kf_[d] * (wc[-d] + wc[d]);
                    s1 += kf_[d + 1] * (wc[-(d + 1)] + wc[d + 1]);
                    s2 += kf_[d + 2] * (wc[-(d + 2)] + wc[d + 2]);
                    s3 += kf_[d + 3] * (wc[-(d + 3)] + wc[d + 3]);
                }
                for (; d <= HALF; ++d)
                    s0 += kf_[d] * (wc[-d] + wc[d]);
                const double acc = s0 + ((s1 + s3) + s2) + kf_[0] * wc[0];
                y[i] = static_cast<float>(acc);
            }

            // BLOCK_SIZE (512) > HIST (254): the new history is simply the
            // tail of this block. Guard anyway for arbitrary frame counts.
            if (frames >= HIST) {
                std::memcpy(hist_[c], x + (frames - HIST), sizeof(float) * HIST);
            } else {
                std::memmove(hist_[c], hist_[c] + (HIST - frames), sizeof(float) * (HIST - frames));
                std::memcpy(hist_[c] + (HIST - frames), x, sizeof(float) * frames);
            }
        }
    }

private:
    double nyquist() const { return sr_ / 2.0; }

    void design() {
        constexpr int M = TAPS - 1;
        const double nyq = nyquist();
        double window[TAPS];
        for (int n = 0; n < TAPS; ++n)
            window[n] = 0.5 - 0.5 * std::cos(2.0 * PI * n / M);

        auto sum = [&](const double* h) {
            double s = 0.0;
            for (int n = 0; n < TAPS; ++n) s += h[n];
            return s;
        };

        auto make_lowpass = [&](double fc, double* h) {
            fc = clampd(fc, 20.0, nyq - 1.0);
            const double m = M / 2.0;
            const double fcn = fc / nyq;
            for (int n = 0; n < TAPS; ++n) h[n] = fcn * sinc(fcn * (n - m)) * window[n];
            const double s = sum(h);
            if (s != 0.0)
                for (int n = 0; n < TAPS; ++n) h[n] /= s;
        };

        double h[TAPS];

        switch (type_) {
            case 0:  // Low Pass
                make_lowpass(cutoff_, h);
                break;
            case 1: {  // High Pass (spectral inversion of LP)
                make_lowpass(cutoff_, h);
                for (int n = 0; n < TAPS; ++n) h[n] = -h[n];
                h[M / 2] += 1.0;  // delta at the shared linear-phase center
                break;
            }
            default: {
                const double bw = clampd(cutoff_ / std::max(static_cast<double>(q_), 1e-3),
                                         20.0, nyquist() - 2.0);
                const double lo = clampd(cutoff_ - bw / 2.0, 20.0, nyquist() - 2.0);
                double hi = std::max(cutoff_ + bw / 2.0, lo + 20.0);
                hi = clampd(hi, lo + 20.0, nyquist() - 1.0);

                double hp[TAPS], hl[TAPS];
                make_lowpass(hi, hp);
                make_lowpass(lo, hl);
                for (int n = 0; n < TAPS; ++n) h[n] = hp[n] - hl[n];

                if (type_ == 3) {  // Band Stop (Notch): spectral inversion of BP
                    for (int n = 0; n < TAPS; ++n) h[n] = -h[n];
                    h[M / 2] += 1.0;
                }
                break;
            }
        }

        // Fold for center-based symmetry convolution:
        //   kf[0]   = h[127]           (center tap)
        //   kf[d]   = h[127+d]         (paired taps, d = 1..127)
        // so y[i] = kf[0]*u[i+127] + sum_d kf[d]*(u[i+127-d] + u[i+127+d])
        kf_[0] = h[M / 2];
        for (int d = 1; d <= M / 2; ++d) kf_[d] = h[M / 2 + d];
    }

    double sr_;
    int type_;
    double cutoff_;
    double q_;

    static constexpr int HALF_TAPS = TAPS / 2 + 1;  // 128: center + 127 pairs
    double kf_[HALF_TAPS];
    float hist_[MAX_CHANNELS][HIST];
    double u_[MAX_CHANNELS][HIST + MAX_BLOCK];
    int last_channels_;
};

extern "C" {

EXPORT void* create() {
    return new (std::nothrow) FirEqProcessor();
}

EXPORT void destroy(void* handle) {
    if (handle) delete static_cast<FirEqProcessor*>(handle);
}

EXPORT void process(void* handle, float* in, float* out, int channels, int frames) {
    if (handle) static_cast<FirEqProcessor*>(handle)->process(in, out, channels, frames);
}

EXPORT void set_param(void* handle, int param_id, float value) {
    if (handle) static_cast<FirEqProcessor*>(handle)->set_param(param_id, value);
}

EXPORT void set_samplerate(void* handle, float samplerate) {
    if (handle) static_cast<FirEqProcessor*>(handle)->set_samplerate(samplerate);
}

EXPORT void reset(void* handle) {
    if (handle) static_cast<FirEqProcessor*>(handle)->reset();
}

} // extern "C"
