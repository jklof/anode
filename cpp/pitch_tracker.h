// Shared real-time pitch tracker (ANode header-only, C++17).
//
// Extracted from cpp/vocal_transformer.cpp track_pitch_and_retune():
// 4:1-decimated NSDF at ~12 kHz, 1.2 kHz anti-alias filter, continuity
// scoring, harmonic unwinding (H2-H6), parabolic interpolation,
// octave-jump confirmation guard, voicing hangover.
//
// Retune/scale/MIDI logic is deliberately NOT included here; callers map
// the returned F0 to their own parameter domain. Used by VocalTransformer
// (retune front end) and WorldVoiceTransformer (WORLD analysis F0).
//
// Zero steady-state heap allocation: fixed member arrays only. The double
// overload converts through a fixed stack buffer (single call for typical
// hop sizes; chunked for larger windows).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

class PitchTracker {
public:
    PitchTracker() {
        set_samplerate(48000.0f);
        reset();
    }

    void set_samplerate(float sr) {
        if (sr <= 0.0f) return;
        sr_ = sr;
        ds_rate_ = sr_ / static_cast<float>(kPitchDecim);
        aa_filter_.set_lowpass(1200.0f, 0.7071f, sr_);
        // 50-800 Hz bounds in the decimated domain; clamp to array limits.
        // max_lag_ is an EXCLUSIVE bound matching the historical inline
        // implementation (lags < 240, window start = BufSize - 240 - 512),
        // so behavior at 48 kHz is bit-identical to it.
        min_lag_ = std::max(2, static_cast<int>(ds_rate_ / 800.0f));
        max_lag_ = std::min(kPitchMaxLag, static_cast<int>(ds_rate_ / 50.0f));
        if (min_lag_ >= max_lag_) {
            min_lag_ = 15;
            max_lag_ = kPitchMaxLag;
        }
        reset();
    }

    void reset() {
        std::memset(pitch_downsample_buf_, 0, sizeof(pitch_downsample_buf_));
        std::memset(nsdf_, 0, sizeof(nsdf_));
        aa_filter_.reset();
        detected_f0_ = 0.0f;
        last_accepted_f0_ = 0.0f;
        jump_pending_f0_ = 0.0f;
        jump_confirm_ = 0;
        voicing_hangover_ = 0;
    }

    // Process host-SR samples; returns detected F0 in Hz (0.0 = unvoiced).
    float process(const float* in, int frames) {
        if (!in || frames <= 0) return detected_f0_;
        const int ds = frames / kPitchDecim;
        if (ds > 0 && ds < kPitchBufSize) {
            std::memmove(pitch_downsample_buf_,
                         pitch_downsample_buf_ + ds,
                         static_cast<size_t>(kPitchBufSize - ds) * sizeof(float));
            for (int i = 0; i < ds; ++i) {
                float y = 0.0f;
                for (int d = 0; d < kPitchDecim; ++d)
                    y = aa_filter_.process(in[i * kPitchDecim + d]);
                pitch_downsample_buf_[kPitchBufSize - ds + i] = y;
            }
        }

        constexpr int n = 512;
        const int start = kPitchBufSize - max_lag_ - n;

        float energy = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float x = pitch_downsample_buf_[start + j];
            energy += x * x;
        }
        const float rms = std::sqrt(energy / static_cast<float>(n));

        if (rms < 0.0015f) {
            detected_f0_ = 0.0f;
            last_accepted_f0_ = 0.0f;
            jump_pending_f0_ = 0.0f;
            jump_confirm_ = 0;
            voicing_hangover_ = 0;
            return detected_f0_;
        }

        for (int tau = min_lag_; tau < max_lag_; ++tau) {
            float num = 0.0f;
            // Seed with energy (== sum of x*x over this window): the loop
            // below then only needs the y*y half of the denominator.
            float den = energy + 1e-9f;
            for (int j = 0; j < n; ++j) {
                const float x = pitch_downsample_buf_[start + j];
                const float y = pitch_downsample_buf_[start + tau + j];
                num += 2.0f * x * y;
                den += y * y;
            }
            nsdf_[tau] = num / den;
        }

        float r_max = 0.0f;
        for (int tau = min_lag_ + 1; tau < max_lag_ - 1; ++tau) {
            if (nsdf_[tau] > 0.0f &&
                nsdf_[tau] > nsdf_[tau - 1] &&
                nsdf_[tau] >= nsdf_[tau + 1])
                r_max = std::max(r_max, nsdf_[tau]);
        }

        int best_tau = -1;
        if (r_max >= 0.42f) {
            if (last_accepted_f0_ > 50.0f) {
                // Continuity scoring (one-block Viterbi): prefer the peak
                // nearest last block's F0; penalise upward jumps hard.
                const float net = std::max(0.35f * r_max, 0.25f);
                float best_score = -1e30f;
                for (int tau = min_lag_ + 1; tau < max_lag_ - 1; ++tau) {
                    const float v = nsdf_[tau];
                    if (v <= 0.0f || v < net) continue;
                    if (!(v > nsdf_[tau - 1] && v >= nsdf_[tau + 1])) continue;
                    const float f = ds_rate_ / static_cast<float>(tau);
                    const float st = 12.0f * std::log2(f / last_accepted_f0_);
                    const float pen = st > 0.0f ? 0.030f * st : -0.012f * st;
                    const float score = v - pen;
                    if (score > best_score) {
                        best_score = score;
                        best_tau = tau;
                    }
                }
            }
            if (best_tau <= 0) {
                const float threshold = r_max * 0.85f;
                for (int tau = min_lag_ + 1; tau < max_lag_ - 1; ++tau) {
                    if (nsdf_[tau] > 0.0f &&
                        nsdf_[tau] > nsdf_[tau - 1] &&
                        nsdf_[tau] >= nsdf_[tau + 1] &&
                        nsdf_[tau] >= threshold) {
                        best_tau = tau;
                        break;
                    }
                }
            }
            // Harmonic unwinding (H2-H6): if an integer multiple of the
            // candidate is itself a clearly higher local peak, the
            // candidate was a harmonic — move toward the fundamental.
            for (int iter = 0; iter < 2 && best_tau > 0; ++iter) {
                bool moved = false;
                for (int m = 2; m <= 6; ++m) {
                    const int cand = best_tau * m;
                    if (cand - 1 < min_lag_ || cand + 1 >= max_lag_)
                        continue;
                    if (nsdf_[cand] > nsdf_[best_tau] + 0.04f &&
                        nsdf_[cand] >= nsdf_[cand - 1] &&
                        nsdf_[cand] >= nsdf_[cand + 1] &&
                        nsdf_[cand] >= 0.5f * r_max) {
                        best_tau = cand;
                        moved = true;
                        break;
                    }
                }
                if (!moved) break;
            }
        }

        if (best_tau > 0) {
            const float y0 = nsdf_[best_tau - 1];
            const float y1 = nsdf_[best_tau];
            const float y2 = nsdf_[best_tau + 1];
            float denom = y0 - 2.0f * y1 + y2;
            if (std::fabs(denom) < 1e-9f) denom = -1e-9f;
            float delta = 0.5f * (y0 - y2) / denom;
            delta = std::max(-0.5f, std::min(0.5f, delta));
            const float tau = static_cast<float>(best_tau) + delta;
            const float raw_f0 = ds_rate_ / tau;

            float accepted = raw_f0;
            if (last_accepted_f0_ > 0.0f) {
                const float jump_st =
                    std::fabs(12.0f * std::log2(raw_f0 / last_accepted_f0_));
                if (jump_st > 6.0f) {
                    if (jump_pending_f0_ > 0.0f &&
                        std::fabs(12.0f * std::log2(raw_f0 / jump_pending_f0_)) < 2.0f)
                        ++jump_confirm_;
                    else
                        jump_confirm_ = 1;

                    jump_pending_f0_ = raw_f0;
                    if (jump_confirm_ < 2)
                        accepted = last_accepted_f0_;
                } else {
                    jump_confirm_ = 0;
                    jump_pending_f0_ = 0.0f;
                }
            }

            last_accepted_f0_ = accepted;
            detected_f0_ = accepted;
            voicing_hangover_ = 4;
        } else if (voicing_hangover_ > 0 && last_accepted_f0_ > 0.0f) {
            --voicing_hangover_;
            detected_f0_ = last_accepted_f0_;
        } else {
            detected_f0_ = 0.0f;
            last_accepted_f0_ = 0.0f;
        }
        return detected_f0_;
    }

    // Zero-allocation double wrapper via a fixed stack buffer.
    float process(const double* in, int frames) {
        if (!in || frames <= 0) return detected_f0_;
        constexpr int kChunk = 512;
        float buf[kChunk];
        int remaining = frames;
        int offset = 0;
        float f0 = detected_f0_;
        while (remaining > 0) {
            int cur = std::min(remaining, kChunk);
            for (int i = 0; i < cur; ++i) buf[i] = static_cast<float>(in[offset + i]);
            f0 = process(buf, cur);
            offset += cur;
            remaining -= cur;
        }
        return f0;
    }

    float detected_f0() const { return detected_f0_; }

private:
    static constexpr int kPitchDecim = 4;
    static constexpr int kPitchBufSize = 1024;
    static constexpr int kPitchMaxLag = 240;

    struct Biquad {
        float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
        float a1 = 0.0f, a2 = 0.0f;
        float z1 = 0.0f, z2 = 0.0f;
        void reset() { z1 = z2 = 0.0f; }
        void set_lowpass(float fc, float q, float sr) {
            fc = std::max(20.0f, std::min(0.45f * sr, fc));
            const float w = 6.28318530717958647692f * fc / sr;
            const float c = std::cos(w);
            const float s = std::sin(w);
            const float alpha = s / (2.0f * q);
            const float a0 = 1.0f + alpha;
            b0 = ((1.0f - c) * 0.5f) / a0;
            b1 = (1.0f - c) / a0;
            b2 = b0;
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

    float sr_ = 48000.0f;
    float ds_rate_ = 12000.0f;
    int min_lag_ = 15;
    int max_lag_ = 240;

    Biquad aa_filter_;
    float pitch_downsample_buf_[kPitchBufSize];
    float nsdf_[kPitchMaxLag];

    float detected_f0_ = 0.0f;
    float last_accepted_f0_ = 0.0f;
    float jump_pending_f0_ = 0.0f;
    int jump_confirm_ = 0;
    int voicing_hangover_ = 0;
};
