// StudioVocalTransformer native processor.
//
// Real-time vocal pitch correction + TD-PSOLA pitch shifting with a
// single-timeline overlap-add core.
//
// Signal path:
//   input -> F0 tracking -> single-timeline TD-PSOLA grains -> OLA
//           normalization -> gentle spectral coloration ->
//           latency-aligned dry/wet mix.
//
// Design notes (and what the previous implementation got wrong):
//
// 1. Single shared timeline. Source (analysis) and synthesis marks live on
//    the SAME absolute sample clock. For each synthesis mark at time T we
//    copy a source grain from *near T* in the input history. Successive
//    synthesis marks advance by Ts = T0/ratio while the search centre
//    advances with them, so upward shifts (Ts < T0) naturally REUSE the same
//    source period several times and downward shifts (Ts > T0) naturally
//    SKIP periods. The lag between input and output stays locked at the
//    fixed emission latency instead of diverging at 512*(ratio-1) samples
//    per block (the old code kept an independent source_cursor_ advancing
//    by T0 per grain while the target advanced by T0/ratio, which made
//    upward shifts consume more input than arrives, stall grain generation,
//    and silently fall back to dry audio - e.g. +12 st produced no shift
//    at all and +3 st produced only ~1.3 st).
//
// 2. Reuse is allowed. The old code rejected any source mark <= the
//    previous one, which explicitly forbade the reuse that upward shifting
//    requires. We only forbid going strictly backwards; equality (reuse)
//    is the correct behaviour for ratio > 1.
//
// 3. Polarity-consistent GCI search. The old search maximised |x|, whose
//    peaks repeat every T0/2 (positive AND negative lobes). With synthesis
//    spacing Ts = T0/ratio that alternates polarity grain-to-grain and
//    partially cancels in the overlap. We maximise x (positive lobe only),
//    whose peaks repeat every T0, so overlapping grains stay in phase.
//
// 4. Timeline stays in sync through unvoiced/bypassed regions. When no
//    grains are generated the synthesis cursor is snapped to the emission
//    frontier (total_received - kLatency) instead of stalling, so returning
//    voicing never faces a catch-up burst of hundreds of grains in one
//    audio callback.
//
// 5. Pitch shifting here is resampling-based (each grain is read with stride
//    `ratio`), so formants move with pitch (classic/vintage character). The
//    formant/gender controls add a conservative broad tilt on top; they do
//    not relocate individual LPC poles (deliberately - that was the unstable
//    part of an older LPC design).
//
// Standard ANode C-ABI. Zero steady-state heap allocation: every buffer is
// a fixed member array.

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

constexpr int kMaxChannels = 2;
constexpr int kRingSize = 8192;
constexpr int kRingMask = kRingSize - 1;
constexpr int kPitchDecim = 4;
constexpr int kPitchBufSize = 1024;
constexpr int kPitchMinLag = 15;   // 800 Hz @ 12 kHz
constexpr int kPitchMaxLag = 240;  // 50 Hz @ 12 kHz
constexpr int kMaxGrainHalf = 800;
constexpr long long kLatency = 768;
constexpr int kMaxGrainsPerBlock = 64;  // RT burst guard for catch-up
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;

struct Biquad {
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
    float a1 = 0.0f, a2 = 0.0f;
    float z1 = 0.0f, z2 = 0.0f;

    void reset() { z1 = z2 = 0.0f; }

    void set_lowpass(float fc, float q, float sr) {
        fc = std::max(20.0f, std::min(0.45f * sr, fc));
        const float w = kTwoPi * fc / sr;
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

class StudioVocalProcessor {
public:
    StudioVocalProcessor()
        : sr_(48000.0f),
          pitch_st_(0.0f), formant_st_(0.0f), gender_morph_(0.0f),
          breathiness_(0.0f), sibilant_mix_(0.85f), mix_(1.0f),
          prev_mix_(1.0f), correction_enable_(1.0f),
          scale_root_(0), scale_mask_(0xFFF),
          retune_speed_ms_(20.0f), vibrato_depth_(0.0f),
          vibrato_rate_(5.5f), midi_mode_(0.0f),
          target_midi_note_(-1.0f), detected_f0_(0.0f),
          current_pitch_semitone_(60.0f),
          target_smoothed_semitone_(60.0f),
          vibrato_phase_(0.0f),
          last_accepted_f0_(0.0f), jump_pending_f0_(0.0f),
          jump_confirm_(0), voicing_hangover_(0) {
        aa_filter_.set_lowpass(1200.0f, 0.7071f, sr_);
        reset();
    }

    void set_samplerate(float sr) {
        if (sr > 1000.0f && sr != sr_) {
            sr_ = sr;
            aa_filter_.set_lowpass(1200.0f, 0.7071f, sr_);
            reset_transient();
        }
    }

    void set_param(int id, float v) {
        switch (id) {
            case 0: correction_enable_ = v > 0.5f ? 1.0f : 0.0f; break;
            case 1: scale_root_ = std::max(0, std::min(11, static_cast<int>(v))); break;
            case 2: scale_mask_ = static_cast<int>(v); break;
            case 3: retune_speed_ms_ = std::max(0.0f, std::min(100.0f, v)); break;
            case 4: pitch_st_ = std::max(-24.0f, std::min(24.0f, v)); break;
            case 5: formant_st_ = std::max(-24.0f, std::min(24.0f, v)); break;
            case 6: gender_morph_ = std::max(-1.0f, std::min(1.0f, v)); break;
            case 7: vibrato_depth_ = std::max(0.0f, std::min(2.0f, v)); break;
            case 8: vibrato_rate_ = std::max(2.0f, std::min(9.0f, v)); break;
            case 9: breathiness_ = std::max(0.0f, std::min(1.0f, v)); break;
            case 10: sibilant_mix_ = std::max(0.0f, std::min(1.0f, v)); break;
            case 11:
                mix_ = std::max(0.0f, std::min(1.0f, v));
                if ((prev_mix_ <= 0.0f) != (mix_ <= 0.0f))
                    reset_transient();
                prev_mix_ = mix_;
                break;
            case 12: midi_mode_ = v > 0.5f ? 1.0f : 0.0f; break;
            case 13: target_midi_note_ = v; break;
            default: break;
        }
    }

    void reset() {
        reset_transient();
        prev_mix_ = mix_;
        target_smoothed_semitone_ = 60.0f;
        current_pitch_semitone_ = 60.0f;
        vibrato_phase_ = 0.0f;
    }

    void process(const float* in, float* out, int channels, int frames) {
        if (!in || !out || frames <= 0) return;
        if (channels < 1) channels = 1;
        if (channels > kMaxChannels) channels = kMaxChannels;

        if (mix_ <= 0.0f) {
            const int chs = (channels == 1) ? kMaxChannels : channels;
            for (int c = 0; c < chs; ++c) {
                const float* src = (channels == 1) ? in : in + c * frames;
                std::memcpy(out + c * frames, src,
                            static_cast<size_t>(frames) * sizeof(float));
            }
            return;
        }

        track_pitch_and_retune(in, frames);

        float total_shift = pitch_st_;
        if (correction_enable_ > 0.5f && detected_f0_ > 50.0f) {
            const float correction =
                target_smoothed_semitone_ - current_pitch_semitone_;
            total_shift += std::max(-12.0f, std::min(12.0f, correction));
        }

        if (vibrato_depth_ > 0.001f) {
            vibrato_phase_ += kTwoPi * vibrato_rate_ *
                              (static_cast<float>(frames) / sr_);
            vibrato_phase_ = std::fmod(vibrato_phase_, kTwoPi);
            total_shift += vibrato_depth_ * std::sin(vibrato_phase_);
        }

        total_shift = std::max(-24.0f, std::min(24.0f, total_shift));
        const float ratio = std::pow(2.0f, total_shift / 12.0f);
        const bool transform_active =
            std::fabs(total_shift) > 0.01f ||
            std::fabs(formant_st_) > 0.01f ||
            std::fabs(gender_morph_) > 0.01f ||
            breathiness_ > 0.001f;

        const bool voiced = detected_f0_ > 50.0f;
        const float voiced_target = voiced ? 1.0f : 0.0f;

        const int chs = (channels == 1) ? kMaxChannels : channels;
        for (int c = 0; c < chs; ++c) {
            const float* in_ch = (channels == 1) ? in : in + c * frames;
            float* out_ch = out + c * frames;

            const long long write_base = total_received_[c];
            for (int i = 0; i < frames; ++i)
                in_ring_[c][(write_base + i) & kRingMask] = in_ch[i];
            total_received_[c] += frames;

            if (transform_active && voiced && std::isfinite(ratio) && ratio > 0.05f) {
                synthesize_grains(c, ratio);
                voicing_smooth_[c] +=
                    0.25f * (voiced_target - voicing_smooth_[c]);
            } else {
                // No grains this block (neutral bypass or unvoiced): keep the
                // synthesis cursor locked to the emission frontier so a later
                // return to voicing never faces a catch-up burst, and clear
                // any stale source-mark guard.
                next_target_[c] =
                    static_cast<double>(total_received_[c] - kLatency);
                last_source_mark_[c] = -1;
                voicing_smooth_[c] +=
                    0.35f * (voiced_target - voicing_smooth_[c]);
            }

            // Emit latency-aligned OLA. Weight normalization is deliberate: the
            // pitch period and therefore the Hann spacing can change every block.
            const long long read_base = total_received_[c] - frames - kLatency;
            for (int i = 0; i < frames; ++i) {
                const long long pos = read_base + i;
                if (pos < 0) {
                    out_ch[i] = 0.0f;
                    continue;
                }

                const int idx = static_cast<int>(pos & kRingMask);
                float wet = ola_ring_[c][idx];
                const float weight = ola_weight_[c][idx];
                ola_ring_[c][idx] = 0.0f;
                ola_weight_[c][idx] = 0.0f;

                if (weight > 1e-4f)
                    wet /= weight;
                else
                    wet = in_ring_[c][idx];

                // Unvoiced consonants should remain close to the original.
                // The crossfade is smooth enough to avoid a hard V/UV boundary.
                const float v = std::max(0.0f, std::min(1.0f, voicing_smooth_[c]));
                float transformed = wet;

                // Conservative broad spectral coloration. We intentionally do not
                // use high-Q formant poles here; those were the failure-prone part
                // of the old LPC design. Positive formant/gender values brighten
                // the upper vocal band; negative values darken it.
                transformed = color_sample(c, transformed);

                const float dry = in_ring_[c][idx];
                float wet_mix = v * transformed + (1.0f - v) * dry;

                // Sibilant preservation is only a gain toward dry on unvoiced
                // material; never add dry to a voiced vowel (comb-filter risk).
                const float sib = (1.0f - v) * sibilant_mix_;
                wet_mix = (1.0f - sib) * wet_mix + sib * dry;

                float y = (1.0f - mix_) * dry + mix_ * wet_mix;
                if (!std::isfinite(y)) y = 0.0f;
                y = std::max(-1.0f, std::min(1.0f, y));
                out_ch[i] = y;
            }
        }

        for (int c = chs; c < kMaxChannels; ++c)
            std::memset(out + c * frames, 0,
                        static_cast<size_t>(frames) * sizeof(float));
    }

private:
    float sr_;
    float pitch_st_, formant_st_, gender_morph_;
    float breathiness_, sibilant_mix_, mix_, prev_mix_;
    float correction_enable_;
    int scale_root_, scale_mask_;
    float retune_speed_ms_, vibrato_depth_, vibrato_rate_;
    float midi_mode_, target_midi_note_;
    float detected_f0_, current_pitch_semitone_, target_smoothed_semitone_;
    float vibrato_phase_;

    Biquad aa_filter_;
    float pitch_downsample_buf_[kPitchBufSize];
    float nsdf_[kPitchMaxLag];

    float in_ring_[kMaxChannels][kRingSize];
    float ola_ring_[kMaxChannels][kRingSize];
    float ola_weight_[kMaxChannels][kRingSize];

    long long total_received_[kMaxChannels];
    double next_target_[kMaxChannels];
    long long last_source_mark_[kMaxChannels];
    float voicing_smooth_[kMaxChannels];

    float formant_lp_[kMaxChannels];
    float formant_hp_[kMaxChannels];
    float air_lp_[kMaxChannels];
    float air_bp_[kMaxChannels];
    unsigned int rng_state_;

    float last_accepted_f0_, jump_pending_f0_;
    int jump_confirm_, voicing_hangover_;

    void reset_transient() {
        for (int c = 0; c < kMaxChannels; ++c) {
            std::memset(in_ring_[c], 0, sizeof(in_ring_[c]));
            std::memset(ola_ring_[c], 0, sizeof(ola_ring_[c]));
            std::memset(ola_weight_[c], 0, sizeof(ola_weight_[c]));
            total_received_[c] = 0;
            next_target_[c] = 0.0;
            last_source_mark_[c] = -1;
            voicing_smooth_[c] = 0.0f;
            formant_lp_[c] = 0.0f;
            formant_hp_[c] = 0.0f;
            air_lp_[c] = 0.0f;
            air_bp_[c] = 0.0f;
        }
        std::memset(pitch_downsample_buf_, 0, sizeof(pitch_downsample_buf_));
        std::memset(nsdf_, 0, sizeof(nsdf_));
        aa_filter_.reset();
        detected_f0_ = 0.0f;
        last_accepted_f0_ = 0.0f;
        jump_pending_f0_ = 0.0f;
        jump_confirm_ = 0;
        voicing_hangover_ = 0;
        rng_state_ = 0x1D872B41u;
    }

    void track_pitch_and_retune(const float* in, int frames) {
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
        const int start = kPitchBufSize - kPitchMaxLag - n;

        float energy = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float x = pitch_downsample_buf_[start + j];
            energy += x * x;
        }
        const float rms = std::sqrt(energy / static_cast<float>(n));

        if (rms < 0.0015f) {
            detected_f0_ = 0.0f;
            last_accepted_f0_ = 0.0f;
            voicing_hangover_ = 0;
            return;
        }

        for (int tau = kPitchMinLag; tau < kPitchMaxLag; ++tau) {
            float num = 0.0f;
            float den = 1e-9f;
            for (int j = 0; j < n; ++j) {
                const float x = pitch_downsample_buf_[start + j];
                const float y = pitch_downsample_buf_[start + tau + j];
                num += 2.0f * x * y;
                den += x * x + y * y;
            }
            nsdf_[tau] = num / den;
        }

        float r_max = 0.0f;
        for (int tau = kPitchMinLag + 1; tau < kPitchMaxLag - 1; ++tau) {
            if (nsdf_[tau] > 0.0f &&
                nsdf_[tau] > nsdf_[tau - 1] &&
                nsdf_[tau] >= nsdf_[tau + 1])
                r_max = std::max(r_max, nsdf_[tau]);
        }

        int best_tau = -1;
        if (r_max >= 0.42f) {
            const float threshold = r_max * 0.85f;
            for (int tau = kPitchMinLag + 1; tau < kPitchMaxLag - 1; ++tau) {
                if (nsdf_[tau] > 0.0f &&
                    nsdf_[tau] > nsdf_[tau - 1] &&
                    nsdf_[tau] >= nsdf_[tau + 1] &&
                    nsdf_[tau] >= threshold) {
                    best_tau = tau;
                    break;
                }
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
            const float raw_f0 = (sr_ / static_cast<float>(kPitchDecim)) / tau;

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

        if (detected_f0_ > 50.0f) {
            current_pitch_semitone_ =
                69.0f + 12.0f * std::log2(detected_f0_ / 440.0f);

            float target = current_pitch_semitone_;
            if (midi_mode_ > 0.5f && target_midi_note_ >= 0.0f)
                target = target_midi_note_;
            else
                target = snap_to_scale(current_pitch_semitone_);

            if (retune_speed_ms_ <= 0.1f) {
                target_smoothed_semitone_ = target;
            } else {
                const float alpha = 1.0f - std::exp(
                    -static_cast<float>(frames) /
                    (retune_speed_ms_ * 0.001f * sr_));
                target_smoothed_semitone_ +=
                    alpha * (target - target_smoothed_semitone_);
            }
        }
    }

    float snap_to_scale(float note) const {
        if (scale_mask_ == 0) return note;
        const int rounded = static_cast<int>(std::round(note));
        int best = rounded;
        int min_dist = 100;
        for (int d = -6; d <= 6; ++d) {
            const int cand = rounded + d;
            int pc = (cand - scale_root_) % 12;
            if (pc < 0) pc += 12;
            if ((scale_mask_ & (1 << (11 - pc))) != 0) {
                const int dist = std::abs(cand - rounded);
                if (dist < min_dist) {
                    min_dist = dist;
                    best = cand;
                }
            }
        }
        return static_cast<float>(best);
    }

    long long find_source_mark(int c, long long expected, int radius) const {
        const long long lo = std::max(0LL, expected - static_cast<long long>(radius));
        const long long hi = std::min(total_received_[c] - 1,
                                      expected + static_cast<long long>(radius));

        // Polarity-consistent GCI search: maximise x (positive lobe only).
        // Positive peaks repeat every T0; |x| peaks repeat every T0/2 and
        // alternate polarity grain-to-grain, which partially cancels in the
        // overlap for non-unity ratios.
        long long best = expected;
        float best_score = -1.0e30f;
        for (long long p = lo; p <= hi; ++p) {
            const float x = in_ring_[c][p & kRingMask];
            if (x > best_score) {
                best_score = x;
                best = p;
            }
        }
        return best;
    }

    void synthesize_grains(int c, float ratio) {
        float f0 = std::max(50.0f, std::min(800.0f, detected_f0_));
        const float T0f = sr_ / f0;
        const int source_half = std::max(16, std::min(kMaxGrainHalf,
                                                       static_cast<int>(std::round(T0f))));
        const float target_half_f =
            std::max(16.0f, std::min(static_cast<float>(kMaxGrainHalf),
                                     T0f / ratio));
        const int target_half = static_cast<int>(std::round(target_half_f));

        // Single shared timeline: the search centre IS the synthesis time.
        // Upward shifts reuse source periods, downward shifts skip them, and
        // the input/output lag stays locked at kLatency.
        const double frontier =
            static_cast<double>(total_received_[c] - kLatency);
        int grains = 0;
        while (next_target_[c] < frontier && grains < kMaxGrainsPerBlock) {
            const long long expected = static_cast<long long>(
                std::llround(next_target_[c]));
            // +/-50% of the period guarantees the nearest positive peak is
            // inside the window no matter the sub-period phase.
            const int radius = std::max(4, source_half / 2);
            long long mark = find_source_mark(c, expected, radius);

            // Never go backwards; equality (grain reuse for ratio > 1) is
            // not only allowed but required for upward shifts.
            if (last_source_mark_[c] >= 0 && mark < last_source_mark_[c])
                mark = last_source_mark_[c];

            // The resampled source window spans target_half*ratio ~= T0
            // samples around the mark; it must be fully received.
            const double src_span =
                static_cast<double>(target_half) * static_cast<double>(ratio);
            const double src_lo = static_cast<double>(mark) - src_span;
            const double src_hi = static_cast<double>(mark) + src_span;
            const double step =
                static_cast<double>(T0f) / static_cast<double>(ratio);
            if (src_lo < 0.0) {
                // Startup pre-history: samples before time 0 will never
                // exist. Skip this grain (it would be silence anyway) and
                // keep advancing instead of stalling forever on a target
                // that can never become available.
                next_target_[c] += step;
                ++grains;
                continue;
            }
            if (src_hi + 4.0 > static_cast<double>(total_received_[c]))
                break;

            const long long target =
                static_cast<long long>(std::llround(next_target_[c]));

            // The source waveform is resampled around the source GCI. Each
            // grain covers ~one source period re-sampled into one target
            // period: the pitch change with the vocal-tract spectrum carried
            // along (classic character; formant/gender tilt is applied later
            // in color_sample()).
            for (int n = -target_half; n <= target_half; ++n) {
                const float w = 0.5f * (1.0f +
                    std::cos(kPi * static_cast<float>(n) /
                             static_cast<float>(target_half)));

                const double src_pos =
                    static_cast<double>(mark) + static_cast<double>(n) * ratio;
                const long long i0 = static_cast<long long>(std::floor(src_pos));
                const float frac =
                    static_cast<float>(src_pos - static_cast<double>(i0));

                float x = 0.0f;
                if (i0 - 1 >= 0 && i0 + 2 < total_received_[c]) {
                    const float ym1 = in_ring_[c][(i0 - 1) & kRingMask];
                    const float y0  = in_ring_[c][i0 & kRingMask];
                    const float y1  = in_ring_[c][(i0 + 1) & kRingMask];
                    const float y2  = in_ring_[c][(i0 + 2) & kRingMask];

                    const float c0 = y0;
                    const float c1 = 0.5f * (y1 - ym1);
                    const float c2 = ym1 - 2.5f * y0 + 2.0f * y1 -
                                     0.5f * y2;
                    const float c3 = 0.5f * (y2 - ym1) +
                                     1.5f * (y0 - y1);
                    x = ((c3 * frac + c2) * frac + c1) * frac + c0;
                } else if (i0 >= 0 && i0 < total_received_[c]) {
                    x = in_ring_[c][i0 & kRingMask];
                }

                const long long wp = target + n;
                const int idx = static_cast<int>(wp & kRingMask);
                ola_ring_[c][idx] += x * w;
                ola_weight_[c][idx] += w;
            }

            last_source_mark_[c] = mark;
            next_target_[c] += static_cast<double>(T0f) / static_cast<double>(ratio);
            ++grains;
        }
    }

    float color_sample(int c, float x) {
        // Very low-cost broad tilt, intentionally capped. This is a tonal
        // correction, not a fake high-Q "formant resonator".
        const float amount =
            std::max(-0.38f, std::min(0.38f,
                0.0125f * formant_st_ + 0.08f * gender_morph_));

        // One-pole low/high split. The pole is deliberately slow and never
        // approaches instability. Positive amount = slightly brighter.
        const float alpha = 0.035f;
        formant_lp_[c] += alpha * (x - formant_lp_[c]);
        const float hp = x - formant_lp_[c];
        formant_hp_[c] += alpha * (hp - formant_hp_[c]);

        float y = x + amount * (hp - 0.35f * formant_lp_[c]);
        if (breathiness_ > 0.001f) {
            // Actual air component, but deliberately far below the old static
            // level. Generate deterministic white noise, high-pass it, then
            // low-pass the result into a broad ~2-8 kHz air band. Peak amplitude
            // at 1.0 is only about -54 dBFS, so normal preset values cannot turn
            // the processor into a hiss generator.
            rng_state_ ^= rng_state_ << 13;
            rng_state_ ^= rng_state_ >> 17;
            rng_state_ ^= rng_state_ << 5;
            const float noise =
                static_cast<float>(rng_state_ & 0xFFFFu) / 32768.0f - 1.0f;
            air_lp_[c] += 0.08f * (noise - air_lp_[c]);
            const float air_hp = noise - air_lp_[c];
            air_bp_[c] += 0.12f * (air_hp - air_bp_[c]);
            y += breathiness_ * 0.002f * air_bp_[c];
        }
        return y;
    }
};

} // namespace

EXPORT void* create(void) {
    return static_cast<void*>(new (std::nothrow) StudioVocalProcessor());
}

EXPORT void destroy(void* h) {
    delete static_cast<StudioVocalProcessor*>(h);
}

EXPORT void set_samplerate(void* h, float samplerate) {
    if (h) static_cast<StudioVocalProcessor*>(h)->set_samplerate(samplerate);
}

EXPORT void set_param(void* h, int id, float value) {
    if (h) static_cast<StudioVocalProcessor*>(h)->set_param(id, value);
}

EXPORT void reset(void* h) {
    if (h) static_cast<StudioVocalProcessor*>(h)->reset();
}

EXPORT void process(void* h, const float* in, float* out, int channels, int frames) {
    if (h) static_cast<StudioVocalProcessor*>(h)->process(in, out, channels, frames);
}
