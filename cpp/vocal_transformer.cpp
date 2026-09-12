// VocalTransformer native processor (ANode standard C-ABI).
//
// Studio-grade vocal pitch / formant / gender transformer:
//   - resampled-analysis-frame pitch shifting (frame reads at stride ratio)
//   - 2048-pt True-Envelope estimation (Roebel-Rodet, SYMMETRIC quefrency
//     lifter with formant-bandwidth-adaptive cutoff). 2048 pts at 48 kHz
//     gives 23.4 Hz bin spacing so male harmonics (80-130 Hz F0) resolve
//     cleanly — 1024 pts smeared adjacent harmonics into one another. The
//     peak-hull pre-interpolation is a monotonic PCHIP (Fritsch-Carlson)
//     cubic Hermite, C1-smooth at the harmonic peaks.
//   - same-grid peak-locked phase vocoder (Laroche-Dolson), prominence-
//     gated peaks, unvoiced fallback
//   - formant-preserving envelope replacement + piecewise-linear knee VTLN
//     warp (F1 decoupled; log-octave spectral tilt, dynamic H1 harmonic
//     boost on the first detected spectral peak)
//   - voiced/unvoiced-gated sibilant bypass (3.5-5.5 kHz raised cosine,
//     complex-bin blend) — dry sibilants only mix on UNVOICED frames so
//     vowels never comb-filter against the pitch-shifted spectrum
//   - tract-shaped 1.5-7 kHz xorshift32 aspiration noise scaled by the
//     spectral ENVELOPE (fills valleys between harmonics; deterministic,
//     RT-safe), pitch-synchronously modulated by +-F0-bin spectral sidebands
//     so the aspiration amplitude pulses with the glottal period
//   - transient-gated phase reset: spectral-flux onsets (plosives) bypass
//     phase-vocoder propagation to avoid dispersive smearing
//   - ring-buffer OLA with FIXED emission latency (kLatency = 9216 samples
//     = 192 ms @ 48 kHz)
//
// Optional real-time retune front end (off by default):
//   input -> F0 tracking (4:1-decimated NSDF at 12 kHz, 1.2 kHz AA filter,
//     continuity scoring, harmonic unwinding, octave-jump guard)
//         -> retune (12-bit scale snap / MIDI note target + exponential glide)
//         -> vibrato -> total pitch shift (manual pitch_st + correction).
//   When correction_enable and MIDI are both off, the downsampling and NSDF
//   tracking are skipped entirely: the node is a pure manual pitch/formant/
//   gender shifter with zero tracking overhead.
//

//
// mix = 0 is a bit-exact memcpy bypass; set_param(mix) clears transient state on
// bypass-boundary transitions (anti-ghosting). All ring indices are wrapped into
// [0, kRingSize) via & kRingMask AFTER reducing positions into int range — the
// double->int cast must never receive an out-of-range value (UB).
//
// Zero steady-state heap allocation: every buffer is a fixed member array.

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

constexpr int   kFFT = 2048;              // 42.7 ms window: resolves male harmonics
constexpr int   kHop = 256;               // 87.5% overlap
constexpr int   kHalf = 1025;             // real frequency bins incl. DC + Nyquist
constexpr int   kLifter = 48;             // wider cepstral lifter for 2048 bins
// NOTE: kFFT/kHop/kHalf/kLifter above are MAXIMUM capacities (member array
// sizes). The active slice is controlled per latency mode via fft_size_/hop_/
// num_bins_/base_lifter_ below; all DSP loops must use the members, never the
// constants (except array declarations and Studio-mode defaults).
constexpr int   kIters = 4;
constexpr int   kMaxChannels = 2;
constexpr int   kRingSize = 16384;        // input history + OLA ring (power of 2;
                                          // must exceed kLatency + kFFT)
constexpr int   kRingMask = kRingSize - 1;
// Fixed emission latency: output block k reads ring positions L behind the
// input stream. Frame m spans 2048*ratio input samples (ratio <= 4 at +-24 st)
// and output sample o is complete once frame floor(o/256) has been processed,
// so L >= 1024 + 2048*4 = 9216 keeps every emitted sample fully accumulated.
constexpr long long kLatency = 9216;
// Operating / latency modes (see configure_mode): Studio keeps the full
// 2048-pt analysis; Live/UltraLive drop to 1024-pt with proportionally
// smaller hops, lifter cutoffs, pitch ranges, and emission latencies.
enum LatencyMode {
    kModeStudio = 0,    // N=2048, H=256, L=9216 (192 ms), pitch +-24 st
    kModeLive = 1,      // N=1024, H=128, L=2560 (53.3 ms), pitch +-12 st
    kModeUltraLive = 2  // N=1024, H=128, L=2048 (42.7 ms), pitch +-7 st
};
// Pitch-tracker analysis constants (12 kHz decimated domain).
constexpr int kPitchDecim = 4;
constexpr int kPitchBufSize = 1024;
constexpr int kPitchMinLag = 15;   // 800 Hz @ 12 kHz
constexpr int kPitchMaxLag = 240;  // 50 Hz @ 12 kHz
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kXoverLowHz = 3500.0f;    // sibilant bypass raised-cosine band
constexpr float kXoverHighHz = 5500.0f;
constexpr float kZoneF1Hz = 1000.0f;    // Zone 1/2 boundary (F1 aperture vs F2)
constexpr float kZoneF2Hz = 2500.0f;    // Zone 2/3 boundary (tongue vs throat)
constexpr float kZoneF3Hz = 5000.0f;    // Zone 3 end; linear compress to Nyquist above
constexpr float kH1MaxHz = 400.0f;        // dynamic H1 boost applies below this
constexpr unsigned int kRngSeed = 0x1D872B41u;
// RT burst guard: frames processed per process() call, per channel. In steady
// state the pipeline needs exactly 2 frames per 512-sample block; the cap only
// engages when a large downward pitch-modulation step suddenly shrinks the
// frame span, leaving frame_index_ up to ~30 frames behind. Without the cap a
// single audio callback would run hundreds of 2048-pt FFTs and blow the RT
// deadline; with it the backlog drains over the following blocks (the lag
// stays far below the 16384-sample input ring, so no frame data is lost).
constexpr int kMaxFramesPerCall = 4;

inline float wrap_phase(float x) {
    // Branchless reduction (a data-dependent while-loop could spin for a very
    // long time on a non-finite / huge upstream value). NaN propagates and is
    // caught by the isfinite clamps in synthesize().
    return x - kTwoPi * std::floor((x + kPi) / kTwoPi);
}

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

class VocalTransformerProcessor {
public:
    VocalTransformerProcessor()
        : sr_(48000.0f),
          pitch_st_(0.0f), formant_st_(0.0f), gender_morph_(0.0f),
          breathiness_(0.0f), sibilant_mix_(0.85f), mix_(1.0f), prev_mix_(1.0f),
          correction_enable_(0.0f),
          scale_root_(0), scale_mask_(0xFFF),
          retune_speed_ms_(20.0f), vibrato_depth_(0.0f),
          vibrato_rate_(5.5f), midi_mode_(0.0f),
          target_midi_note_(-1.0f), detected_f0_(0.0f),
          current_pitch_semitone_(60.0f),
          target_smoothed_semitone_(60.0f),
          vibrato_phase_(0.0f),
          ext_f0_mode_(0.0f), ext_f0_hz_(0.0f), ext_voiced_(0.0f),
          last_accepted_f0_(0.0f), jump_pending_f0_(0.0f),
          jump_confirm_(0), voicing_hangover_(0),
          current_h1_bin_(-1), rng_state_(kRngSeed) {
        aa_filter_.set_lowpass(1200.0f, 0.7071f, sr_);
        build_tables();
        recompute_tables();
        reset();
    }

    void set_samplerate(float sr) {
        // Windows are normalized-frequency; sr scales the Hz mapping of the
        // VTLN band boundary, the tilt/H1 shaping bands, and the breath band
        // limits — every table below must be rebuilt when the rate changes.
        // The pitch-tracker anti-alias filter tracks the rate too; the
        // transient pipeline is cleared like any other rate change.
        if (sr > 1.0f && sr != sr_) {
            sr_ = sr;
            aa_filter_.set_lowpass(1200.0f, 0.7071f, sr_);
            recompute_tables();
            reset_transient();
        }
    }

    void set_param(int id, float v) {
        switch (id) {
            case 0:
                // Off->on transition: the decimated tracker buffer holds
                // stale audio (or zeros); clear it so the first tracked
                // blocks start from current input instead of history.
                if (v > 0.5f && correction_enable_ <= 0.5f) reset_tracker();
                correction_enable_ = v > 0.5f ? 1.0f : 0.0f;
                break;
            case 1: scale_root_ = std::max(0, std::min(11, static_cast<int>(v))); break;
            case 2: scale_mask_ = static_cast<int>(v); break;
            case 3: retune_speed_ms_ = std::max(0.0f, std::min(100.0f, v)); break;
            case 4: pitch_st_ = std::max(-24.0f, std::min(24.0f, v)); break;
            case 5: formant_st_ = std::max(-24.0f, std::min(24.0f, v)); break;
            case 6:
                gender_morph_ = std::max(-1.0f, std::min(1.0f, v));
                // VTLN warp buckets and spectral tilt depend
                // only on gender_morph; rebuild every table here (param-change
                // path) instead of recomputing 1025 log2/pow per frame
                // inside replace_envelope()/synthesize().
                recompute_tables();
                break;
            case 7: vibrato_depth_ = std::max(0.0f, std::min(2.0f, v)); break;
            case 8: vibrato_rate_ = std::max(2.0f, std::min(9.0f, v)); break;
            case 9: breathiness_ = std::max(0.0f, std::min(1.0f, v)); break;
            case 10: sibilant_mix_ = std::max(0.0f, std::min(1.0f, v)); break;
            case 11:
                mix_ = std::max(0.0f, std::min(1.0f, v));
                // Bypass-boundary anti-ghosting: crossing mix = 0 invalidates the
                // transient pipeline (stale rings / OLA / phase history).
                if ((prev_mix_ <= 0.0f) != (mix_ <= 0.0f)) reset_transient();
                prev_mix_ = mix_;
                break;
            case 12:
                if (v > 0.5f && midi_mode_ <= 0.5f) reset_tracker();
                midi_mode_ = v > 0.5f ? 1.0f : 0.0f;
                break;
            case 13: target_midi_note_ = v; break;
            case 14:
                configure_mode(static_cast<int>(v));
                break;
            case 15: {
                // External F0 mode: 0 = internal NSDF, 1 = external with
                // authoritative gate wire, 2 = external with local fallback
                // voicing (RMS silence veto in process() + spectral veto in
                // synthesize()). A mode change invalidates F0 history so a
                // stale pitch cannot carry across sources.
                int m = static_cast<int>(v);
                if (m < 0) m = 0;
                if (m > 2) m = 2;
                if (static_cast<float>(m) != ext_f0_mode_) {
                    ext_f0_mode_ = static_cast<float>(m);
                    reset_tracker();
                }
                break;
            }
            case 16: ext_f0_hz_ = std::max(0.0f, std::min(2000.0f, v)); break;
            case 17: ext_voiced_ = (v > 0.5f) ? 1.0f : 0.0f; break;
            default: break;
        }
    }

    int latency_samples() const { return static_cast<int>(latency_samples_); }

    void reset() {
        reset_transient();
        prev_mix_ = mix_;
        target_smoothed_semitone_ = 60.0f;
        current_pitch_semitone_ = 60.0f;
        vibrato_phase_ = 0.0f;
        // Clean slate for the external-F0 wire state: the Python wrapper
        // re-pushes mode/hz/voiced on the next block when f0_in is still
        // connected (change-detected with a None sentinel on start()), so
        // clearing here cannot strand a connected session, and a
        // disconnected session cannot inherit a stale external mode.
        ext_f0_mode_ = 0.0f;
        ext_f0_hz_ = 0.0f;
        ext_voiced_ = 0.0f;
    }

    void process(const float* in, float* out, int channels, int frames) {
        if (!in || !out || frames <= 0) return;
        if (channels < 1) channels = 1;
        if (channels > kMaxChannels) channels = kMaxChannels;

        // Dry bypass: bit-exact memcpy early-out (zero latency, zero CPU).
        // Must honor the mono->stereo duplication contract of the normal path,
        // otherwise a raw channels==1 call would leave output channel 1 stale.
        if (mix_ <= 0.0f) {
            const int chs = (channels == 1) ? kMaxChannels : channels;
            for (int c = 0; c < chs; ++c) {
                const float* src = (channels == 1) ? in : in + c * frames;
                std::memcpy(out + c * frames, src,
                            static_cast<size_t>(frames) * sizeof(float));
            }
            return;
        }

        // Retune front end: F0 tracking, scale/MIDI target, glide, vibrato.
        // Skipped entirely (zero tracking overhead) when pitch correction
        // and MIDI targeting are both off: the node is then a pure manual
        // pitch/formant/gender shifter. F0 tracking reads channel 0.
        // When the external-F0 wire (f0_in) is connected, the 12 kHz NSDF
        // tracker is bypassed entirely: detected_f0_ comes from the CV
        // instead. detected_f0_ is still published for the H1/H2 glottal
        // shaping and pitch-synchronous aspiration in synthesize()/
        // reconstruct() even with correction disabled (natural-inflection
        // speech conversion); the scale/MIDI correction is only APPLIED to
        // total_shift when correction_enable is on.
        float total_shift = pitch_st_;
        const bool ext_active =
            (ext_f0_mode_ == 1.0f) || (ext_f0_mode_ == 2.0f);
        if (ext_active) {
            bool voiced = (ext_voiced_ > 0.5f);
            if (ext_f0_mode_ == 2.0f) {
                // Fallback voicing (no gate wire): the upstream node holds
                // its last pitch through pauses, so qualify the held CV
                // against local input energy. Layout note: the native input
                // is planar (channel blocks), matching track_pitch_and_retune
                // which reads in[i*kPitchDecim+d] off channel 0 — in[0..frames)
                // is channel 0 here as well. Spectral (fricative) rejection
                // happens per-frame in synthesize() via compute_voiced_prob.
                float energy = 0.0f;
                for (int i = 0; i < frames; ++i) energy += in[i] * in[i];
                const float rms = std::sqrt(
                    energy / static_cast<float>(frames > 0 ? frames : 1));
                if (rms < 0.001f) voiced = false;
            }

            if (voiced && ext_f0_hz_ >= 50.0f && ext_f0_hz_ <= 2000.0f) {
                detected_f0_ = ext_f0_hz_;
                last_accepted_f0_ = detected_f0_;
                voicing_hangover_ = 4;
                jump_confirm_ = 0;
                jump_pending_f0_ = 0.0f;
            } else if (voicing_hangover_ > 0 && last_accepted_f0_ > 0.0f) {
                --voicing_hangover_;
                detected_f0_ = last_accepted_f0_;
            } else {
                detected_f0_ = 0.0f;
                last_accepted_f0_ = 0.0f;
            }

            if (detected_f0_ > 50.0f) {
                update_retune_target(frames);
                if (correction_enable_ > 0.5f) {
                    const float correction =
                        target_smoothed_semitone_ - current_pitch_semitone_;
                    total_shift += std::max(-12.0f, std::min(12.0f, correction));
                }
            }
        } else {
            const bool need_tracking =
                (correction_enable_ > 0.5f) || (midi_mode_ > 0.5f);
            if (need_tracking) {
                track_pitch_and_retune(in, frames);
                if (correction_enable_ > 0.5f && detected_f0_ > 50.0f) {
                    const float correction =
                        target_smoothed_semitone_ - current_pitch_semitone_;
                    total_shift += std::max(-12.0f, std::min(12.0f, correction));
                }
            } else {
                detected_f0_ = 0.0f;
            }
        }

        if (vibrato_depth_ > 0.001f) {
            vibrato_phase_ += kTwoPi * vibrato_rate_ *
                              (static_cast<float>(frames) / sr_);
            vibrato_phase_ = std::fmod(vibrato_phase_, kTwoPi);
            total_shift += vibrato_depth_ * std::sin(vibrato_phase_);
        }

        // Mode-aware clamp: Live/UltraLive budgets only cover their reduced
        // max span, so manual shift + correction + vibrato together must never
        // exceed the mode range (belt & braces: ratio clamped again below).
        total_shift = std::max(-max_pitch_st_, std::min(max_pitch_st_, total_shift));
        const float ratio = std::min(std::pow(2.0f, total_shift / 12.0f),
                                     max_pitch_ratio_);

        // Mono input: duplicate internally to both output channels.
        const int chs = (channels == 1) ? kMaxChannels : channels;

        for (int c = 0; c < chs; ++c) {
            const float* in_ch = (channels == 1) ? in : in + c * frames;
            float* out_ch = out + c * frames;

            // Push this block's input into the history ring.
            for (int i = 0; i < frames; ++i)
                in_ring_[c][(total_received_[c] + i) & kRingMask] = in_ch[i];
            total_received_[c] += frames;

            // Process every frame whose input span is fully available.
            // Frame m starts at input position 256*m and spans 2048*ratio
            // input samples — the pitch shift happens in the frame read.
            // The dry sibilant frame in reconstruct() reads stride 1.0, i.e.
            // the FULL 2048-sample window, so dispatch must wait for
            // 2048*max(ratio, 1) input samples. For downward shifts
            // (ratio < 1) the pitch-shifted frame is available earlier than
            // the dry frame; dispatching on 2048*ratio alone would read
            // not-yet-received future samples as stale ring data and corrupt
            // the dry sibilant spectrum on every unvoiced frame.
            const double span = static_cast<double>(fft_size_) * (double)std::max(ratio, 1.0f);
            int frames_this_call = 0;
            while (static_cast<double>(hop_) * (double)frame_index_[c] + span
                       <= (double)total_received_[c] + 1e-6
                    && frames_this_call < max_frames_per_call_) {
                process_frame(c, ratio);
                ++frames_this_call;
            }

            // Per-block wet/dry endpoints; the emission loop interpolates
            // between them per sample so mix automation cannot click at
            // block boundaries.
            const float m_start = mix_smooth_[c];
            mix_smooth_[c] += 0.5f * (mix_ - mix_smooth_[c]);
            const float m_end = std::max(0.0f, std::min(1.0f, mix_smooth_[c]));
            const float m0 = std::max(0.0f, std::min(1.0f, m_start));

            // Emit this block's output from the fixed latency L behind the
            // input stream (every emitted sample is fully accumulated; the
            // first L samples of the stream are silence). The dry path is
            // read from the input history ring at the same latency-aligned
            // position, so (0,1) mix values crossfade without comb filtering.
            const long long read_base = total_received_[c] - latency_samples_;
            for (int i = 0; i < frames; ++i) {
                const long long o = read_base + i;
                if (o >= 0) {
                    const float wet = ola_ring_[c][o & kRingMask];
                    ola_ring_[c][o & kRingMask] = 0.0f;
                    const float dry = in_ring_[c][o & kRingMask];
                    const float t = static_cast<float>(i) / static_cast<float>(frames);
                    const float mb = m0 + (m_end - m0) * t;
                    float y = (1.0f - mb) * dry + mb * wet;
                    if (!std::isfinite(y)) y = 0.0f;
                    out_ch[i] = y;
                } else {
                    out_ch[i] = 0.0f;
                }
            }
        }

        // Zero any unused output channels (anti-ghosting). Defensive only:
        // chs is always kMaxChannels today (mono is internally duplicated).
        for (int c = chs; c < kMaxChannels; ++c)
            std::memset(out + c * frames, 0, static_cast<size_t>(frames) * sizeof(float));
    }

private:
    // ---- Retune front end --------------------------------------------------
    float sr_;
    float pitch_st_;       // [-24, +24] manual pitch relocation (semitones)
    float formant_st_;     // [-24, +24] semitones
    float gender_morph_;   // [-1, +1] VTLN warping
    float breathiness_;    // [0, 1]
    float sibilant_mix_;   // [0, 1]
    float mix_;            // [0, 1]
    float prev_mix_;       // bypass-transition detection
    float correction_enable_;
    int scale_root_, scale_mask_;
    float retune_speed_ms_, vibrato_depth_, vibrato_rate_;
    float midi_mode_, target_midi_note_;
    // External F0 CV (driven per-block by the Python wrapper when f0_in is
    // wired; NOT staged parameters, so never in PARAM_MAP / save files):
    //   ext_f0_mode_ 0 = internal NSDF, 1 = external + gate wire,
    //                2 = external + local fallback voicing
    //   ext_f0_hz_   continuous fundamental in Hz (clamped 0-2000)
    //   ext_voiced_  1 = voiced, 0 = unvoiced (gate wire or Python fallback)
    float ext_f0_mode_, ext_f0_hz_, ext_voiced_;
    float detected_f0_, current_pitch_semitone_, target_smoothed_semitone_;
    float vibrato_phase_;

    // ---- Latency mode (active DSP slice of the max-sized buffers) ---------
    int latency_mode_ = kModeStudio;
    int fft_size_ = kFFT;
    int hop_ = kHop;
    int num_bins_ = kHalf;
    int base_lifter_ = kLifter;
    float max_pitch_st_ = 24.0f;
    float max_pitch_ratio_ = 4.0f;
    long long latency_samples_ = kLatency;
    int max_frames_per_call_ = kMaxFramesPerCall;

    Biquad aa_filter_;
    float pitch_downsample_buf_[kPitchBufSize];
    float nsdf_[kPitchMaxLag];

    float last_accepted_f0_, jump_pending_f0_;
    int jump_confirm_, voicing_hangover_;

    // ---- Per-channel spectral pipeline state --------------------------------
    float in_ring_[kMaxChannels][kRingSize];   // raw input history
    float ola_ring_[kMaxChannels][kRingSize];  // output OLA ring (trailing read)
    long long total_received_[kMaxChannels];   // input samples received
    long long frame_index_[kMaxChannels];      // frames processed (start = 256*m)
    float prev_phase_in_[kMaxChannels][kHalf];
    float prev_phase_out_[kMaxChannels][kHalf];
    float mix_smooth_[kMaxChannels];           // de-clicked wet/dry per channel

    // ---- Pre-computed tables (constructor only) -----------------------------
    float window_[kFFT];        // analysis Hann
    float synth_window_[kFFT];  // Hann / 1.5 (exact COLA unity at 75% overlap)
    int   rev_[kFFT];           // FFT bit-reversal permutation
    float cos_tab_[kFFT / 2];   // twiddle tables
    float sin_tab_[kFFT / 2];
    float vtln_warp_bin_[kHalf]; // VTLN warp buckets: source bin -> dest pos
                                // (rebuilt in recompute_tables() on gender/sr change)
    float excitation_shaper_[kHalf]; // spectral tilt * H1 emphasis, multiplied
                                // into the fine excitation in synthesize()
                                // (rebuilt in recompute_tables() too)

    // ---- Per-hop scratch (no allocation) ------------------------------------
    float scratch_time_[kFFT];     // synthesized output frame (pre-window)
    float fft_re_[kFFT];           // complex FFT work buffers (planar layout)
    float fft_im_[kFFT];
    float spec_re_[kHalf];         // analysis half-spectrum (original, for blending)
    float spec_im_[kHalf];
    float mag_[kHalf];
    float phase_[kHalf];
    float log_mag_[kHalf];         // running log-envelope A_i(k)
    float envelope_[kHalf];        // original envelope -> final warped envelope
    float warped_envelope_[kHalf]; // formant-resampled stage
    float analysis_envelope_[kHalf]; // unwarped analysis envelope snapshot for peak gating
    float excitation_[kHalf];      // fine excitation H(k), from ORIGINAL envelope
    float synth_mag_[kHalf];       // destination grid
    float synth_phase_[kHalf];     // destination grid
    int   peak_bins_[kHalf];
    int   peak_owner_[kHalf];      // source bin -> nearest peak index
    float dry_time_[kFFT];         // unshifted dry frame for unvoiced sibilant bypass
    float dry_spec_re_[kHalf];
    float dry_spec_im_[kHalf];
    float log_mag_in_[kHalf];      // precomputed log(mag_) input for the hull
    int   raw_peak_bins_[kHalf];   // peak detection work buffer for upper hull
    float raw_peak_slopes_[kHalf]; // PCHIP (Fritsch-Carlson) slopes d_k at raw peaks
    float prev_mag_[kMaxChannels][kHalf]; // previous frame magnitude (spectral flux)
    float noise_raw_re_[kHalf];    // raw aspiration noise spectrum (pitch-sync)
    float noise_raw_im_[kHalf];
    int   current_h1_bin_;         // detected F0 bin for this frame (-1 = undetermined)
    int   last_h1_bin_[kMaxChannels]; // last overtone-confirmed H1 per channel
    int   h1_hangover_[kMaxChannels]; // frames of continuity fallback remaining
    unsigned int rng_state_;

    void reset_transient() {
        for (int c = 0; c < kMaxChannels; ++c) {
            std::memset(in_ring_[c], 0, sizeof(float) * kRingSize);
            std::memset(ola_ring_[c], 0, sizeof(float) * kRingSize);
            total_received_[c] = 0;
            frame_index_[c] = 0;
            std::memset(prev_phase_in_[c], 0, sizeof(float) * kHalf);
            std::memset(prev_phase_out_[c], 0, sizeof(float) * kHalf);
            std::memset(prev_mag_[c], 0, sizeof(float) * kHalf);
            last_h1_bin_[c] = -1;
            h1_hangover_[c] = 0;
            mix_smooth_[c] = mix_;
        }
        reset_tracker();
        current_h1_bin_ = -1;
        rng_state_ = kRngSeed;
    }

    void reset_tracker() {
        std::memset(pitch_downsample_buf_, 0, sizeof(pitch_downsample_buf_));
        std::memset(nsdf_, 0, sizeof(nsdf_));
        aa_filter_.reset();
        detected_f0_ = 0.0f;
        last_accepted_f0_ = 0.0f;
        jump_pending_f0_ = 0.0f;
        jump_confirm_ = 0;
        voicing_hangover_ = 0;
    }

    // ---- Retune front end ---------------------------------------------------

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
            jump_pending_f0_ = 0.0f;
            jump_confirm_ = 0;
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
            if (last_accepted_f0_ > 50.0f) {
                // Continuity scoring (one-block Viterbi): speech F0 moves
                // continuously, so among all viable peaks prefer the one
                // nearest last block's F0. Upward jumps are penalised hard:
                // an octave-up lock would feed the retune corrector a false
                // target and warble the output, while a downward error only
                // coarsens tracking benignly. Real jumps still win: when the
                // voice truly moves, the old peak vanishes and the new one
                // takes r_max unopposed.
                const float net = std::max(0.35f * r_max, 0.25f);
                float best_score = -1e30f;
                for (int tau = kPitchMinLag + 1; tau < kPitchMaxLag - 1; ++tau) {
                    const float v = nsdf_[tau];
                    if (v <= 0.0f || v < net) continue;
                    if (!(v > nsdf_[tau - 1] && v >= nsdf_[tau + 1])) continue;
                    const float f = (sr_ / static_cast<float>(kPitchDecim)) /
                                    static_cast<float>(tau);
                    const float st = 12.0f * std::log2(f / last_accepted_f0_);
                    const float pen = st > 0.0f ? 0.030f * st : -0.012f * st;
                    const float score = v - pen;
                    if (score > best_score) {
                        best_score = score;
                        best_tau = tau;
                    }
                }
                // Fall through to MPM acquisition below if nothing scored.
            }
            if (best_tau <= 0) {
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
            // Fundamental preference: the smallest-lag pick above is easily
            // a strong upper harmonic (H2-H6 sit near fundamental level in
            // real vowels), locking an octave or more sharp. If a small
            // integer multiple of the candidate is itself a local peak and
            // clearly higher, the candidate was a harmonic: unwind toward
            // the fundamental. The margin keeps true high voices (whose
            // multiples correlate almost as well) from doubling down.
            for (int iter = 0; iter < 2 && best_tau > 0; ++iter) {
                bool moved = false;
                for (int m = 2; m <= 6; ++m) {
                    const int cand = best_tau * m;
                    if (cand - 1 < kPitchMinLag || cand + 1 >= kPitchMaxLag)
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
            update_retune_target(frames);
        }
    }

    // Shared scale/MIDI target + glide update for a freshly accepted F0.
    // Called by both the internal NSDF path (end of track_pitch_and_retune)
    // and the external-F0 bypass branch in process(). Keeping one copy
    // avoids glide-math drift between the two sources. Runs whenever F0 is
    // valid — even with correction disabled — so re-enabling correction
    // starts from a warm glide state instead of sweeping from stale history.
    // The caller decides whether the resulting correction is APPLIED to
    // total_shift (correction_enable gate in process()).
    void update_retune_target(int frames) {
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

    // Latency-mode switch (param-change path only, never steady-state):
    // rescales the active DSP slice, rebuilds tables, and clears transient
    // state so the dry-delay read pointer step cannot pop. No heap use:
    // every buffer stays at its static maximum capacity.
    void configure_mode(int mode) {
        if (mode < kModeStudio) mode = kModeStudio;
        if (mode > kModeUltraLive) mode = kModeUltraLive;
        if (mode == latency_mode_) return;
        latency_mode_ = mode;
        switch (mode) {
            case kModeLive:
                fft_size_ = 1024;
                hop_ = 128;
                num_bins_ = 513;
                base_lifter_ = 24;
                max_pitch_st_ = 12.0f;
                max_pitch_ratio_ = 2.0f;
                latency_samples_ = 2560;
                break;
            case kModeUltraLive:
                fft_size_ = 1024;
                hop_ = 128;
                num_bins_ = 513;
                base_lifter_ = 24;
                max_pitch_st_ = 7.0f;
                max_pitch_ratio_ = 1.4983f;  // 2^(7/12)
                latency_samples_ = 2048;
                break;
            case kModeStudio:
            default:
                fft_size_ = kFFT;
                hop_ = kHop;
                num_bins_ = kHalf;
                base_lifter_ = kLifter;
                max_pitch_st_ = 24.0f;
                max_pitch_ratio_ = 4.0f;
                latency_samples_ = kLatency;
                break;
        }
        // Steady state needs BLOCK_SIZE/hop frames per block; keep the RT
        // burst guard proportional so backlogs can still drain in Live modes.
        max_frames_per_call_ = kMaxFramesPerCall * kHop / hop_;
        build_tables();
        recompute_tables();
        reset_transient();
    }

    void build_tables() {
        // Synthesis window normalization: the effective OLA window is
        // analysis*synthesis = hann^2, whose COLA sum at hop H is
        // (N/H)*0.375 (exact for Hann^2 at any N/H integer ratio).
        // H = N/4 -> 1.5, H = N/8 (2048/256) -> 3.0.
        const float cola = static_cast<float>(fft_size_) / static_cast<float>(hop_)
                           * 0.375f;
        for (int i = 0; i < fft_size_; ++i) {
            const float w = 0.5f - 0.5f * std::cos(kTwoPi * static_cast<float>(i) / fft_size_);
            window_[i] = w;
            synth_window_[i] = w / cola;
        }
        int bits = 0;
        while ((1 << bits) < fft_size_) ++bits;
        for (int i = 0; i < fft_size_; ++i) {
            int r = 0, x = i;
            for (int b = 0; b < bits; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
            rev_[i] = r;
        }
        for (int j = 0; j < fft_size_ / 2; ++j) {
            cos_tab_[j] = std::cos(kTwoPi * static_cast<float>(j) / fft_size_);
            sin_tab_[j] = std::sin(kTwoPi * static_cast<float>(j) / fft_size_);
        }
    }

    // Precompute every parameter-dependent spectral table:
    //   vtln_warp_bin_[k]   — 3-zone anatomical VTLN warp (destination bin k
    //                         -> source position). Male/female tracts do not
    //                         scale uniformly (the male pharynx is ~20-25%
    //                         longer, the oral cavity only ~8-10%), and the
    //                         legacy 1 kHz anchor dominates absolute peak
    //                         positions, so zone rates are tuned for absolute
    //                         band shifts (measured on a two-resonance assay):
    //                         Zone 1 (< 1 kHz, F1/vowel aperture): legacy
    //                         reduced-intensity taper, kept verbatim to
    //                         protect vowel intelligibility;
    //                         Zone 2 (1-2.5 kHz, F2/tongue position): 0.30x
    //                         the base shift — mid peak lands ~+2.1 st
    //                         (+13.0%, inside the +10-14% oral band);
    //                         Zone 3 (2.5-5 kHz, F3/F4 throat length and sex
    //                         identity): 1.50x the base shift — high peak
    //                         lands ~+2.75 st (+17.2%, inside the +15-20%
    //                         throat band).
    //                         Above 5 kHz the map compresses linearly to
    //                         Nyquist (sibilance/air band, no formants).
    //                         Closed-form piecewise segments share knot
    //                         values, so the map is C0-continuous and strictly
    //                         monotonic by construction (all C1 slope jumps
    //                         stay within the legacy roughness); identity at
    //                         neutral.
    //   excitation_shaper_[k] — logarithmic (dB/octave) spectral tilt anchored
    //                         at 1 kHz, clamped to +-10 dB. Human spectral
    //                         slope is logarithmic in octaves, not linear in
    //                         Hz, so this keeps the tilt audible in the
    //                         1-3 kHz speech-intelligence band.
    // Depends only on gender_morph_ / sr_; rebuilt on param or rate change.
    void recompute_tables() {
        const float bin_hz = sr_ / static_cast<float>(fft_size_);
        // Base tract shift: +1 (feminine) => +3 st shortening (upward
        // formant shift), -1 (masculine) => -3 st lengthening.
        const float base_st = gender_morph_ * 3.0f;
        const float f_nyq = sr_ * 0.5f;
        // Zone rates (destination f -> source f_s = f / r within a zone).
        const float w_full = std::pow(2.0f, base_st / 12.0f);
        const float r_zone2 = std::pow(2.0f, base_st * 0.30f / 12.0f);
        const float r_zone3 = std::pow(2.0f, base_st * 1.50f / 12.0f);
        // Knot values: F1000 matches the legacy taper exactly at 1 kHz
        // (the taper factor is 0.6 + 0.4 * 1.0 = 1.0 there).
        const float F1000 = 1000.0f / w_full;
        const float F2500 = F1000 + (kZoneF2Hz - kZoneF1Hz) / r_zone2;
        const float F5000 = F2500 + (kZoneF3Hz - kZoneF2Hz) / r_zone3;
        const float hi_slope = (f_nyq - F5000) / (f_nyq - kZoneF3Hz);
        // Gender morph: +1 (fem) adds -2.5 dB/oct (softer, leaking glottal
        // source), -1 (masc) adds +2.5 dB/oct (sharper, buzzy closure).
        const float tilt_db_per_oct = -gender_morph_ * 2.5f;

        for (int k = 0; k < num_bins_; ++k) {
            const float f_hz = static_cast<float>(k) * bin_hz;

            // 1. Three-zone anatomical VTLN warp table.
            float f_source;
            if (base_st == 0.0f) {
                f_source = f_hz;
            } else if (f_hz <= kZoneF1Hz) {
                // Zone 1: legacy F1 decoupling taper, verbatim.
                const float r = 1.0f + (w_full - 1.0f)
                                      * (0.6f + 0.4f * (f_hz / 1000.0f));
                f_source = f_hz / r;
            } else if (f_hz <= kZoneF2Hz) {
                f_source = F1000 + (f_hz - kZoneF1Hz) / r_zone2;
            } else if (f_hz <= kZoneF3Hz) {
                f_source = F2500 + (f_hz - kZoneF2Hz) / r_zone3;
            } else {
                f_source = F5000 + hi_slope * (f_hz - kZoneF3Hz);
            }
            float b = f_source / bin_hz;
            if (b < 0.0f) b = 0.0f;
            if (b > static_cast<float>(num_bins_ - 1))
                b = static_cast<float>(num_bins_ - 1);
            vtln_warp_bin_[k] = b;

            // 2. Logarithmic (dB/octave) spectral tilt, 1 kHz anchor.
            float tilt_db = 0.0f;
            if (f_hz > 0.0f) {
                const float octaves = std::log2(std::max(f_hz, 50.0f) / 1000.0f);
                tilt_db = std::max(-10.0f, std::min(10.0f,
                                                    octaves * tilt_db_per_oct));
            }
            excitation_shaper_[k] = std::pow(10.0f, tilt_db / 20.0f);
        }
    }

    // ---- Per-hop pipeline ---------------------------------------------------

    void process_frame(int c, float ratio) {
        // 1. Analysis frame: read from the input ring at stride `ratio`
        //    (4-point cubic Hermite interpolation). Frame m starts at input position 256*m;
        //    the pitch shift happens right here in the frame read.
        //    fpos grows without bound over a long session, so wrap it into
        //    ring range BEFORE the double->int cast — an out-of-range cast is
        //    UB and (on x86 cvttsd2si) silently reads ring position 0.
        const double fpos = std::fmod(static_cast<double>(hop_) * (double)frame_index_[c], (double)kRingSize);
        for (int i = 0; i < fft_size_; ++i) {
            const double pos = fpos + (double)i * (double)ratio;
            const int i0 = (int)std::floor(pos);
            const float frac = (float)(pos - (double)i0);

            const float ym1 = in_ring_[c][(i0 - 1) & kRingMask];
            const float y0  = in_ring_[c][i0 & kRingMask];
            const float y1  = in_ring_[c][(i0 + 1) & kRingMask];
            const float y2  = in_ring_[c][(i0 + 2) & kRingMask];

            const float c0 = y0;
            const float c1 = 0.5f * (y1 - ym1);
            const float c2 = ym1 - 2.5f * y0 + 2.0f * y1 - 0.5f * y2;
            const float c3 = 0.5f * (y2 - ym1) + 1.5f * (y0 - y1);
            const float v  = ((c3 * frac + c2) * frac + c1) * frac + c0;

            scratch_time_[i] = v * window_[i];
        }
        forward_real(scratch_time_, spec_re_, spec_im_);

        // 2. Magnitude / phase.
        for (int k = 0; k < num_bins_; ++k) {
            mag_[k] = std::sqrt(spec_re_[k] * spec_re_[k] + spec_im_[k] * spec_im_[k]);
            phase_[k] = std::atan2(spec_im_[k], spec_re_[k]);
        }

        // 3. True envelope (Roebel-Rodet with peak-hull pre-interpolation).
        true_envelope();

        // 4. Fine excitation from the analysis envelope (before replacement),
        //    clamped to prevent explosion in spectral troughs.
        for (int k = 0; k < num_bins_; ++k) {
            float h = mag_[k] / (envelope_[k] + 1e-9f);
            excitation_[k] = h < 0.0f ? 0.0f : (h > 100.0f ? 100.0f : h);
        }

        // 5. Spectral peaks on the analysis grid BEFORE envelope replacement!
        // Prominence gating mag_[k] > envelope_[k] * 0.3f compares mag_ against
        // its own true envelope, not the warped target envelope.
        const int num_peaks = find_peaks();

        // Snapshot the unwarped analysis envelope so peak-locking in synthesize()
        // compares mag_ against its original envelope rather than the warped target.
        std::memcpy(analysis_envelope_, envelope_, num_bins_ * sizeof(float));

        // 6. Formant-preserving envelope replacement + formant/VTLN warp.
        replace_envelope(ratio);

        // 7. Peak-locked phases (time-stretch propagation) + magnitudes.
        synthesize(c, num_peaks, ratio);

        // 8. Sibilant complex-bin blend + reconstruction into scratch_time_.
        reconstruct(c, fpos, num_peaks);

        // 9. Synthesis window, accumulate into the OLA ring at this frame's
        //     output position (256*m). Emission is handled by the trailing
        //     read pointer in process() once the frame is the last contributor.
        const long long base = static_cast<long long>(hop_) * frame_index_[c];
        for (int i = 0; i < fft_size_; ++i)
            ola_ring_[c][(base + i) & kRingMask] += scratch_time_[i] * synth_window_[i];
        frame_index_[c]++;
    }

    void true_envelope() {
        // Step 1: Detect local peaks in mag_ to construct an upper hull in the log
        // magnitude domain. This bridges the 50 dB inter-harmonic valleys before
        // cepstral liftering, preventing the envelope from dipping between harmonics
        // and causing comb notches when shifting.
        int num_raw_peaks = 0;
        raw_peak_bins_[num_raw_peaks++] = 0;
        for (int k = 1; k < num_bins_ - 1; ++k) {
            if (mag_[k] >= mag_[k - 1] && mag_[k] >= mag_[k + 1] && mag_[k] > 1e-6f) {
                raw_peak_bins_[num_raw_peaks++] = k;
                if (num_raw_peaks >= num_bins_ - 1) break;
            }
        }
        raw_peak_bins_[num_raw_peaks++] = num_bins_ - 1;

        // Precompute the input log-magnitudes once: the hull loop below needs
        // log(mag_[k]) at every bin plus log() at both bracketing peaks —
        // recomputing them inline triples the log() count per frame.
        for (int k = 0; k < num_bins_; ++k)
            log_mag_in_[k] = std::log(mag_[k] + 1e-9f);

        float* A = log_mag_;
        if (num_raw_peaks > 2) {
            // PCHIP (Fritsch-Carlson) monotonic cubic Hermite upper hull.
            // Piecewise-linear hulls have slope discontinuities at the harmonic
            // peaks; the cubic hull is C1-smooth, which removes artificial
            // ripples in the cepstral lifter output while staying monotone
            // between the detected peaks.
            const int M = num_raw_peaks;
            // Pass 1: interval secants into raw_peak_slopes_[0 .. M-2].
            for (int p = 0; p < M - 1; ++p) {
                const float dy = log_mag_in_[raw_peak_bins_[p + 1]]
                               - log_mag_in_[raw_peak_bins_[p]];
                raw_peak_slopes_[p] = dy / static_cast<float>(raw_peak_bins_[p + 1]
                                                              - raw_peak_bins_[p]);
            }
            // Pass 2: monotonic slope limiting, written backwards so the
            // secants still being consumed (slot p-1, p) are intact. Endpoints
            // use the one-sided secant.
            raw_peak_slopes_[M - 1] = raw_peak_slopes_[M - 2];
            for (int p = M - 2; p >= 1; --p) {
                const float dA = raw_peak_slopes_[p - 1];
                const float dB = raw_peak_slopes_[p];
                float d;
                if (dA * dB <= 0.0f) {
                    d = 0.0f;   // local extremum between peaks: flat tangent
                } else {
                    const float denom = dA + dB;
                    // Harmonic mean (2*dA*dB/(dA+dB)); denom == 0 is guarded
                    // (only reachable via non-finite secants).
                    d = (denom != 0.0f) ? (2.0f * dA * dB / denom) : 0.0f;
                }
                raw_peak_slopes_[p] = d;
            }
            // d_0 = delta_0 is already in slot 0.

            int p_idx = 0;
            for (int k = 0; k < num_bins_; ++k) {
                while (p_idx + 1 < M && raw_peak_bins_[p_idx + 1] < k) {
                    ++p_idx;
                }
                const int p0 = raw_peak_bins_[p_idx];
                const int p1 = (p_idx + 1 < M) ? raw_peak_bins_[p_idx + 1] : p0;
                float h;
                if (p1 > p0) {
                    const float hp = static_cast<float>(p1 - p0);
                    const float t = static_cast<float>(k - p0) / hp;
                    const float y0 = log_mag_in_[p0];
                    const float y1 = log_mag_in_[p1];
                    const float d0 = raw_peak_slopes_[p_idx];
                    const float d1 = (p_idx + 1 < M) ? raw_peak_slopes_[p_idx + 1]
                                                     : d0;
                    // Cubic Hermite basis (equivalent to H0..H3 form).
                    const float t2 = t * t;
                    const float t3 = t2 * t;
                    h = y0 * (2.0f * t3 - 3.0f * t2 + 1.0f)
                      + hp * d0 * (t3 - 2.0f * t2 + t)
                      + y1 * (-2.0f * t3 + 3.0f * t2)
                      + hp * d1 * (t3 - t2);
                } else {
                    h = log_mag_in_[p0];
                }
                A[k] = std::max(h, log_mag_in_[k]);
            }
        } else {
            for (int k = 0; k < num_bins_; ++k)
                A[k] = log_mag_in_[k];
        }

        for (int it = 0; it < kIters; ++it) {
            // Real, EVEN 2048-pt spectrum from A -> IFFT -> real cepstrum.
            for (int k = 0; k < num_bins_; ++k) { fft_re_[k] = A[k]; fft_im_[k] = 0.0f; }
            for (int k = 1; k < num_bins_ - 1; ++k) {
                fft_re_[fft_size_ - k] = A[k];
                fft_im_[fft_size_ - k] = 0.0f;
            }
            fft_run(true);

            // Symmetric liftering on BOTH quefrency halves: lifting only the
            // positive half would destroy c(n) = c(N - n) symmetry and make the
            // forward transform complex (imaginary leakage into the envelope).
            //
            // Adaptive lifter cutoff: feminine morphs (gender_morph_ > 0) have
            // higher acoustic wall/radiation losses relative to vocal-tract
            // volume, broadening formant bandwidths (lower Q). Shorten the
            // cutoff toward 3/4 of base (broader peaks) as gender_morph_ -> 1;
            // retain the full base-quefrency resolution (sharp peaks) at =< 0.
            // Scales with the FFT size (48/36 at N=2048, 24/18 at N=1024).
            int eff_lifter = base_lifter_;
            if (gender_morph_ > 0.0f) {
                const int lifter_delta = base_lifter_ / 4;
                const int lifter_floor = base_lifter_ - lifter_delta;
                eff_lifter = static_cast<int>(base_lifter_ - gender_morph_ * lifter_delta);
                if (eff_lifter < lifter_floor) eff_lifter = lifter_floor;
            }
            apply_symmetric_lifter(fft_re_, fft_size_, eff_lifter, gender_morph_);

            // Forward FFT of the liftered (real, even) cepstrum.
            for (int i = 0; i < fft_size_; ++i) fft_im_[i] = 0.0f;
            fft_run(false);
            for (int k = 0; k < num_bins_; ++k)
                A[k] = std::max(A[k], fft_re_[k]);   // imag ~ 0 by evenness
        }

        // Clamp before exponentiation: the running max() can only grow.
        const float a_max = std::log(1.0e6f);
        for (int k = 0; k < num_bins_; ++k)
            envelope_[k] = std::exp(std::min(A[k], a_max));
    }

    static void apply_symmetric_lifter(float* cep, int n, int cutoff, float gender_morph) {
        // Radial exponential damping for feminine morphs (broadens formant bandwidths / lowers Q):
        // c[q] *= gamma^q where gamma = 1.0 - 0.02 * gender_morph
        const float gamma = (gender_morph > 0.0f) ? (1.0f - 0.02f * gender_morph) : 1.0f;
        float gamma_pow = 1.0f;
        for (int q = 0; q < cutoff; ++q) {
            const float w = std::cos(0.5f * kPi * static_cast<float>(q) / cutoff) * gamma_pow;
            cep[q] *= w;
            if (q > 0) cep[n - q] *= w;   // mirror quefrency (q = 0 applied once)
            gamma_pow *= gamma;
        }
        std::memset(cep + cutoff, 0,
                    static_cast<size_t>(n - 2 * cutoff + 1) * sizeof(float));
    }

    float interp(const float* buf, float pos) {
        // Clamp the FRACTION too: clamping only the integer index would leave a
        // negative fraction when pos < 0, extrapolating past the array's lower
        // edge and potentially producing a negative envelope.
        if (pos < 0.0f) pos = 0.0f;
        if (pos > static_cast<float>(num_bins_ - 1)) pos = static_cast<float>(num_bins_ - 1);
        const int i = static_cast<int>(pos);
        const int j = (i < num_bins_ - 1) ? i + 1 : i;
        return buf[i] + (buf[j] - buf[i]) * (pos - static_cast<float>(i));
    }

    // Formant-preserving envelope replacement + formant/VTLN warp.
    // Single-pass formant-preserving envelope replacement + formant/VTLN warp.
    // Composes all three coordinate warps algebraically:
    //   Stage (a): pos_a = k * ratio
    //   Stage (b): pos_b = (k / rf) * ratio = k * (ratio / rf)
    //   Stage (c): pos_c = vtln_warp_bin_[k] * (ratio / rf)
    // Evaluating this in a single pass directly against the original continuous envelope_
    // eliminates intermediate discrete grid quantization and cascading linear-interpolation
    // low-pass filtering, preserving sharp formant peak definition and amplitude.
    void replace_envelope(float ratio) {
        const float rf = std::pow(2.0f, formant_st_ / 12.0f);
        const float scale = ratio / rf;
        for (int k = 0; k < num_bins_; ++k) {
            float pos = vtln_warp_bin_[k] * scale;
            if (pos < 0.0f) pos = 0.0f;
            if (pos > static_cast<float>(num_bins_ - 1))
                pos = static_cast<float>(num_bins_ - 1);
            warped_envelope_[k] = interp(envelope_, pos);
        }
        std::memcpy(envelope_, warped_envelope_, num_bins_ * sizeof(float));
    }

    int find_peaks() {
        // Prominence-gated peak picking: a candidate must be a local maximum
        // over a 5-bin span AND stand at least 30% above the local spectral
        // envelope. In noisy speech, plain 3-bin maxima declare 80-150 false
        // peaks per frame, tearing the phase vocoder's rigid peak locking and
        // causing random phase diffusion in low-energy regions.
        int n = 0;
        for (int k = 2; k < num_bins_ - 2; ++k) {
            if (mag_[k] > mag_[k - 1] && mag_[k] >= mag_[k + 1] &&
                mag_[k] > mag_[k - 2] && mag_[k] > mag_[k + 2] &&
                mag_[k] > envelope_[k] * 0.3f && mag_[k] > 1e-4f) {
                peak_bins_[n++] = k;
                if (n >= num_bins_) break;
            }
        }
        // Assign each source bin to its nearest peak (-1 when unvoiced).
        int pi = 0;
        for (int k = 0; k < num_bins_; ++k) {
            while (pi + 1 < n &&
                   std::abs(k - peak_bins_[pi + 1]) < std::abs(k - peak_bins_[pi]))
                ++pi;
            peak_owner_[k] = (n > 0) ? pi : -1;
        }
        return n;
    }

    void synthesize(int c, int num_peaks, float ratio) {
        // Transient-gated phase reset: detect sharp energy onsets (plosives)
        // via half-wave-rectified relative spectral flux above the low-frequency
        // rumble floor (k >= 4 ~ 94 Hz). On a transient the analysis spectrum is
        // already phase-coherent across bins, so we bypass the phase-vocoder
        // propagation (which would smear the onset over the 42 ms window) and
        // copy the analysis phases straight to the synthesis grid.
        bool is_transient = false;
        {
            float flux = 0.0f;
            float energy = 0.0f;
            // Rumble floor scales with bin width (~94 Hz in every mode:
            // bin k=4 at N=2048, k=2 at N=1024).
            const int kFluxMin = num_bins_ >> 8;
            for (int k = kFluxMin; k < num_bins_; ++k) {
                const float d = mag_[k] - prev_mag_[c][k];
                if (d > 0.0f) flux += d;
                energy += mag_[k];
            }
            const float flux_rel = flux / (energy + 1.0e-6f * static_cast<float>(num_bins_));
            // Thresholds: measured steady-state (periodic voiced) flux_rel is
            // ~0.002; a 10 ms plosive burst reaches ~0.6. 0.4 leaves a ~175x
            // margin against false positives while catching real onsets.
            is_transient = (flux_rel > 0.4f) && (energy > 0.05f);
        }

        // Classic same-grid peak-locked time-stretch propagation:
        //   analysis hop  Ha' = 256/ratio  (frame starts advance 256 input
        //                  samples = Ha' analysis samples)
        //   synthesis hop Hs  = 256
        //   phi_syn(k) += Omega_k*Hs + dphi(k)*(Hs/Ha') = Omega_k*256 + dphi*ratio
        const float ha = static_cast<float>(hop_) / ratio;
        for (int k = 0; k < num_bins_; ++k) {
            if (is_transient) {
                synth_phase_[k] = phase_[k];
                continue;
            }
            const float omega_k = kTwoPi * static_cast<float>(k) / fft_size_;
            const float dphi = wrap_phase(phase_[k] - prev_phase_in_[c][k]
                                          - omega_k * ha);
            synth_phase_[k] = wrap_phase(prev_phase_out_[c][k]
                                         + omega_k * static_cast<float>(hop_) + dphi * ratio);
        }

        // Peak-locking restricted to the main lobe of detected peaks:
        // Lock bin k to peak kp ONLY if |k - kp| <= 2 bins and mag_[k] is within prominence
        // against the unwarped analysis_envelope_.
        // This prevents spectral valleys and high-frequency noise from locking to harmonic peaks
        // (which turns breath/noise into a metallic, robotic buzz).
        // Skipped on transient frames (phases are already coherent).
        if (num_peaks > 0 && !is_transient) {
            for (int k = 0; k < num_bins_; ++k) {
                const int kp = peak_bins_[peak_owner_[k]];
                if (std::abs(k - kp) <= 2 && mag_[k] > analysis_envelope_[k] * 0.15f) {
                    synth_phase_[k] = wrap_phase(synth_phase_[kp]
                                                 + phase_[k] - phase_[kp]);
                }
            }
        }

        // Magnitudes: replaced envelope x (fine excitation x precomputed
        // excitation shaper — log-octave spectral tilt).
        for (int k = 0; k < num_bins_; ++k)
            synth_mag_[k] = envelope_[k] * (excitation_[k] * excitation_shaper_[k]);

        // Glottal source reshaping: H1 boost and H2 attenuation for feminine morphs
        // (creating the characteristic female H1 >> H2 balance). The H1 harmonic
        // confirmation is also run unconditionally so the detected F0 bin feeds
        // the pitch-synchronous aspiration modulation in reconstruct().
        //
        // H1 identification is deliberately conservative: the first spectral
        // peak is frequently sub-rumble, breath, or a plosive (< 80 Hz), and
        // boosting THAT instead of the fundamental puts +6.8 dB on mud. When
        // a valid F0 is known (external CV or internal NSDF track), the
        // guided search below takes precedence; otherwise:
        //   1. candidates are restricted to the biological H1 window 80-400 Hz
        //      (above it the boost is gated off by kH1MaxHz anyway);
        //   2. a candidate wins only with an overtone at 2k (+-2 bins);
        //   3. brief dropouts hold the last confirmed bin (~21 ms hangover)
        //      instead of snapping onto rumble; longer gaps report -1.
        int h1_bin = -1;
        int h2_bin = -1;
        const float bin_hz = sr_ / static_cast<float>(fft_size_);
        // Harmonic-guided peak tracking: when a valid F0 is known (external
        // CV or internal NSDF), pin the H1 search to k0 = round(F0/df)
        // instead of the blind 80-400 Hz overtone-pair search. The guide is
        // additionally vetoed per-frame by compute_voiced_prob so a held CV
        // through fricatives/silence cannot plant a boost on noise: with no
        // peak near k0 the bin clamps to k0 only on voiced frames, otherwise
        // the search yields to the continuity hangover below. In an external
        // session the blind search is skipped entirely (hangover only), so a
        // stale held F0 never degrades into rumble-boosting; the internal
        // path keeps the blind fallback for frames where NSDF has no pitch.
        const bool ext_session =
            (ext_f0_mode_ == 1.0f) || (ext_f0_mode_ == 2.0f);
        bool guided_done = false;
        if (detected_f0_ >= 60.0f && detected_f0_ <= 1000.0f) {
            const float frame_voiced = compute_voiced_prob(num_peaks);
            if (frame_voiced > 0.25f) {
                const int target_k0 =
                    static_cast<int>(std::round(detected_f0_ / bin_hz));
                if (target_k0 >= 1 && target_k0 < num_bins_) {
                    // Nearest peak within +-2 bins of k0; clamp to k0 itself
                    // when H1 is submerged beneath formant energy.
                    int best_p = -1;
                    int min_dist = 3;
                    for (int p = 0; p < num_peaks; ++p) {
                        const int dist =
                            std::abs(peak_bins_[p] - target_k0);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_p = peak_bins_[p];
                        }
                    }
                    h1_bin = (best_p > 0) ? best_p : target_k0;

                    // H2: nearest peak within +-3 bins of 2*h1.
                    const int target_k1 = 2 * h1_bin;
                    int best_h2 = -1;
                    min_dist = 4;
                    for (int p = 0; p < num_peaks; ++p) {
                        const int dist =
                            std::abs(peak_bins_[p] - target_k1);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_h2 = peak_bins_[p];
                        }
                    }
                    h2_bin = best_h2;
                    last_h1_bin_[c] = h1_bin;
                    h1_hangover_[c] = 4;
                    guided_done = true;
                }
            }
        }
        if (!guided_done && !ext_session && num_peaks > 0) {
            const int kMin = std::max(1, static_cast<int>(80.0f / bin_hz));
            const int kMax = static_cast<int>(400.0f / bin_hz) + 1;

            int confirmed = -1;
            for (int p = 0; p < num_peaks; ++p) {
                const int k = peak_bins_[p];
                if (k < kMin || k > kMax) continue;
                for (int q = 0; q < num_peaks; ++q) {
                    if (std::abs(peak_bins_[q] - 2 * k) <= 2) {
                        confirmed = k;
                        break;
                    }
                }
                if (confirmed > 0) break;
            }
            if (confirmed > 0) {
                h1_bin = confirmed;
                int bestd = 1 << 28;
                for (int p = 0; p < num_peaks; ++p) {
                    const int d = std::abs(peak_bins_[p] - 2 * confirmed);
                    if (d < bestd) { bestd = d; h2_bin = peak_bins_[p]; }
                }
                if (bestd > 3) h2_bin = -1;
                last_h1_bin_[c] = confirmed;
                h1_hangover_[c] = 4;
            }
        }
        if (h1_bin < 0) {
            if (h1_hangover_[c] > 0 && last_h1_bin_[c] > 0) {
                --h1_hangover_[c];
                h1_bin = last_h1_bin_[c];
                h2_bin = -1;  // stale H2 location untrusted: skip the cut
            } else {
                h1_hangover_[c] = 0;
            }
        }
        if (num_peaks > 0) {
            if (gender_morph_ > 0.0f
                    && static_cast<float>(h1_bin) * bin_hz < kH1MaxHz) {
                const float boost = 1.0f + 1.2f * gender_morph_;  // up to +6.8 dB
                for (int d = -2; d <= 2; ++d) {
                    const int kb = h1_bin + d;
                    if (kb >= 1 && kb < num_bins_) {
                        const float w =
                            0.5f * (1.0f + std::cos(kPi * static_cast<float>(d) / 3.0f));
                        synth_mag_[kb] *= (1.0f + (boost - 1.0f) * w);
                    }
                }

                // If H2 was located, attenuate it for feminine morphs
                if (h2_bin > 0) {
                    const float cut = 1.0f - 0.4f * gender_morph_; // down to -4.5 dB
                    for (int d = -2; d <= 2; ++d) {
                        const int kb = h2_bin + d;
                        if (kb >= 1 && kb < num_bins_) {
                            const float w =
                                0.5f * (1.0f + std::cos(kPi * static_cast<float>(d) / 3.0f));
                            synth_mag_[kb] *= (1.0f + (cut - 1.0f) * w);
                        }
                    }
                }
            }
        }

        // Publish the detected F0 bin for the pitch-synchronous aspiration
        // modulation in reconstruct() (consumed within this same frame).
        current_h1_bin_ = h1_bin;

        // Update phase history for the next frame.
        for (int k = 0; k < num_bins_; ++k) {
            prev_phase_in_[c][k] = phase_[k];
            prev_phase_out_[c][k] = synth_phase_[k];
        }
        // Magnitude history for the next frame's transient-flux detection.
        std::memcpy(prev_mag_[c], mag_, sizeof(float) * num_bins_);
    }

    // Composite voiced/unvoiced detector: low-to-high frequency energy ratio
    // combined with speech-band spectral flatness.
    float compute_voiced_prob(int num_peaks) const {
        const float bin_hz = sr_ / static_cast<float>(fft_size_);
        const int k_1500 = static_cast<int>(1500.0f / bin_hz);
        const int k_3500 = static_cast<int>(3500.0f / bin_hz);

        float e_lf = 0.0f;
        for (int k = 1; k < k_1500 && k < num_bins_; ++k)
            e_lf += mag_[k] * mag_[k];

        float e_hf = 0.0f;
        for (int k = k_3500; k < num_bins_; ++k)
            e_hf += mag_[k] * mag_[k];

        const float ratio = (e_lf + 1e-7f) / (e_hf + 1e-7f);
        const float ratio_db = 10.0f * std::log10(ratio);

        const int k_300 = static_cast<int>(300.0f / bin_hz);
        const int k_3000 = static_cast<int>(3000.0f / bin_hz);
        float log_sum = 0.0f;
        float lin_sum = 0.0f;
        int n_sub = 0;
        for (int k = k_300; k < k_3000 && k < num_bins_; ++k) {
            log_sum += std::log(mag_[k] + 1e-9f);
            lin_sum += mag_[k];
            ++n_sub;
        }
        float flatness = 0.5f;
        if (n_sub > 0 && lin_sum > 0.0f) {
            const float geo = std::exp(log_sum / static_cast<float>(n_sub));
            flatness = geo / (lin_sum / static_cast<float>(n_sub));
        }

        int lf_peaks = 0;
        for (int p = 0; p < num_peaks; ++p) {
            if (peak_bins_[p] < k_1500) ++lf_peaks;
        }

        float v_ratio = (ratio_db + 12.0f) / 20.0f;
        if (v_ratio < 0.0f) v_ratio = 0.0f;
        if (v_ratio > 1.0f) v_ratio = 1.0f;

        float v_flat = 1.0f - flatness / 0.35f;
        if (v_flat < 0.0f) v_flat = 0.0f;
        if (v_flat > 1.0f) v_flat = 1.0f;

        float prob = (lf_peaks >= 2) ? (0.65f * v_ratio + 0.35f * v_flat)
                                     : (0.85f * v_ratio + 0.15f * v_flat);
        if (prob < 0.0f) prob = 0.0f;
        if (prob > 1.0f) prob = 1.0f;
        return prob;
    }

    void reconstruct(int c, double fpos, int num_peaks) {
        // Voiced/unvoiced-gated sibilant crossfade + complex-bin blending of
        // the BAND-LIMITED, tract-shaped aspiration noise, then rebuild the
        // full conjugate-symmetric spectrum and invert.
        const float bin_hz = sr_ / static_cast<float>(fft_size_);

        // Dynamic V/UV detection: sibilant dry-bin mixing is active ONLY on
        // unvoiced frames.
        const float voiced_prob = compute_voiced_prob(num_peaks);
        const float unvoiced_weight = (1.0f - voiced_prob) * sibilant_mix_;

        // If unvoiced sibilant bypass is active, read the true UNSHIFTED dry frame
        // from in_ring_ at stride 1.0 and compute its FFT for blending.
        bool has_dry_spec = false;
        if (unvoiced_weight > 0.01f) {
            // fpos is always an exact integer (256*m mod kRingSize), so the
            // stride-1.0 dry read needs no floor/interpolation — frac is
            // identically zero and linear interpolation here is dead math.
            const int fp = static_cast<int>(fpos);
            for (int i = 0; i < fft_size_; ++i)
                dry_time_[i] = in_ring_[c][(fp + i) & kRingMask] * window_[i];
            forward_real(dry_time_, dry_spec_re_, dry_spec_im_);
            has_dry_spec = true;
        }

        // Tract-shaped aspiration: signal-proportional noise scaled by the vocal tract
        // envelope and gated by voicing (natural vocal cord leakage occurs during phonation).
        const int k_low = static_cast<int>(1500.0f * (fft_size_ / sr_));
        const int k_high = static_cast<int>(7000.0f * (fft_size_ / sr_));
        const float breath_scale = breathiness_ * 0.04f * voiced_prob;

        // Pitch-synchronous (glottally modulated) aspiration: blending the noise
        // spectrum with copies shifted by +-k0 (k0 = F0 bin) is the exact
        // frequency-domain equivalent of time-domain gating 1 + 2*beta*cos(w0 t),
        // so the aspiration amplitude pulses once per glottal period instead of
        // sounding like static tape hiss. Falls back to stationary injection on
        // unvoiced frames or when F0 is undetermined.
        bool pitch_sync = (breath_scale > 0.0f && voiced_prob > 0.2f
                           && current_h1_bin_ > 1);
        int k0 = 0;
        int nb_lo = 0;
        int nb_hi = 0;
        const float beta = 0.35f;
        const float sideband_norm = 1.0f / std::sqrt(1.0f + 2.0f * beta * beta);
        if (pitch_sync) {
            k0 = current_h1_bin_;
            if (k0 > fft_size_ / 32) k0 = fft_size_ / 32;   // sanity clamp (F0 <= ~1.5 kHz)
            nb_lo = k_low - k0;
            if (nb_lo < 0) nb_lo = 0;
            nb_hi = k_high + k0;
            if (nb_hi > num_bins_ - 1) nb_hi = num_bins_ - 1;
            for (int k = nb_lo; k <= nb_hi; ++k) {
                rng_state_ ^= rng_state_ << 13;
                rng_state_ ^= rng_state_ >> 17;
                rng_state_ ^= rng_state_ << 5;
                noise_raw_re_[k] =
                    (static_cast<float>(rng_state_ & 0xFFFFu) / 32768.0f) - 1.0f;
                rng_state_ ^= rng_state_ << 13;
                rng_state_ ^= rng_state_ >> 17;
                rng_state_ ^= rng_state_ << 5;
                noise_raw_im_[k] =
                    (static_cast<float>(rng_state_ & 0xFFFFu) / 32768.0f) - 1.0f;
            }
        }

        for (int kd = 0; kd < num_bins_; ++kd) {
            float vr = synth_mag_[kd] * std::cos(synth_phase_[kd]);
            float vi = synth_mag_[kd] * std::sin(synth_phase_[kd]);

            // Complex tract-filtered breath injection (1.5-7 kHz band).
            if (breath_scale > 0.0f && kd >= k_low && kd <= k_high && kd < num_bins_ - 1) {
                float n_re;
                float n_im;
                if (pitch_sync) {
                    // N(k) + beta*N(k-k0) + beta*N(k+k0), energy-normalized.
                    // kd - k0 >= nb_lo and kd + k0 <= nb_hi by construction.
                    n_re = sideband_norm * (noise_raw_re_[kd]
                                            + beta * noise_raw_re_[kd - k0]
                                            + beta * noise_raw_re_[kd + k0]);
                    n_im = sideband_norm * (noise_raw_im_[kd]
                                            + beta * noise_raw_im_[kd - k0]
                                            + beta * noise_raw_im_[kd + k0]);
                } else {
                    rng_state_ ^= rng_state_ << 13;
                    rng_state_ ^= rng_state_ >> 17;
                    rng_state_ ^= rng_state_ << 5;
                    n_re =
                        (static_cast<float>(rng_state_ & 0xFFFFu) / 32768.0f) - 1.0f;
                    rng_state_ ^= rng_state_ << 13;
                    rng_state_ ^= rng_state_ >> 17;
                    rng_state_ ^= rng_state_ << 5;
                    n_im =
                        (static_cast<float>(rng_state_ & 0xFFFFu) / 32768.0f) - 1.0f;
                }

                const float noise_gain = breath_scale * envelope_[kd];
                vr += n_re * noise_gain;
                vi += n_im * noise_gain;
            }

            float band_gain = 0.0f;
            const float f_hz = static_cast<float>(kd) * bin_hz;
            if (f_hz >= kXoverHighHz) {
                band_gain = 1.0f;
            } else if (f_hz > kXoverLowHz) {
                const float mu = (f_hz - kXoverLowHz) / (kXoverHighHz - kXoverLowHz);
                band_gain = 0.5f * (1.0f - std::cos(kPi * mu));
            }
            // Gate the dry sibilant blend by unvoiced-ness: during vowels
            // (voiced_prob ~ 1) the bypass is fully disengaged; on unvoiced frames
            // the true unshifted dry consonant spectrum passes through cleanly.
            const float g = unvoiced_weight * band_gain;

            float yr, yi;
            if (g > 0.0f && has_dry_spec) {
                yr = (1.0f - g) * vr + g * dry_spec_re_[kd];
                yi = (1.0f - g) * vi + g * dry_spec_im_[kd];
            } else {
                yr = vr;
                yi = vi;
            }

            if (!std::isfinite(yr)) yr = 0.0f;   // NaN/Inf -> 0 guard
            if (!std::isfinite(yi)) yi = 0.0f;
            fft_re_[kd] = yr;
            fft_im_[kd] = yi;
        }

        // Real DC / Nyquist + conjugate-symmetric upper half (planar layout).
        fft_im_[0] = 0.0f;
        fft_im_[num_bins_ - 1] = 0.0f;
        for (int k = 1; k < num_bins_ - 1; ++k) {
            fft_re_[fft_size_ - k] = fft_re_[k];
            fft_im_[fft_size_ - k] = -fft_im_[k];
        }
        fft_run(true);

        // Safety clamp on the synthesized frame.
        for (int i = 0; i < fft_size_; ++i) {
            float v = fft_re_[i];
            if (!std::isfinite(v)) v = 0.0f;
            if (v > 4.0f) v = 4.0f;
            if (v < -4.0f) v = -4.0f;
            scratch_time_[i] = v;
        }
    }

    void forward_real(const float* time, float* re513, float* im513) {
        for (int i = 0; i < fft_size_; ++i) { fft_re_[i] = time[i]; fft_im_[i] = 0.0f; }
        fft_run(false);
        for (int k = 0; k < num_bins_; ++k) {
            re513[k] = fft_re_[k];
            im513[k] = fft_im_[k];
        }
        // DC and Nyquist are purely real.
        im513[0] = 0.0f;
        im513[num_bins_ - 1] = 0.0f;
    }

    // Iterative radix-2 complex FFT on the active size. Inverse includes 1/N.
    void fft_run(bool inverse) {
        for (int i = 0; i < fft_size_; ++i) {
            const int j = rev_[i];
            if (j > i) {
                std::swap(fft_re_[i], fft_re_[j]);
                std::swap(fft_im_[i], fft_im_[j]);
            }
        }
        for (int len = 2; len <= fft_size_; len <<= 1) {
            const int half = len >> 1;
            const int step = fft_size_ / len;
            for (int i = 0; i < fft_size_; i += len) {
                for (int j = 0; j < half; ++j) {
                    const int t = j * step;
                    const float wr = cos_tab_[t];
                    const float wi = inverse ? sin_tab_[t] : -sin_tab_[t];
                    const int a = i + j, b = a + half;
                    const float xr = fft_re_[b] * wr - fft_im_[b] * wi;
                    const float xi = fft_re_[b] * wi + fft_im_[b] * wr;
                    fft_re_[b] = fft_re_[a] - xr;
                    fft_im_[b] = fft_im_[a] - xi;
                    fft_re_[a] += xr;
                    fft_im_[a] += xi;
                }
            }
        }
        if (inverse) {
            const float s = 1.0f / fft_size_;
            for (int i = 0; i < fft_size_; ++i) { fft_re_[i] *= s; fft_im_[i] *= s; }
        }
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Standard ANode C-ABI (bound by ffi_base._bind_functions; no extended exports)
// ---------------------------------------------------------------------------

EXPORT void* create(void) {
    return static_cast<void*>(new (std::nothrow) VocalTransformerProcessor());
}

EXPORT void destroy(void* h) {
    delete static_cast<VocalTransformerProcessor*>(h);
}

EXPORT void set_samplerate(void* h, float samplerate) {
    if (h) static_cast<VocalTransformerProcessor*>(h)->set_samplerate(samplerate);
}

EXPORT void set_param(void* h, int id, float value) {
    if (h) static_cast<VocalTransformerProcessor*>(h)->set_param(id, value);
}

EXPORT void reset(void* h) {
    if (h) static_cast<VocalTransformerProcessor*>(h)->reset();
}

EXPORT void process(void* h, const float* in, float* out, int channels, int frames) {
    if (h) static_cast<VocalTransformerProcessor*>(h)->process(in, out, channels, frames);
}

EXPORT int get_latency_samples(void* h) {
    if (!h) return 9216;
    return static_cast<VocalTransformerProcessor*>(h)->latency_samples();
}
