// WorldVoiceTransformer native processor (ANode standard C-ABI).
//
// Hybrid ReIm + WORLD voice transformer (Effects):
//   input ring -> ReIm sample framing (5 ms / 240-sample cadence)
//         -> silence gate -> ReIm DIO + IF + SRH F0
//         -> WORLD CheapTrick (spectral envelope) + WORLD D4C (aperiodicity)
//         -> pitch scale + bilinear all-pass VTLN formant warp
//         -> WORLD sequential real-time synthesizer (Synthesis2)
//         -> output FIFO + latency-aligned dry delay + mix
//
// Division of responsibilities:
//   ReIm (vendored C99 core in cpp/reim): audio_frame cadence, silence
//     gating, F0 tracking. We do NOT use ReIm's Sp/Ap/synthesis.
//   WORLD v1.0.1 (FetchContent): CheapTrick, D4C, synthesisrealtime.
//
// Allocation policy (AGENTS.md §4, relaxed rule): C++ may allocate.
// Per-hop upstream allocations (CheapTrick/D4C FFT temps at ~200 hops/s)
// are acceptable; large state (rings, ReIm contexts, WorldSynthesizer,
// option structs, warp tables, frame slots) is created in
// set_samplerate()/create() and only cleared by reset(). The Python/host
// path still performs zero per-block allocation (see plugin wrapper).
//
// Latency: fixed L_total = 1536 samples (32 ms @ 48 kHz) with an F0 floor
// of 71 Hz (N_FFT = 2048). mix <= 0 is a bit-exact memcpy bypass.

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cstdio>

#include "world/cheaptrick.h"
#include "world/d4c.h"
#include "world/synthesisrealtime.h"

#include "reim/audio_frame.h"
#include "reim/analyze_fo.h"
#include "reim/analyze_silence.h"
#include "reim/vocoder.h"

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

namespace {

constexpr int   kMaxChannels = 2;
constexpr int   kInRingSize = 8192;      // input history + dry delay (power of 2)
constexpr int   kInRingMask = kInRingSize - 1;
constexpr int   kSynthRingSize = 32768;  // synthesized stream ring (power of 2)
constexpr int   kSynthRingMask = kSynthRingSize - 1;
constexpr int   kHop = 240;              // 5.0 ms analysis cadence @ 48 kHz
constexpr int   kLatency = 1024;         // fixed throughput latency (21.3 ms); end-to-end ~= 1556 (32.4 ms)
constexpr int   kExcerptLen = 3072;      // WORLD analysis excerpt (last N samples)
constexpr int   kExcerptCenter = 2048;   // spectrum instant: 1024 samples ago,
                                         // aligned with the ReIm frame center
constexpr int   kSynthBuf = 240;         // Synthesis2 chunk == 1 hop
constexpr int   kSynthPointers = 64;       // parameter queue (320 ms @ 5 ms);
                                           // Synthesis2 needs ~5 frames of
                                           // lookahead, so a short queue
                                           // overflows and starves synthesis
constexpr int   kFrameSlots = 64;        // Sp/Ap slot ring (must be >=
                                         // kSynthPointers: the synthesizer
                                         // retains our frame pointers)
constexpr double kSilenceRms = 1e-4;     // -80 dBFS silence gate
constexpr double kFoFloor = 71.0;
constexpr double kFoCeil = 800.0;
constexpr double kFramePeriodMs = 5.0;

// Param IDs (must match plugins/world_voice_transformer.py PARAM_MAP).
enum ParamId { kPitch = 0, kFormant = 1, kMix = 2, kGainDb = 3, kGender = 4, kBreathiness = 5 };

struct FrameSlot {
    double f0 = 0.0;
    std::vector<double> sp;
    std::vector<double> ap;
    // Persistent pointer cells: AddParameters() RETAINS the double**
    // (synth->spectrogram[pointer] = spectrogram) and dereferences it in
    // later Synthesis2() calls, so these addresses must outlive the
    // push_synth_channel() stack frame. Never pass stack arrays here.
    double* sp_ptr = nullptr;
    double* ap_ptr = nullptr;
};

struct ChannelState {
    // Absolute-indexed input history (dry-delay source).
    std::vector<float> in_ring = std::vector<float>(kInRingSize, 0.0f);
    long long total_received = 0;
    // Synthesized stream (absolute-indexed).
    std::vector<float> synth_ring = std::vector<float>(kSynthRingSize, 0.0f);
    long long synth_generated = 0;
    WorldSynthesizer synth;
    bool synth_ready = false;

    ChannelState() { std::memset(&synth, 0, sizeof(synth)); }
};

// Shared mono analysis stream: one F0/envelope/aperiodicity estimation on
// the downmixed input feeds both channel synthesizers. Halves analysis
// cost (D4C dominates the profile); stereo width is preserved via
// independent per-channel synthesis noise. Mono-duplicated inputs still
// produce bit-identical outputs via the memcmp-copy in process().
struct AnalysisState {
    std::vector<float> mono_ring = std::vector<float>(kInRingSize, 0.0f);
    long long total_received = 0;
    vocoder_context_t* vocoder = nullptr;
    audio_frame_t* frame = nullptr;
    fo_context_t* fo_ctx = nullptr;
    std::vector<double> reim_waveform;  // fftsize + 1 (frame + 1-sample delay)
    std::vector<double> excerpt;
    std::vector<double> logsp;  // reused formant-warp workspace (no per-hop alloc)
    std::vector<FrameSlot> slots;
    int slot_idx = 0;

    AnalysisState() : reim_waveform(2049, 0.0), excerpt(kExcerptLen, 0.0) {}
};

class WorldTransformerProcessor {
public:
    WorldTransformerProcessor()
        : sr_(48000.0f), pitch_st_(0.0f), formant_st_(0.0f),
          mix_(1.0f), prev_mix_(1.0f), gain_db_(0.0f),
          gender_morph_(0.0f), breathiness_(0.0f),
          fft_size_(2048), numbins_(1025),
          profile_(std::getenv("WORLD_PROFILE") != nullptr),
          t_reim_(0), t_ct_(0), t_d4c_(0), t_synth_(0), hops_(0), blocks_(0) {
        InitializeCheapTrickOption(static_cast<int>(sr_), &ct_opt_);
        ct_opt_.f0_floor = kFoFloor;
        fft_size_ = GetFFTSizeForCheapTrick(static_cast<int>(sr_), &ct_opt_);
        ct_opt_.fft_size = fft_size_;
        numbins_ = fft_size_ / 2 + 1;
        InitializeD4COption(&d4c_opt_);
        build_warp_table();
        build_tilt_table();
        setup_analysis();
        for (int c = 0; c < kMaxChannels; ++c) setup_synth(c);
    }

    ~WorldTransformerProcessor() {
        for (int c = 0; c < kMaxChannels; ++c) teardown_synth(c);
        teardown_analysis();
    }

    void set_samplerate(float sr) {
        if (sr <= 0.0f || sr == sr_) return;
        sr_ = sr;
        InitializeCheapTrickOption(static_cast<int>(sr_), &ct_opt_);
        ct_opt_.f0_floor = kFoFloor;
        fft_size_ = GetFFTSizeForCheapTrick(static_cast<int>(sr_), &ct_opt_);
        ct_opt_.fft_size = fft_size_;
        numbins_ = fft_size_ / 2 + 1;
        InitializeD4COption(&d4c_opt_);
        build_warp_table();
        build_tilt_table();
        teardown_analysis();
        setup_analysis();
        for (int c = 0; c < kMaxChannels; ++c) {
            teardown_synth(c);
            setup_synth(c);
        }
    }

    void set_param(int id, float v) {
        switch (id) {
            case kPitch: pitch_st_ = std::max(-12.0f, std::min(12.0f, v)); break;
            case kFormant:
                formant_st_ = std::max(-12.0f, std::min(12.0f, v));
                build_warp_table();
                break;
            case kMix: {
                mix_ = std::max(0.0f, std::min(1.0f, v));
                if ((prev_mix_ <= 0.0f) != (mix_ <= 0.0f)) reset_transient();
                prev_mix_ = mix_;
                break;
            }
            case kGainDb: gain_db_ = std::max(-12.0f, std::min(12.0f, v)); break;
            case kGender:
                gender_morph_ = std::max(-1.0f, std::min(1.0f, v));
                build_warp_table();
                build_tilt_table();
                break;
            case kBreathiness:
                breathiness_ = std::max(0.0f, std::min(1.0f, v));
                break;
            default: break;
        }
    }

    int latency_samples() const { return kLatency; }

    void reset() {
        reset_transient();
        prev_mix_ = mix_;
    }

    void process(const float* in, float* out, int channels, int frames) {
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
        const float gain_lin = std::pow(10.0f, gain_db_ / 20.0f);

        // Push this block into the per-channel histories and the shared
        // mono analysis stream (downmix drives ReIm framing sample by
        // sample at its proven 5 ms cadence).
        for (int i = 0; i < frames; ++i) {
            float mono;
            if (channels == 1) {
                mono = in[i];
                ch_[0].in_ring[(ch_[0].total_received) & kInRingMask] = mono;
                ch_[1].in_ring[(ch_[1].total_received) & kInRingMask] = mono;
            } else {
                const float l = in[i];
                const float r = in[frames + i];
                ch_[0].in_ring[(ch_[0].total_received) & kInRingMask] = l;
                ch_[1].in_ring[(ch_[1].total_received) & kInRingMask] = r;
                mono = 0.5f * (l + r);
            }
            ch_[0].total_received++;
            ch_[1].total_received++;
            an_.mono_ring[(an_.total_received) & kInRingMask] = mono;
            an_.total_received++;
            if (next_audio_frame(an_.frame, static_cast<double>(mono),
                                 an_.reim_waveform.data())) {
                analyze_hop();
            }
        }

        for (int c = 0; c < chs; ++c) {
            float* out_ch = out + c * frames;
            ChannelState& st = ch_[c];

            // Emit latency-aligned output: stream position o was captured
            // L_total samples ago for both wet and dry paths.
            const long long read_base = st.total_received - kLatency;
            for (int i = 0; i < frames; ++i) {
                const long long o = read_base + i;
                float wet = 0.0f;
                float dry = 0.0f;
                if (o >= 0) {
                    dry = st.in_ring[o & kInRingMask];
                    if (o < st.synth_generated)
                        wet = st.synth_ring[o & kSynthRingMask];
                }
                float y = (1.0f - mix_) * dry + mix_ * (wet * gain_lin);
                out_ch[i] = std::isfinite(y) ? y : 0.0f;
            }
        }

        // Mono-duplicated inputs must produce bit-identical stereo outputs:
        // independent per-channel synthesis noise would otherwise diverge.
        // Both pipelines stay aligned (both ran fully above); we just publish
        // channel 0 twice when the inputs were identical.
        if (channels == 2 &&
            std::memcmp(in, in + frames, static_cast<size_t>(frames) * sizeof(float)) == 0) {
            std::memcpy(out + frames, out,
                        static_cast<size_t>(frames) * sizeof(float));
        }
        // Zero any unused output channels (anti-ghosting, defensive).
        for (int c = chs; c < kMaxChannels; ++c)
            std::memset(out + c * frames, 0, static_cast<size_t>(frames) * sizeof(float));

        if (profile_ && ++blocks_ % 200 == 0 && hops_ > 0) {
            std::fprintf(stderr,
                         "[world] blocks=%ld hops=%ld avg/hop ms: reim=%.3f ct=%.3f d4c=%.3f synth=%.3f total=%.3f\n",
                         blocks_, hops_, t_reim_ / hops_, t_ct_ / hops_,
                         t_d4c_ / hops_, t_synth_ / hops_,
                         (t_reim_ + t_ct_ + t_d4c_ + t_synth_) / hops_);
        }
    }

private:
    float sr_;
    float pitch_st_, formant_st_, mix_, prev_mix_, gain_db_;
    float gender_morph_, breathiness_;
    int fft_size_, numbins_;
    bool profile_;
    double t_reim_, t_ct_, t_d4c_, t_synth_;
    long hops_, blocks_;
    long f0_debug_ = 0;
    CheapTrickOption ct_opt_;
    D4COption d4c_opt_;
    std::vector<double> warp_table_;  // dest bin -> source position
    std::vector<double> tilt_table_;  // glottal spectral tilt multiplier per bin
    std::vector<double> last_ap_;     // held aperiodicity (D4C runs half-rate)
    bool ap_valid_ = false;
    long hop_count_ = 0;
    ChannelState ch_[kMaxChannels];
    AnalysisState an_;

    void setup_analysis() {
        an_.reim_waveform.assign(static_cast<size_t>(fft_size_) + 1, 0.0);
        an_.excerpt.assign(kExcerptLen, 0.0);
        an_.logsp.assign(static_cast<size_t>(numbins_), 0.0);
        last_ap_.assign(static_cast<size_t>(numbins_), 1.0);
        ap_valid_ = false;
        hop_count_ = 0;
        an_.slots.clear();
        an_.slots.resize(kFrameSlots);
        for (auto& s : an_.slots) {
            s.f0 = 0.0;
            s.sp.assign(static_cast<size_t>(numbins_), 0.0);
            s.ap.assign(static_cast<size_t>(numbins_), 1.0);
            s.sp_ptr = s.sp.data();
            s.ap_ptr = s.ap.data();
        }
        an_.slot_idx = 0;
        an_.vocoder = create_vocoder_context(kFramePeriodMs,
                                             static_cast<size_t>(fft_size_),
                                             kFoFloor, kFoCeil,
                                             static_cast<double>(sr_));
        an_.frame = create_audio_frame(static_cast<double>(sr_),
                                       kFramePeriodMs,
                                       static_cast<size_t>(fft_size_));
        an_.fo_ctx = create_fo_context(an_.vocoder);
    }

    void teardown_analysis() {
        if (an_.fo_ctx) { destroy_fo_context(&an_.fo_ctx); an_.fo_ctx = nullptr; }
        if (an_.frame) { destroy_audio_frame(&an_.frame); an_.frame = nullptr; }
        if (an_.vocoder) { destroy_vocoder_context(&an_.vocoder); an_.vocoder = nullptr; }
    }

    void setup_synth(int c) {
        ChannelState& st = ch_[c];
        std::memset(&st.synth, 0, sizeof(st.synth));
        InitializeSynthesizer(static_cast<int>(sr_), kFramePeriodMs,
                              fft_size_, kSynthBuf, kSynthPointers,
                              &st.synth);
        st.synth_ready = true;
    }

    void teardown_synth(int c) {
        ChannelState& st = ch_[c];
        if (st.synth_ready) {
            DestroySynthesizer(&st.synth);
            std::memset(&st.synth, 0, sizeof(st.synth));
            st.synth_ready = false;
        }
    }

    void reset_transient() {
        std::fill(an_.mono_ring.begin(), an_.mono_ring.end(), 0.0f);
        an_.total_received = 0;
        an_.slot_idx = 0;
        for (auto& s : an_.slots) {
            s.f0 = 0.0;
            std::fill(s.sp.begin(), s.sp.end(), 0.0);
            std::fill(s.ap.begin(), s.ap.end(), 1.0);
        }
        // ReIm has no reset API: recreate contexts (rare path: transport
        // start / mix-boundary crossing, never steady-state blocks).
        teardown_analysis();
        setup_analysis();
        f0_debug_ = 0;
        for (int c = 0; c < kMaxChannels; ++c) {
            ChannelState& st = ch_[c];
            std::fill(st.in_ring.begin(), st.in_ring.end(), 0.0f);
            std::fill(st.synth_ring.begin(), st.synth_ring.end(), 0.0f);
            st.total_received = 0;
            st.synth_generated = 0;
            teardown_synth(c);
            setup_synth(c);
        }
    }

    void build_warp_table() {
        warp_table_.assign(static_cast<size_t>(numbins_), 0.0);
        // Unified VTLN: manual formant shift plus 3 semitones per unit of
        // gender morph (matches VocalTransformer's gender base).
        const double eff_formant =
            static_cast<double>(formant_st_) + 3.0 * static_cast<double>(gender_morph_);
        const double alpha = -std::max(-0.4, std::min(0.4, eff_formant * 0.025));
        const double n1 = static_cast<double>(numbins_ - 1);
        for (int k = 0; k < numbins_; ++k) {
            const double w = M_PI * static_cast<double>(k) / n1;
            double warped = w;
            if (alpha != 0.0) {
                warped = w + 2.0 * std::atan((alpha * std::sin(w)) /
                                             (1.0 - alpha * std::cos(w)));
            }
            double pos = warped / M_PI * n1;
            if (pos < 0.0) pos = 0.0;
            if (pos > n1) pos = n1;
            warp_table_[static_cast<size_t>(k)] = pos;
        }
    }

    void build_tilt_table() {
        // Glottal spectral tilt, log-octave anchored at 1 kHz and clamped to
        // +-10 dB (mirrors VocalTransformer's excitation shaper, applied here
        // to the warped envelope). Sized to numbins_, never hardcoded.
        tilt_table_.assign(static_cast<size_t>(numbins_), 1.0);
        if (gender_morph_ == 0.0f) return;
        const double f_step = static_cast<double>(sr_) / static_cast<double>(fft_size_);
        const double tilt_db_per_oct = -static_cast<double>(gender_morph_) * 2.5;
        for (int k = 0; k < numbins_; ++k) {
            const double f_hz = static_cast<double>(k) * f_step;
            double tilt_db = 0.0;
            if (f_hz > 0.0) {
                const double octaves = std::log2(std::max(f_hz, 50.0) / 1000.0);
                tilt_db = std::max(-10.0, std::min(10.0, octaves * tilt_db_per_oct));
            }
            tilt_table_[static_cast<size_t>(k)] = std::pow(10.0, tilt_db / 20.0);
        }
    }

    static double interp_log(const std::vector<double>& logsp, double pos, int n) {
        if (pos < 0.0) pos = 0.0;
        const double maxp = static_cast<double>(n - 1);
        if (pos > maxp) pos = maxp;
        const int i = static_cast<int>(pos);
        const int j = (i < n - 1) ? i + 1 : i;
        const double f = pos - static_cast<double>(i);
        return logsp[static_cast<size_t>(i)] +
               (logsp[static_cast<size_t>(j)] - logsp[static_cast<size_t>(i)]) * f;
    }

    void analyze_hop() {
        const double* waveform = an_.reim_waveform.data() + 1;  // current frame
        const double* delayed = an_.reim_waveform.data();       // 1-sample delay

        FrameSlot& slot = an_.slots[static_cast<size_t>(an_.slot_idx)];
        an_.slot_idx = (an_.slot_idx + 1) % kFrameSlots;

        // 1. Silence gate: skip FFTs, synthesize ~silence (zero envelope).
        if (analyze_silence(an_.vocoder, waveform, kSilenceRms)) {
            slot.f0 = 0.0;
            std::fill(slot.sp.begin(), slot.sp.end(), 0.0);
            std::fill(slot.ap.begin(), slot.ap.end(), 1.0);
            push_synth(slot);
            return;
        }

        // 2. F0 via ReIm DIO + instantaneous-frequency refinement + SRH.
        auto t0 = profile_ ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        double f0 = analyze_fo(an_.vocoder, an_.fo_ctx, waveform, delayed);
        if (!std::isfinite(f0) || f0 < kFoFloor || f0 > kFoCeil) f0 = 0.0;
        if (profile_ && f0_debug_ < 8) {
            std::fprintf(stderr, "[world] hop f0=%.2f\n", f0);
            ++f0_debug_;
        }

        // 3. Spectral envelope (CheapTrick) + aperiodicity (D4C), single
        //    frame each (f0_length = 1 → O(window), not O(history)).
        //    Excerpt = last kExcerptLen mono samples; the spectrum instant
        //    is kExcerptCenter samples in, aligned with the ReIm frame center.
        const long long base = an_.total_received - kExcerptLen;
        for (int i = 0; i < kExcerptLen; ++i) {
            long long pos = base + i;
            double x = 0.0;
            if (pos >= 0) x = static_cast<double>(an_.mono_ring[pos & kInRingMask]);
            an_.excerpt[static_cast<size_t>(i)] = x;
        }
        double temporal[1] = { static_cast<double>(kExcerptCenter) /
                               static_cast<double>(sr_) };
        double f0arr[1] = { f0 };
        double* sp_ptrs[1] = { slot.sp.data() };
        double* ap_ptrs[1] = { slot.ap.data() };
        const int fs = static_cast<int>(sr_);
        auto t1 = profile_ ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        CheapTrick(an_.excerpt.data(), kExcerptLen, fs, temporal, f0arr,
                          1, &ct_opt_, sp_ptrs);
        auto t2 = profile_ ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        // D4C dominates the profile, so it runs at half rate (every 2nd
        // hop, 10 ms): band-aperiodicity evolves slowly compared to F0 and
        // the envelope, and the held frame stays valid across one hop.
        ++hop_count_;
        if ((hop_count_ & 1) == 0 || !ap_valid_) {
            D4C(an_.excerpt.data(), kExcerptLen, fs, temporal, f0arr, 1,
                       fft_size_, &d4c_opt_, ap_ptrs);
            last_ap_.assign(slot.ap.begin(), slot.ap.end());
            ap_valid_ = true;
        } else {
            slot.ap = last_ap_;
        }
        auto t3 = profile_ ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        for (int k = 0; k < numbins_; ++k) {
            if (!std::isfinite(slot.sp[static_cast<size_t>(k)]))
                slot.sp[static_cast<size_t>(k)] = 0.0;
            double a = slot.ap[static_cast<size_t>(k)];
            if (!std::isfinite(a)) a = 1.0;
            if (a < 0.0) a = 0.0;
            if (a > 1.0) a = 1.0;
            slot.ap[static_cast<size_t>(k)] = a;
        }

        // 4. Pitch scale (voiced only) + Timbre Path (VTLN warp, glottal
        //    tilt, H1/H2 balance, Ap breathiness injection).
        double target_f0 = (f0 > 0.0)
            ? f0 * std::pow(2.0, static_cast<double>(pitch_st_) / 12.0) : 0.0;

        // Unified formant warp on log Sp; the gate covers gender-only morphs.
        const double eff_formant =
            static_cast<double>(formant_st_) + 3.0 * static_cast<double>(gender_morph_);
        if (eff_formant != 0.0) {
            for (int k = 0; k < numbins_; ++k)
                an_.logsp[static_cast<size_t>(k)] =
                    std::log(slot.sp[static_cast<size_t>(k)] + 1e-12);
            for (int k = 0; k < numbins_; ++k)
                slot.sp[static_cast<size_t>(k)] =
                    std::exp(interp_log(an_.logsp, warp_table_[static_cast<size_t>(k)],
                                        numbins_));
        }

        // Glottal spectral tilt (precomputed per-bin multipliers).
        if (gender_morph_ != 0.0f) {
            for (int k = 0; k < numbins_; ++k)
                slot.sp[static_cast<size_t>(k)] *= tilt_table_[static_cast<size_t>(k)];
        }

        // Dynamic H1/H2 glottal balance, keyed on the target fundamental
        // (where the synthesis excitation actually pulses). Gated to low
        // registers; above ~450 Hz target the boost gracefully bypasses.
        if (target_f0 > 0.0 && gender_morph_ > 0.0f) {
            const double f_step = static_cast<double>(sr_) / static_cast<double>(fft_size_);
            const int k1 = static_cast<int>(std::round(target_f0 / f_step));
            if (k1 >= 1 && target_f0 <= 450.0 && k1 < numbins_) {
                const double boost = 1.0 + 1.2 * static_cast<double>(gender_morph_);
                for (int d = -2; d <= 2; ++d) {
                    const int kb = k1 + d;
                    if (kb >= 1 && kb < numbins_) {
                        const double w =
                            0.5 * (1.0 + std::cos(M_PI * static_cast<double>(d) / 3.0));
                        slot.sp[static_cast<size_t>(kb)] *= (1.0 + (boost - 1.0) * w);
                    }
                }
                const int k2 = 2 * k1;
                if (k2 < numbins_) {
                    const double cut = 1.0 - 0.4 * static_cast<double>(gender_morph_);
                    for (int d = -2; d <= 2; ++d) {
                        const int kb = k2 + d;
                        if (kb >= 1 && kb < numbins_) {
                            const double w =
                                0.5 * (1.0 + std::cos(M_PI * static_cast<double>(d) / 3.0));
                            slot.sp[static_cast<size_t>(kb)] *= (1.0 + (cut - 1.0) * w);
                        }
                    }
                }
            }
        }

        // Native aperiodicity breathiness injection (1.5-7 kHz raised-cosine
        // band). Applied to slot.ap AFTER the last_ap_ clean snapshot above,
        // so held half-rate frames never accumulate drift. WORLD's
        // power-complementary periodic/noise split keeps headroom intact.
        // Synthesis excitation is Gaussian white noise (randn/RandnState),
        // distinct from ReIm's velvet-noise generator.
        if (breathiness_ > 0.0f && f0 > 0.0) {
            const double f_step = static_cast<double>(sr_) / static_cast<double>(fft_size_);
            for (int k = 0; k < numbins_; ++k) {
                const double f_hz = static_cast<double>(k) * f_step;
                if (f_hz >= 1500.0 && f_hz <= 7000.0) {
                    const double mu = (f_hz - 1500.0) / (7000.0 - 1500.0);
                    const double band_weight = 0.5 * (1.0 - std::cos(2.0 * M_PI * mu));
                    const double injection =
                        static_cast<double>(breathiness_) * 0.45 * band_weight;
                    double a = slot.ap[static_cast<size_t>(k)] + injection;
                    if (a > 1.0) a = 1.0;
                    slot.ap[static_cast<size_t>(k)] = a;
                }
            }
        }

        slot.f0 = target_f0;
        auto t4 = profile_ ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point();
        push_synth(slot);
        if (profile_) {
            auto t5 = std::chrono::steady_clock::now();
            using ms = std::chrono::duration<double, std::milli>;
            t_reim_ += ms(t1 - t0).count();
            t_ct_ += ms(t2 - t1).count();
            t_d4c_ += ms(t3 - t2).count();
            t_synth_ += ms(t5 - t4).count();
            ++hops_;
        }
    }

    void push_synth(FrameSlot& slot) {
        for (int c = 0; c < kMaxChannels; ++c) {
            push_synth_channel(c, slot);
        }
    }

    void push_synth_channel(int c, FrameSlot& slot) {
        ChannelState& st = ch_[c];
        // Refresh the cells (defensive: vector storage is stable after
        // setup, but re-taking data() is free) and pass their ADDRESSES:
        // AddParameters retains these pointers past our return, so stack
        // arrays here would dangle.
        slot.sp_ptr = slot.sp.data();
        slot.ap_ptr = slot.ap.data();
        // A full queue must never wipe queued context (RefreshSynthesizer
        // is destructive): with 64 pointers it cannot fill in practice;
        // if it ever does, skip this frame and keep draining.
        if (AddParameters(&slot.f0, 1, &slot.sp_ptr, &slot.ap_ptr, &st.synth) != 1) {
            return;
        }
        int guard = 0;
        while (Synthesis2(&st.synth) != 0) {
            for (int i = 0; i < kSynthBuf; ++i) {
                double v = st.synth.buffer[i];
                float y = (!std::isfinite(v)) ? 0.0f : static_cast<float>(v);
                if (y > 4.0f) y = 4.0f;
                if (y < -4.0f) y = -4.0f;
                st.synth_ring[(st.synth_generated + i) & kSynthRingMask] = y;
            }
            st.synth_generated += kSynthBuf;
            if (++guard > 64) break;
        }
        if (IsLocked(&st.synth) == 1)
            RefreshSynthesizer(&st.synth);
    }
};

WorldTransformerProcessor* self_of(void* h) {
    return static_cast<WorldTransformerProcessor*>(h);
}

}  // namespace

EXPORT void* create() {
    try {
        return new WorldTransformerProcessor();
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
    (void)handle;
    return kLatency;
}
