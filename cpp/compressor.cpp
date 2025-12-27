#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <new>

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

constexpr int SIDECHAIN_DOWNSAMPLE_FACTOR = 16;
constexpr float EPSILON = 1e-9f;

class CompressorProcessor {
public:
    CompressorProcessor()
        : _samplerate(48000.0f), // ANode default
          _threshold_db(-20.0f),
          _ratio(4.0f),
          _knee_db(6.0f),
          _attack_ms(10.0f),
          _release_ms(100.0f),
          _envelope(0.0f),
          _delay_samples(SIDECHAIN_DOWNSAMPLE_FACTOR / 2),
          _write_head(0),
          _max_channels(2) {
        
        recalc_coeffs();
        resize_buffers(_max_channels);
    }

    void set_param(int id, float value) {
        switch(id) {
            case 0: _threshold_db = value; break;
            case 1: _ratio = std::max(1.0f, value); break;
            case 2: _attack_ms = std::max(0.1f, value); recalc_coeffs(); break;
            case 3: _release_ms = std::max(1.0f, value); recalc_coeffs(); break;
            case 4: _knee_db = std::max(0.0f, value); break;
            case 5: _makeup_gain_db = value; break; // Added makeup gain
        }
    }

    // ANode buffers are flat: [Ch0_Samples..., Ch1_Samples...]
    // We process them using the flat pointers directly to avoid allocations.
    void process(float* in_flat, float* sc_flat, float* out_flat, int channels, int frames) {
        if (!in_flat || !out_flat) return;

        if (channels != _max_channels) {
            resize_buffers(channels);
        }

        // 1. Prepare Pointers for "Planar" access within flat buffer
        std::vector<float*> in_ptrs(channels);
        std::vector<float*> sc_ptrs(channels);
        std::vector<float*> out_ptrs(channels);

        for (int c = 0; c < channels; ++c) {
            in_ptrs[c] = in_flat + (c * frames);
            out_ptrs[c] = out_flat + (c * frames);
            if (sc_flat) {
                sc_ptrs[c] = sc_flat + (c * frames);
            } else {
                sc_ptrs[c] = nullptr; // Use input for detection
            }
        }

        // 2. Prepare analysis buffers
        if (power_sidechain.size() < static_cast<size_t>(frames)) power_sidechain.resize(frames);
        if (gain_reduction_linear.size() < static_cast<size_t>(frames)) gain_reduction_linear.resize(frames);

        // 3. Level Detection (Max across channels)
        for (int i = 0; i < frames; ++i) {
            float max_val = 0.0f;
            for (int ch = 0; ch < channels; ++ch) {
                float sample;
                if (sc_ptrs[ch]) {
                    sample = sc_ptrs[ch][i];
                } else {
                    sample = in_ptrs[ch][i];
                }
                max_val = std::max(max_val, std::abs(sample));
            }
            power_sidechain[i] = max_val * max_val;
        }

        // 4. Envelope & Gain Calc (Downsampled)
        float current_linear_gain = 1.0f;
        float slope = 1.0f / _ratio - 1.0f;
        float knee_start = _threshold_db - _knee_db / 2.0f;
        float knee_end = _threshold_db + _knee_db / 2.0f;
        float makeup_linear = std::pow(10.0f, _makeup_gain_db / 20.0f);

        for (int i = 0; i < frames; ++i) {
            if (i % SIDECHAIN_DOWNSAMPLE_FACTOR == 0) {
                float avg_power = 0.0f;
                int end_idx = std::min(i + SIDECHAIN_DOWNSAMPLE_FACTOR, frames);
                for (int j = i; j < end_idx; ++j) avg_power += power_sidechain[j];
                avg_power /= static_cast<float>(end_idx - i);

                float target = avg_power; // Power is already squared logic
                // Simple RMS-like ballistic
                float coeff = (target > _envelope) ? _attack_coeff : _release_coeff;
                _envelope = target + coeff * (_envelope - target);

                float envelope_db = 10.0f * std::log10(_envelope + EPSILON);
                float gr_db = 0.0f;

                if (envelope_db > knee_end) {
                    gr_db = (envelope_db - _threshold_db) * slope;
                } else if (envelope_db > knee_start) {
                    float x = envelope_db - knee_start;
                    gr_db = (slope / (2.0f * std::max(EPSILON, _knee_db))) * x * x;
                }

                current_linear_gain = std::pow(10.0f, gr_db / 20.0f);
            }
            gain_reduction_linear[i] = current_linear_gain;
        }
        
        // Store last GR for UI
        _last_gr = current_linear_gain;

        // 5. Apply Gain + Delay + Makeup
        for (int i = 0; i < frames; ++i) {
            int read_head = (_write_head - (_delay_samples - 1) + _delay_samples) % _delay_samples;
            
            for (int ch = 0; ch < channels; ++ch) {
                size_t write_idx = static_cast<size_t>(ch) * _delay_samples + _write_head;
                size_t read_idx = static_cast<size_t>(ch) * _delay_samples + read_head;

                float delayed_sample = _delay_buffer[read_idx];
                
                // Output = Delayed Input * GR * Makeup
                out_ptrs[ch][i] = delayed_sample * gain_reduction_linear[i] * makeup_linear;
                
                // Write Input to Delay
                _delay_buffer[write_idx] = in_ptrs[ch][i];
            }
            _write_head = (_write_head + 1) % _delay_samples;
        }
    }

    float get_last_gr() const { return _last_gr; }

private:
    void recalc_coeffs() {
        float ds_rate = _samplerate / static_cast<float>(SIDECHAIN_DOWNSAMPLE_FACTOR);
        float att_s = _attack_ms / 1000.0f;
        float rel_s = _release_ms / 1000.0f;
        _attack_coeff = (att_s > EPSILON) ? std::exp(-1.0f / (ds_rate * att_s)) : 0.0f;
        _release_coeff = (rel_s > EPSILON) ? std::exp(-1.0f / (ds_rate * rel_s)) : 0.0f;
    }

    void resize_buffers(int channels) {
        _max_channels = channels;
        _delay_buffer.assign(static_cast<size_t>(_max_channels) * _delay_samples, 0.0f);
        _write_head = 0;
    }

    float _samplerate;
    float _threshold_db;
    float _ratio;
    float _knee_db;
    float _attack_ms;
    float _release_ms;
    float _makeup_gain_db = 0.0f;

    float _attack_coeff;
    float _release_coeff;
    float _envelope;
    float _last_gr = 1.0f;

    std::vector<float> _delay_buffer;
    std::vector<float> power_sidechain;
    std::vector<float> gain_reduction_linear;
    
    int _delay_samples;
    int _write_head;
    int _max_channels;
};

// --- Exports ---

EXPORT void* create() {
    return new (std::nothrow) CompressorProcessor();
}

EXPORT void destroy(void* handle) {
    if (handle) delete static_cast<CompressorProcessor*>(handle);
}

// Standard ANode API (no sidechain)
EXPORT void process(void* handle, float* in, float* out, int channels, int frames) {
    static_cast<CompressorProcessor*>(handle)->process(in, nullptr, out, channels, frames);
}

// Extended API (with sidechain)
EXPORT void process_with_sidechain(void* handle, float* in, float* sc, float* out, int channels, int frames) {
    static_cast<CompressorProcessor*>(handle)->process(in, sc, out, channels, frames);
}

EXPORT void set_param(void* handle, int param_id, float value) {
    static_cast<CompressorProcessor*>(handle)->set_param(param_id, value);
}

EXPORT float get_gain_reduction(void* handle) {
    return static_cast<CompressorProcessor*>(handle)->get_last_gr();
}