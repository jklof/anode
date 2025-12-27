#include <vector>
#include <cmath>
#include <algorithm>
#include <new>

#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

// A reasonable maximum delay buffer (e.g., 5 seconds at 48k)
// This prevents reallocation during runtime which causes audio glitches.
constexpr int MAX_DELAY_SECONDS = 5;

class DelayProcessor {
public:
    DelayProcessor()
        : _samplerate(48000.0f),
          _time_ms(250.0f),
          _feedback(0.5f),
          _mix(0.5f),
          _write_head(0),
          _max_channels(2) {
        
        // Pre-calculate buffer size
        _buffer_size = static_cast<int>(_samplerate * MAX_DELAY_SECONDS);
        _delay_buffer.resize(_buffer_size * _max_channels, 0.0f);
        update_delay_samples();
    }

    void set_param(int id, float value) {
        switch(id) {
            case 0: // Time (ms)
                _time_ms = std::max(0.0f, value); 
                update_delay_samples(); 
                break;
            case 1: // Feedback
                _feedback = std::max(0.0f, std::min(value, 1.1f)); // Allow slight self-oscillation
                break;
            case 2: // Mix
                _mix = std::max(0.0f, std::min(value, 1.0f)); 
                break;
        }
    }

    void process(float* in_flat, float* out_flat, int channels, int frames) {
        if (!in_flat || !out_flat || frames <= 0) return;

        // Resize if channel count increased (rare, but possible)
        if (channels > _max_channels) {
            _max_channels = channels;
            // Note: This allocation causes a glitch, but only happens once upon connection change
            _delay_buffer.resize(_buffer_size * _max_channels, 0.0f);
        }

        // We use a flat buffer approach.
        // Input:  [Ch0_0, Ch0_1... | Ch1_0, Ch1_1...]
        // Delay:  [Ch0_DelayBuf... | Ch1_DelayBuf...]
        
        int current_write_head = _write_head;

        // Optimization: Loop Channels Outer, Samples Inner for cache locality
        for (int ch = 0; ch < channels; ++ch) {
            float* in_ptr = in_flat + (ch * frames);
            float* out_ptr = out_flat + (ch * frames);
            float* delay_ptr = _delay_buffer.data() + (ch * _buffer_size); // Pointer to this channel's delay line

            int local_write = current_write_head;

            for (int i = 0; i < frames; ++i) {
                // 1. Calculate Read Head
                float read_pos = static_cast<float>(local_write) - _delay_samples;
                
                // Wrap logic for float
                while (read_pos < 0.0f) read_pos += _buffer_size;
                while (read_pos >= _buffer_size) read_pos -= _buffer_size;

                // Linear Interpolation
                int idx_a = static_cast<int>(read_pos);
                int idx_b = (idx_a + 1);
                if (idx_b >= _buffer_size) idx_b = 0;
                
                float frac = read_pos - idx_a;

                float wet = delay_ptr[idx_a] * (1.0f - frac) + delay_ptr[idx_b] * frac;
                float dry = in_ptr[i];

                // 2. Output
                out_ptr[i] = dry * (1.0f - _mix) + wet * _mix;

                // 3. Feedback Loop -> Buffer
                float fb_val = dry + (wet * _feedback);
                
                // Soft clipping in feedback loop to prevent explosion
                if (fb_val > 2.0f) fb_val = 2.0f;
                else if (fb_val < -2.0f) fb_val = -2.0f;

                delay_ptr[local_write] = fb_val;

                // Advance
                local_write++;
                if (local_write >= _buffer_size) local_write = 0;
            }
        }

        // Advance global write head by the number of frames processed
        _write_head = (_write_head + frames) % _buffer_size;
    }

private:
    void update_delay_samples() {
        // Convert ms to samples
        float samples = (_time_ms / 1000.0f) * _samplerate;
        // Clamp to buffer limit minus padding
        if (samples > (_buffer_size - 100)) samples = (_buffer_size - 100);
        _delay_samples = samples;
    }

    std::vector<float> _delay_buffer;
    int _buffer_size;
    int _max_channels;
    int _write_head;

    float _samplerate;
    float _time_ms;
    float _delay_samples;
    float _feedback;
    float _mix;
};

// --- Exports ---

EXPORT void* create() {
    return new (std::nothrow) DelayProcessor();
}

EXPORT void destroy(void* handle) {
    if (handle) delete static_cast<DelayProcessor*>(handle);
}

EXPORT void process(void* handle, float* in, float* out, int channels, int frames) {
    static_cast<DelayProcessor*>(handle)->process(in, out, channels, frames);
}

EXPORT void set_param(void* handle, int param_id, float value) {
    static_cast<DelayProcessor*>(handle)->set_param(param_id, value);
}