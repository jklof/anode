#include <vector>
#include <cmath>

// Windows Export Macro
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// --- The DSP Logic Class ---
class GainProcessor {
public:
    float volume = 1.0f;

    void process(float* input, float* output, int channels, int frames) {
        // PyTorch buffers are usually planar (CH, FRAMES) or interleaved depending on stride.
        // In ANode core.py, buffers are allocated as (CH, BLOCK_SIZE).
        // Since PyTorch default is row-major (C-style), the data layout is:
        // [Ch1_Sample1, Ch1_Sample2, ... , Ch2_Sample1, Ch2_Sample2, ...]
        
        int total_samples = channels * frames;
        
        // Simple vectorized gain
        for (int i = 0; i < total_samples; ++i) {
            output[i] = input[i] * volume;
        }
    }

    void setParam(int id, float value) {
        if (id == 0) { // defined in Python PARAM_MAP
            volume = value;
        }
    }
};

// --- The C-ABI Bridge (Standard for all plugins) ---

extern "C" {

    EXPORT void* create() {
        return new GainProcessor();
    }

    EXPORT void destroy(void* handle) {
        delete static_cast<GainProcessor*>(handle);
    }

    EXPORT void process(void* handle, float* in_ptr, float* out_ptr, int channels, int frames) {
        auto* dsp = static_cast<GainProcessor*>(handle);
        dsp->process(in_ptr, out_ptr, channels, frames);
    }

    EXPORT void set_param(void* handle, int param_id, float value) {
        auto* dsp = static_cast<GainProcessor*>(handle);
        dsp->setParam(param_id, value);
    }

}