#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <memory>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <future>
#include <chrono>

// Include NAM core headers
#include "NAM/dsp.h" 
#include "NAM/get_dsp.h" 

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

#ifdef NAM_SAMPLE_FLOAT
    #define NAM_SAMPLE float
#else
    #define NAM_SAMPLE double
#endif

class NamProcessor {
public:
    NamProcessor() : _sample_rate(48000.0), _block_size(512) {}

    // Destructor: Must wait for any background thread to finish
    ~NamProcessor() {
        if (_pending_load.valid()) {
            _pending_load.wait();
        }
    }

    void load_model(const char* nam_path, double sample_rate, int max_block_size) {
        if (!nam_path) return;

        // 1. REFUSE IF BUSY
        // If a thread is already running, ignore this request to prevent queuing/blocking.
        if (_pending_load.valid()) {
            return;
        }

        // Save config for resets
        _sample_rate = sample_rate;
        _block_size = max_block_size;

        std::string path_str(nam_path);

        // 2. LAUNCH ASYNC TASK
        // Use std::launch::async to force a new thread
        _pending_load = std::async(std::launch::async, 
            [path_str, sample_rate, max_block_size]() -> std::unique_ptr<nam::DSP> {
                try {
                    auto dsp = nam::get_dsp(std::filesystem::path(path_str));
                    if (dsp) {
                        dsp->Reset(sample_rate, max_block_size);
                    }
                    return dsp;
                } catch (...) {
                    return nullptr; 
                }
            }
        );
    }

    void reset_state() {
        if (_dsp) {
            _dsp->Reset(_sample_rate, _block_size);
        }
    }

    void process(float* inputs, float* outputs, int channels, int frames) {
        // 3. CHECK FOR COMPLETION (Non-blocking)
        if (_pending_load.valid()) {
            // Check status immediately
            auto status = _pending_load.wait_for(std::chrono::seconds(0));
            
            if (status == std::future_status::ready) {
                try {
                    auto new_dsp = _pending_load.get();
                    if (new_dsp) {
                        // Atomic-like swap of the DSP engine
                        _dsp = std::move(new_dsp);
                    }
                } catch (...) {
                    // Ignore load errors, continue with old model or pass-through
                }
            }
        }

        // 4. AUDIO PROCESSING
        // If no model is loaded, PASS THROUGH audio (Bypass)
        if (!_dsp) {
            std::memcpy(outputs, inputs, channels * frames * sizeof(float));
            return;
        }

        NAM_SAMPLE* mono_in = (NAM_SAMPLE*)inputs; 
        NAM_SAMPLE* mono_out = (NAM_SAMPLE*)outputs;

        try {
            // Process Channel 0 (Mono)
            _dsp->process(mono_in, mono_out, frames);
        } catch (...) {
            // If the DSP crashes, reset it and pass through
            _dsp.reset();
            std::memcpy(outputs, inputs, channels * frames * sizeof(float));
            return;
        }

        // Broadcast Channel 0 -> Stereo
        // (NAM is mono; we duplicate the result to other channels)
        for (int c = 1; c < channels; ++c) {
            float* dest_ptr = outputs + (c * frames);
            std::memcpy(dest_ptr, mono_out, frames * sizeof(float));
        }
    }

private:
    std::unique_ptr<nam::DSP> _dsp;
    std::future<std::unique_ptr<nam::DSP>> _pending_load;
    double _sample_rate;
    int _block_size;
};

// --- C-ABI ---
extern "C" {
    EXPORT void* create() { return new (std::nothrow) NamProcessor(); }
    EXPORT void destroy(void* handle) { if (handle) delete static_cast<NamProcessor*>(handle); }
    EXPORT void process(void* handle, float* in, float* out, int ch, int fr) {
        static_cast<NamProcessor*>(handle)->process(in, out, ch, fr);
    }
    EXPORT void set_param(void* handle, int param_id, float value) {}
    
    EXPORT void load_nam_model(void* handle, const char* path, double sr, int bs) {
        static_cast<NamProcessor*>(handle)->load_model(path, sr, bs);
    }

    EXPORT void reset(void* handle) {
        static_cast<NamProcessor*>(handle)->reset_state();
    }
}