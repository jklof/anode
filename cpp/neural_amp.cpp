#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <memory>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <atomic>

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

    ~NamProcessor() {}  // nothing to join anymore

    void load_model_sync(const char* nam_path, double sample_rate, int max_block_size) {
        if (!nam_path) return;
        std::unique_ptr<nam::DSP> new_dsp = nullptr;
        try {
            new_dsp = nam::get_dsp(std::filesystem::path(nam_path));
            if (new_dsp) new_dsp->Reset(sample_rate, max_block_size);
        } catch (...) {}
        std::lock_guard<std::mutex> lock(_staged_mutex);
        _staged_dsp = std::move(new_dsp);
        _has_staged.store(true, std::memory_order_release);
    }

    void reset_state() {
        if (_dsp) {
            _dsp->Reset(_sample_rate, _block_size);
        }
    }

    void process(float* inputs, float* outputs, int channels, int frames) {
        // 1. CHECK FOR COMPLETION (Non-blocking RT safe)
        if (_has_staged.load(std::memory_order_acquire)) {
            if (_staged_mutex.try_lock()) {
                if (_has_staged.load(std::memory_order_relaxed)) {
                    _dsp = std::move(_staged_dsp);
                    _has_staged.store(false, std::memory_order_release);
                }
                _staged_mutex.unlock();
            }
        }

        // 2. AUDIO PROCESSING
        // If no model is loaded, PASS THROUGH audio (Bypass)
        if (!_dsp) {
            std::memcpy(outputs, inputs, channels * frames * sizeof(float));
            return;
        }

        NAM_SAMPLE* mono_in_ptr = (NAM_SAMPLE*)inputs; 
        NAM_SAMPLE* mono_out_ptr = (NAM_SAMPLE*)outputs;
        NAM_SAMPLE* in_channels[1]  = { mono_in_ptr };
        NAM_SAMPLE* out_channels[1] = { mono_out_ptr };

        try {
            // Process Channel 0 (Mono)
            _dsp->process(in_channels, out_channels, frames);
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
            std::memcpy(dest_ptr, mono_out_ptr, frames * sizeof(float));
        }
    }

private:
    std::unique_ptr<nam::DSP> _dsp;
    std::unique_ptr<nam::DSP> _staged_dsp;
    std::atomic<bool> _has_staged{false};
    std::mutex _staged_mutex;
    
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
    
    EXPORT void load_model_sync(void* handle, const char* path, double sr, int bs) {
        static_cast<NamProcessor*>(handle)->load_model_sync(path, sr, bs);
    }

    EXPORT void reset(void* handle) {
        static_cast<NamProcessor*>(handle)->reset_state();
    }
}