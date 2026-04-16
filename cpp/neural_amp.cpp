#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <memory>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <condition_variable>
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
    NamProcessor() : _sample_rate(48000.0), _block_size(512), _running(true) {
        _loader_thread = std::thread(&NamProcessor::_loader_loop, this);
    }

    // Destructor: Must wait for any background thread to finish
    ~NamProcessor() {
        _running = false;
        _cv.notify_one();
        if (_loader_thread.joinable()) {
            _loader_thread.join();
        }
    }

    void load_model(const char* nam_path, double sample_rate, int max_block_size) {
        if (!nam_path) return;

        std::lock_guard<std::mutex> lock(_mutex);
        _sample_rate = sample_rate;
        _block_size = max_block_size;
        _pending_path = std::string(nam_path);
        _has_pending = true;
        _cv.notify_one();
    }

    void reset_state() {
        if (_dsp) {
            _dsp->Reset(_sample_rate, _block_size);
        }
    }

    void process(float* inputs, float* outputs, int channels, int frames) {
        // 3. CHECK FOR COMPLETION (Non-blocking RT safe)
        if (_has_staged.load(std::memory_order_acquire)) {
            if (_staged_mutex.try_lock()) {
                if (_has_staged.load(std::memory_order_relaxed)) {
                    _dsp = std::move(_staged_dsp);
                    _has_staged.store(false, std::memory_order_release);
                }
                _staged_mutex.unlock();
            }
        }

        // 4. AUDIO PROCESSING
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
    void _loader_loop() {
        while (_running) {
            std::string path_to_load;
            double sr;
            int bs;
            
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _cv.wait(lock, [this] { return !_running || _has_pending; });
                if (!_running) break;
                
                path_to_load = _pending_path;
                _has_pending = false;
                sr = _sample_rate;
                bs = _block_size;
            }
            
            std::unique_ptr<nam::DSP> new_dsp = nullptr;
            try {
                auto path = std::filesystem::path(path_to_load);
                new_dsp = nam::get_dsp(path);
                if (new_dsp) {
                    new_dsp->Reset(sr, bs);
                }
            } catch(...) {}
            
            {
                std::lock_guard<std::mutex> lock(_staged_mutex);
                _staged_dsp = std::move(new_dsp);
                _has_staged.store(true, std::memory_order_release);
            }
        }
    }

    std::unique_ptr<nam::DSP> _dsp;
    std::unique_ptr<nam::DSP> _staged_dsp;
    std::atomic<bool> _has_staged{false};
    std::mutex _staged_mutex;
    
    std::thread _loader_thread;
    std::mutex _mutex;
    std::condition_variable _cv;
    std::atomic<bool> _running{true};
    bool _has_pending{false};
    std::string _pending_path;
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