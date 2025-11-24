#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <memory>
#include <filesystem>
#include <cstring> // for memcpy
#include <stdexcept>

// Include NAM core headers
#include "NAM/dsp.h" 
#include "NAM/get_dsp.h" 

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// Forward declare NAM_SAMPLE
// NOTE: Ensure your build system defines NAM_SAMPLE_FLOAT
#ifdef NAM_SAMPLE_FLOAT
    #define NAM_SAMPLE float
#else
    #define NAM_SAMPLE double
#endif

class NamProcessor {
public:
    NamProcessor() : _sample_rate(48000.0) {}

    void load_model(const char* nam_path, double sample_rate, int max_block_size) {
        if (!nam_path)
            return;
        std::string path_str(nam_path);
        
        // Prevent reloading if identical
        if (_dsp && path_str == _nam_file && 
            std::abs(sample_rate - _sample_rate) < 1e-6) {
            return;
        }

        try {
            _dsp = nam::get_dsp(std::filesystem::path(path_str));
            _nam_file = path_str;
            
            if (_dsp) {
                _sample_rate = sample_rate;
                _dsp->Reset(_sample_rate, max_block_size); 
            }
        } catch (const std::exception& e) {
            _dsp.reset();
        }
    }

    void process(float* inputs, float* outputs, int channels, int frames) {
        // Safety / Silence on error
        if (!_dsp) {
            std::memset(outputs, 0, channels * frames * sizeof(float));
            return;
        }

        // 1. Cast pointers (Assumes NAM_SAMPLE is float)
        NAM_SAMPLE* mono_in = (NAM_SAMPLE*)inputs; 
        NAM_SAMPLE* mono_out = (NAM_SAMPLE*)outputs;

        // 2. Process Mono (Channel 0)
        try {
            _dsp->process(mono_in, mono_out, frames);
        } catch (...) {
            _dsp.reset();
            std::memset(outputs, 0, channels * frames * sizeof(float));
            return;
        }

        // 3. broadcast mono -> stereo
        for (int c = 1; c < channels; ++c) {
            // Calculate offset for the next channel
            float* dest_ptr = outputs + (c * frames);
            // Copy from Channel 0
            std::memcpy(dest_ptr, mono_out, frames * sizeof(float));
        }
    }

private:
    std::string _nam_file;
    std::unique_ptr<nam::DSP> _dsp;
    double _sample_rate;
};

extern "C" {

    EXPORT void* create() {
        return new (std::nothrow) NamProcessor();
    }

    EXPORT void destroy(void* handle) {
        if (handle) delete static_cast<NamProcessor*>(handle);
    }

    EXPORT void process(void* handle, float* in_ptr, float* out_ptr, int channels, int frames) {
        auto* proc = static_cast<NamProcessor*>(handle);
        proc->process(in_ptr, out_ptr, channels, frames);
    }

    EXPORT void set_param(void* handle, int param_id, float value) {
    }

    EXPORT void load_nam_model(void* handle, const char* path, double sample_rate, int max_block_size) {
        auto* proc = static_cast<NamProcessor*>(handle);
        proc->load_model(path, sample_rate, max_block_size);
    }
}