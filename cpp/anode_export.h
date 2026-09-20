#pragma once

// Shared C-ABI export macro for all native DSP translation units
// (AGENTS.md section 7). Include this header and prefix all C-ABI
// functions with EXPORT instead of pasting the macro per file.
#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif
