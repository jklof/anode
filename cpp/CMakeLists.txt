cmake_minimum_required(VERSION 3.15)
project(anode_plugins C CXX)

# Shared (.so) plugins link a static WORLD below; non-PIC archives fail
# to link on x86-64, so force PIC globally (harmless for the other targets).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- fetch nam core ----
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    # CMP0169 was introduced in 3.24, so only set it if the version is high enough
    cmake_policy(SET CMP0169 OLD) 
endif()

include(FetchContent)
FetchContent_Declare(
    NAM_CORE
    GIT_REPOSITORY https://github.com/sdatkinson/NeuralAmpModelerCore.git
    GIT_TAG        main
    # The source files will be downloaded to: ${nam_core_SOURCE_DIR}
)


# --- Reusable function to copy target to directory ---
function(copy_target_to_directory target_name destination_dir)
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${destination_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "$<TARGET_FILE:${target_name}>"
            "${destination_dir}/$<TARGET_FILE_NAME:${target_name}>"
        COMMENT "Moving ${target_name} to ${destination_dir}"
    )
endfunction()

# --- Target: Delay Processor ---
add_library(delay SHARED delay.cpp)
copy_target_to_directory(delay "${PROJECT_SOURCE_DIR}/../plugins")

# --- Target: Compressor ---
add_library(compressor SHARED compressor.cpp)
copy_target_to_directory(compressor "${PROJECT_SOURCE_DIR}/../plugins")

# --- Target: Biquad Filter (IIR) ---
add_library(biquad SHARED biquad.cpp)
target_compile_features(biquad PRIVATE cxx_std_17)
copy_target_to_directory(biquad "${PROJECT_SOURCE_DIR}/../plugins")

# --- Target: Linear Phase EQ (FIR) ---
add_library(fir_eq SHARED fir_eq.cpp)
target_compile_features(fir_eq PRIVATE cxx_std_17)
copy_target_to_directory(fir_eq "${PROJECT_SOURCE_DIR}/../plugins")


# --- Target: Envelope Follower (CV/Gate extractor) ---
add_library(envelope SHARED envelope.cpp)
target_compile_features(envelope PRIVATE cxx_std_17)
copy_target_to_directory(envelope "${PROJECT_SOURCE_DIR}/../plugins")

# --- Target: Chorus / Flanger ---
add_library(chorus SHARED chorus.cpp)
target_compile_features(chorus PRIVATE cxx_std_17)
copy_target_to_directory(chorus "${PROJECT_SOURCE_DIR}/../plugins")


# --- Target: Karplus-Strong Physical Modeling Synth ---
add_library(karplus SHARED karplus.cpp)
target_compile_features(karplus PRIVATE cxx_std_17)
copy_target_to_directory(karplus "${PROJECT_SOURCE_DIR}/../plugins")

# --- Target: 6-Stage Stereo Phaser ---
add_library(phaser SHARED phaser.cpp)
target_compile_features(phaser PRIVATE cxx_std_17)
copy_target_to_directory(phaser "${PROJECT_SOURCE_DIR}/../plugins")


# --- Target: NAM Processor ---
FetchContent_Populate(NAM_CORE)
set(NAM_PATH ${nam_core_SOURCE_DIR})
set(NAM_DEPS_PATH ${nam_core_SOURCE_DIR}/Dependencies)
file(GLOB_RECURSE NAM_SOURCES ${NAM_PATH}/NAM/*.cpp  ${NAM_PATH}/NAM/*.c  ${NAM_PATH}/NAM/*.h)
add_library(neural_amp SHARED neural_amp.cpp ${NAM_SOURCES})
target_include_directories(neural_amp PRIVATE ${NAM_PATH} ${NAM_DEPS_PATH}/eigen ${NAM_DEPS_PATH}/nlohmann)
target_compile_features(neural_amp PUBLIC cxx_std_20)
target_compile_definitions(neural_amp PRIVATE NAM_SAMPLE_FLOAT)

copy_target_to_directory(neural_amp "${PROJECT_SOURCE_DIR}/../plugins")

# --- Target: Vocal Transformer (spectral vocal core + retune front end) ---
add_library(vocal_transformer SHARED vocal_transformer.cpp)
target_compile_features(vocal_transformer PRIVATE cxx_std_17)
copy_target_to_directory(vocal_transformer "${PROJECT_SOURCE_DIR}/../plugins")

# --- Target: WORLD Voice Transformer (ReIm framing/F0 + WORLD analysis/synth) ---
# WORLD: official vocoder (CheapTrick/D4C/synthesisrealtime), fetched at
# configure time. ReIm: vendored C99 core in cpp/reim (MIT), used for
# sample-accurate framing, silence gating, and DIO+IF+SRH F0 tracking.
set(WORLD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WORLD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/world/CMakeLists.txt")
    # Offline fallback: vendored WORLD checkout at cpp/world
    add_subdirectory(world)
    set(WORLD_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/world/src")
else()
    FetchContent_Declare(
        WORLD_CORE
        GIT_REPOSITORY https://github.com/mmorise/World.git
        GIT_TAG        v1.0.1
    )
    FetchContent_MakeAvailable(WORLD_CORE)
    set(WORLD_SRC_DIR "${world_core_SOURCE_DIR}/src")
endif()
file(GLOB REIM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/reim/src/*.c")
add_library(world_transformer SHARED world_transformer.cpp ${REIM_SOURCES})
target_link_libraries(world_transformer PRIVATE world)
target_include_directories(world_transformer PRIVATE
    ${WORLD_SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/reim/include
)
target_compile_features(world_transformer PRIVATE cxx_std_17)
copy_target_to_directory(world_transformer "${PROJECT_SOURCE_DIR}/../plugins")

# --- Target: De-Esser (split-band dynamic HF attenuator) ---
add_library(deesser SHARED deesser.cpp)
target_compile_features(deesser PRIVATE cxx_std_17)
copy_target_to_directory(deesser "${PROJECT_SOURCE_DIR}/../plugins")

# --- Target: Signalsmith Vocal Processor (sub-band pitch/formant shifter) ---
# Pinned to main@57b93f4 (header version 1.3.2; upstream publishes no v1.3.x
# tag). The INTERFACE target auto-fetches signalsmith-linear 0.3.1.
FetchContent_Declare(
    signalsmith_stretch
    GIT_REPOSITORY https://github.com/Signalsmith-Audio/signalsmith-stretch.git
    GIT_TAG        57b93f4e9206a089a45387eaa39bdc9f310d3308
)
FetchContent_MakeAvailable(signalsmith_stretch)
add_library(signalsmith_vocal SHARED signalsmith_vocal.cpp)
target_compile_features(signalsmith_vocal PRIVATE cxx_std_17)
target_link_libraries(signalsmith_vocal PRIVATE signalsmith-stretch)
# The root header (signalsmith-stretch.h) lives at the repo root, while the
# INTERFACE include dir points at <src>/include: cover both layouts.
target_include_directories(signalsmith_vocal PRIVATE ${signalsmith_stretch_SOURCE_DIR})
copy_target_to_directory(signalsmith_vocal "${PROJECT_SOURCE_DIR}/../plugins")
