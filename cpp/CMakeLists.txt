cmake_minimum_required(VERSION 3.15)
project(anode_plugins CXX)

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
add_library(gain SHARED gain.cpp)
copy_target_to_directory(gain "${PROJECT_SOURCE_DIR}/../plugins")


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
