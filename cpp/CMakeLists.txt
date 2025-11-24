cmake_minimum_required(VERSION 3.15)
project(anode_plugins CXX)


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

