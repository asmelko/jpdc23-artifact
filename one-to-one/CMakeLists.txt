cmake_minimum_required(VERSION 3.18)

# Handle superbuild first
option (USE_SUPERBUILD "Whether or not a superbuild should be invoked" OFF)
if (USE_SUPERBUILD)
    project (SUPERBUILD NONE)
    # execute the superbuild (this script will be invoked again without the
    # USE_SUPERBUILD option this time)
    include (cmake/superbuild.cmake)
    return()                      # stop processing this file further
else()
    project(CROSS LANGUAGES C CXX CUDA)
endif()

# List in format <option_name>:<release value>:<debug value>:<description>
set(ALG_LIMIT_OPTIONS
        "SHUFFLE_MULTIMAT_RIGHT_RIGHT_MATRICES_PER_THREAD_LIMIT:8:4:Maximum value of num_right_rows to generate code for"

        "SHUFFLE_MULTIROW_RIGHT_RIGHT_ROWS_LIMIT:8:4:Maximum value of right_rows to generate code for"

        "SHUFFLE_MULTIROW_BOTH_SHIFTS_PER_THREAD_LIMIT:8:4:Maximum value of shifts_per_thread to generate code for"
        "SHUFFLE_MULTIROW_BOTH_LEFT_ROWS_LIMIT:8:2:Maximum value of left_rows to generate code for"

        "SHUFFLE_MULTIROW_BOTH_LOCAL_MEM_SHIFTS_PER_THREAD_LIMIT:1:4:Maximum value of shifts_per_thread to generate code for"
        "SHUFFLE_MULTIROW_BOTH_LOCAL_MEM_LEFT_ROWS_LIMIT:1:2:Maximum value of left_rows to generate code for"

        "SHUFFLE_MULTIROW_RIGHT_MULTIMAT_RIGHT_RIGHT_ROWS_LIMIT:1:2:Maximum value of right_rows to generate code for"
        "SHUFFLE_MULTIROW_RIGHT_MULTIMAT_RIGHT_RIGHT_MATS_LIMIT:1:2:Maximum value of right_mats to generate code for"

        "SHUFFLE_N_TO_M_MULTIMAT_BOTH_LEFT_MATRICES_PER_THREAD_LIMIT:1:2:Maximum value of left_matrices_per_thread to generate code for"
        "SHUFFLE_N_TO_M_MULTIMAT_BOTH_RIGHT_MATRICES_PER_THREAD_LIMIT:1:2:Maximum value of right_matrices_per_thread to generate code for"

        "SHUFFLE_N_TO_M_MULTIMAT_BOTH_LOCAL_MEM_LEFT_MATRICES_PER_THREAD_LIMIT:1:2:Maximum value of left_matrices_per_thread to generate code for"
        "SHUFFLE_N_TO_M_MULTIMAT_BOTH_LOCAL_MEM_RIGHT_MATRICES_PER_THREAD_LIMIT:1:2:Maximum value of right_matrices_per_thread to generate code for"

        "SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_SHIFTS_PER_THREAD_PER_RIGHT_MATRIX_LIMIT:1:2:Maximum value of shifts_per_thread_per_right_matrix to generate code for"
        "SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_RIGHT_MATRICES_PER_THREAD_LIMIT:1:2:Maximum value of right_matrices_per_thread to generate code for"
        "SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_LEFT_MATRICES_PER_THREAD_LIMIT:1:2:Maximum value of left_matrices_per_thread to generate code for"
        "SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_LEFT_ROWS_PER_ITERATION_LIMIT:1:2:Maximum value of left_rows_per_iteration to generate code for"

        "SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_SHIFTS_PER_RIGHT_MATRIX_LIMIT:1:2:Maximum value of shifts_per_right_matrix to generate code for"
        "SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_RIGHT_MATRICES_PER_THREAD_LIMIT:1:2:Maximum value of right_matrices_per_thread to generate code for"
        "SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_LEFT_ROWS_PER_ITERATION_LIMIT:1:2:Maximum value of left_rows_per_iteration to generate code for"

        "WARP_PER_SHIFT_SHARED_MEM_RIGHT_MATRICES_PER_BLOCK_LIMIT:1:4:Maximum value of right_matrices_per_block to generate code for"
)

function(get_alg_option_part ALG_OPTION PART_START)
    string(SUBSTRING "${ALG_OPTION}" "${PART_START}" -1 SEARCHED_PART)
    string(FIND "${SEARCHED_PART}" ":" POS)
    string(SUBSTRING "${SEARCHED_PART}" 0 "${POS}" OPTION_PART)

    math(EXPR NEXT_PART_START "${PART_START} + ${POS} + 1")
    set(RETURN_VAL "${OPTION_PART}" PARENT_SCOPE)
    set(RETURN_POS "${NEXT_PART_START}" PARENT_SCOPE)
endfunction()

function(parse_alg_limit_option ALG_LIMIT_OPTION)
    get_alg_option_part("${ALG_LIMIT_OPTION}" 0)
    if (RETURN_POS EQUAL 0)
        message(WARNING "Malformed alg limit option (no option name): ${ALG_LIMIT_OPTION}")
        return()
    endif()
    set(RET_OPTION_NAME "${RETURN_VAL}" PARENT_SCOPE)
    set(NEXT_PART_START "${RETURN_POS}")

    get_alg_option_part("${ALG_LIMIT_OPTION}" "${RETURN_POS}")
    if (RETURN_POS EQUAL NEXT_PART_START)
        message(WARNING "Malformed alg limit option (no release value): ${ALG_LIMIT_OPTION}")
        unset(RET_OPTION_NAME PARENT_SCOPE)
        return()
    endif()
    set(RET_RELEASE_VALUE "${RETURN_VAL}" PARENT_SCOPE)
    set(NEXT_PART_START "${RETURN_POS}")

    get_alg_option_part("${ALG_LIMIT_OPTION}" "${RETURN_POS}")
    if (RETURN_POS EQUAL NEXT_PART_START)
        message(WARNING "Malformed alg limit option (no debug value): ${ALG_LIMIT_OPTION}")
        unset(RET_OPTION_NAME PARENT_SCOPE)
        unset(RET_RELEASE_VALUE PARENT_SCOPE)
        return()
    endif()
    set(RET_DEBUG_VALUE "${RETURN_VAL}" PARENT_SCOPE)
    set(NEXT_PART_START "${RETURN_POS}")

    get_alg_option_part("${ALG_LIMIT_OPTION}" "${RETURN_POS}")
    if (NOT RETURN_POS EQUAL NEXT_PART_START)
        message(WARNING "Malformed alg limit option (no description): ${ALG_LIMIT_OPTION}")
        unset(RET_OPTION_NAME PARENT_SCOPE)
        unset(RET_RELEASE_VALUE PARENT_SCOPE)
        unset(RET_DEBUG_VALUE PARENT_SCOPE)
        return()
    endif()
    set(RET_DESCRIPTION "${RETURN_VAL}" PARENT_SCOPE)
endfunction()



foreach(ALG_LIMIT_OPTION IN LISTS ALG_LIMIT_OPTIONS)
    parse_alg_limit_option("${ALG_LIMIT_OPTION}")
    if (NOT DEFINED RET_OPTION_NAME)
        continue()
    endif()

    # TODO: Maybe set debug version of the variable
    set("${RET_OPTION_NAME}" "${RET_RELEASE_VALUE}" CACHE STRING "${RET_DESCRIPTION}")
endforeach()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Requires cmake 3.18
set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 89-real)

# See https://stackoverflow.com/questions/27018340/cmake-does-not-properly-find-cuda-library
FIND_PACKAGE(CUDAToolkit 12.2 REQUIRED)
FIND_PACKAGE(Boost 1.71 COMPONENTS program_options REQUIRED)
# apt install nlohmann-json3-dev
FIND_PACKAGE(nlohmann_json 3.7.3 REQUIRED)


set(CPP_SOURCES
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        ${CMAKE_SOURCE_DIR}/src/row_distribution.cpp
        ${CMAKE_SOURCE_DIR}/src/argument_error.cpp
        ${CMAKE_SOURCE_DIR}/src/run_args.cpp
        ${CMAKE_SOURCE_DIR}/src/kernel_args.cpp
)
# Route everything through nvcc
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE CUDA)

set(CUDA_SOURCES
        ${CMAKE_SOURCE_DIR}/src/hadamard.cu
        ${CMAKE_SOURCE_DIR}/src/naive_original.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_multimat_right.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_n_to_m_multimat_both.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_n_to_m_multirow_both_multimat_both.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_multirow_right.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_multirow_both.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_multirow_right_multimat_right.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_one_to_many_multirow_both_multimat_right.cu
        ${CMAKE_SOURCE_DIR}/src/naive_warp_per_shift.cu
        ${CMAKE_SOURCE_DIR}/src/naive_warp_per_shift_shared_mem.cu
        ${CMAKE_SOURCE_DIR}/src/naive_block_per_shift.cu
        ${CMAKE_SOURCE_DIR}/src/scatter.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_n_to_m_multimat_both_local_mem.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_multirow_both_local_mem.cu

)

add_executable(cross ${CPP_SOURCES} ${CUDA_SOURCES})


target_include_directories(cross PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(cross PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(cross PUBLIC ${Boost_LIBRARIES})
target_link_libraries(cross PUBLIC CUDA::cudart)
target_link_libraries(cross PUBLIC CUDA::cufft)
target_link_libraries(cross PRIVATE nlohmann_json::nlohmann_json)

target_compile_options(cross PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
target_compile_options(cross PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>)

# target_compile_options(cross PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>)

target_compile_options(cross PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g -rdynamic>)
target_compile_options(cross PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -Xcompiler -rdynamic>)

foreach(ALG_LIMIT_OPTION IN LISTS ALG_LIMIT_OPTIONS)
    parse_alg_limit_option("${ALG_LIMIT_OPTION}")
    if (NOT DEFINED RET_OPTION_NAME)
        continue()
    endif()
    target_compile_definitions(cross PRIVATE "${RET_OPTION_NAME}=$<IF:$<CONFIG:Release>,${${RET_OPTION_NAME}},${RET_DEBUG_VALUE}>")
endforeach()
