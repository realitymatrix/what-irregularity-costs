cmake_minimum_required(VERSION 3.20)
project(osn_tsdf LANGUAGES CXX)

# CUDA arm (A3) is optional so the CPU arm still builds on a machine without a
# toolkit. Native sm_120 by default and deliberately no PTX fallback: an arm
# that JITs while another runs native SASS would invalidate the comparison, so
# a wrong arch must fail at load rather than silently JIT.
option(OSN_TSDF_CUDA "Build the CUDA arm (A3)" ON)

set(CMAKE_CXX_STANDARD 20)          # std::atomic_ref
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(OSN_TSDF_SOURCES src/volume.cpp src/marching_tets.cpp src/c_api.cpp src/triton_launch.cpp)
if(OSN_TSDF_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
    list(APPEND OSN_TSDF_SOURCES src/volume_cuda.cu src/marching_tets_cuda.cu src/c_api_cuda.cu)
endif()

add_library(osn_tsdf STATIC ${OSN_TSDF_SOURCES})
if(OSN_TSDF_CUDA)
    target_compile_definitions(osn_tsdf PUBLIC OSN_TSDF_HAVE_CUDA=1)
    set_target_properties(osn_tsdf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # Device debug info must stay OFF even in debug builds: -G disables device
    # optimisation outright and inflates register use enough to break launches.
    target_compile_options(osn_tsdf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
target_include_directories(osn_tsdf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(osn_tsdf PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
find_package(Threads REQUIRED)
target_link_libraries(osn_tsdf PUBLIC Threads::Threads)

# The Triton launcher uses the CUDA DRIVER API, which is a different library
# from the runtime the CUDA arm links.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(osn_tsdf PUBLIC CUDA::cuda_driver)

install(TARGETS osn_tsdf ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

enable_testing()
add_executable(test_analytic tests/test_analytic.cpp)
target_include_directories(test_analytic PRIVATE tests)
target_link_libraries(test_analytic PRIVATE osn_tsdf)
add_test(NAME analytic COMMAND test_analytic)

if(OSN_TSDF_CUDA)
    add_executable(test_cuda_analytic tests/test_cuda_analytic.cu)
    target_include_directories(test_cuda_analytic PRIVATE tests ${CMAKE_CURRENT_SOURCE_DIR}/../artifacts/triton)
    target_link_libraries(test_cuda_analytic PRIVATE osn_tsdf)

    # Arm A4 is a Rust staticlib built separately by cuda-oxide (it needs the
    # pinned nightly). Linked in when present so the cross-check can drive it;
    # the test compiles without it and simply skips that section.
    set(A4_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../crates/tsdf-rust-cuda/target/release/libtsdf_rust_cuda.a)
    if(EXISTS ${A4_LIB})
        target_compile_definitions(test_cuda_analytic PRIVATE OSN_TSDF_HAVE_A4=1)
        # cuda_core uses the CUDA DRIVER API (cuInit, cuCtxSetCurrent, ...),
        # which is a separate library from the runtime the C++ arms link.
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(test_cuda_analytic PRIVATE ${A4_LIB} CUDA::cuda_driver dl pthread)
        message(STATUS "Arm A4 (Rust/cuda-oxide) will be linked into the cross-check")
    else()
        message(STATUS "Arm A4 staticlib not found; cross-check will skip it")
    endif()

    add_test(NAME cuda_analytic COMMAND test_cuda_analytic)

    add_executable(bench_arms bench/bench_arms.cu)
    target_include_directories(bench_arms PRIVATE tests
        ${CMAKE_CURRENT_SOURCE_DIR}/../artifacts/triton)
    target_link_libraries(bench_arms PRIVATE osn_tsdf)
    if(EXISTS ${A4_LIB})
        target_compile_definitions(bench_arms PRIVATE OSN_TSDF_HAVE_A4=1)
        target_link_libraries(bench_arms PRIVATE ${A4_LIB} CUDA::cuda_driver dl pthread)

        add_executable(probe_a4_tail bench/probe_a4_tail.cu)
        target_include_directories(probe_a4_tail PRIVATE tests)
        target_link_libraries(probe_a4_tail PRIVATE osn_tsdf ${A4_LIB} CUDA::cuda_driver dl pthread)

        # Prices the three components of A4's allocate path one at a time.
        add_executable(probe_a4_bisect bench/probe_a4_bisect.cu)
        target_include_directories(probe_a4_bisect PRIVATE tests)
        target_link_libraries(probe_a4_bisect PRIVATE osn_tsdf ${A4_LIB} CUDA::cuda_driver dl pthread)

        # Splits the extract measurement into kernel, allocation and PCIe
        # readback. Needed to decide whether extraction can join the language
        # comparison at all; see docs/ARM-SCOPE.md.
        add_executable(probe_extract bench/probe_extract.cu)
        target_include_directories(probe_extract PRIVATE tests)
        target_link_libraries(probe_extract PRIVATE osn_tsdf CUDA::cuda_driver dl pthread)
    endif()
endif()
