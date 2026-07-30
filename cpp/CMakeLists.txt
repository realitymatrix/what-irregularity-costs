cmake_minimum_required(VERSION 3.20)
project(osn_tsdf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)          # std::atomic_ref
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(osn_tsdf STATIC
    src/volume.cpp
    src/marching_tets.cpp
    src/c_api.cpp
)
target_include_directories(osn_tsdf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(osn_tsdf PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
find_package(Threads REQUIRED)
target_link_libraries(osn_tsdf PUBLIC Threads::Threads)

install(TARGETS osn_tsdf ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

enable_testing()
add_executable(test_analytic tests/test_analytic.cpp)
target_link_libraries(test_analytic PRIVATE osn_tsdf)
add_test(NAME analytic COMMAND test_analytic)
