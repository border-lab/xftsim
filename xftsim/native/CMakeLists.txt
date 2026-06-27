cmake_minimum_required(VERSION 3.14)
project(grg_recomb_native CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Point at the grgl source tree. Tries common relative layouts in order:
#   ../../grgl   -- repo layout (benchmark/native/ inside a tree that has grgl/ at the root)
#   ../grgl      -- flatter layout (native/ sits as a sibling of grgl/)
# Override with -DGRGL_ROOT=/abs/path or the GRGL_ROOT env var if neither matches.
if(NOT DEFINED GRGL_ROOT)
    foreach(_candidate "../../grgl" "../grgl")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_candidate}/CMakeLists.txt")
            set(GRGL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/${_candidate}")
            break()
        endif()
    endforeach()
endif()
if(NOT DEFINED GRGL_ROOT)
    message(FATAL_ERROR
        "Could not auto-detect a grgl source tree. Tried ../../grgl and ../grgl "
        "relative to ${CMAKE_CURRENT_SOURCE_DIR}. Pass -DGRGL_ROOT=/abs/path or set "
        "the GRGL_ROOT environment variable.")
endif()
get_filename_component(GRGL_ROOT "${GRGL_ROOT}" ABSOLUTE)
if(NOT EXISTS "${GRGL_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "GRGL_ROOT does not point to a grgl checkout: ${GRGL_ROOT}")
endif()
message(STATUS "Using grgl from: ${GRGL_ROOT}")

# Pull in grgl as a CMake subproject. EXCLUDE_FROM_ALL means we only build the
# pieces we actually link (libgrgl, libvbyte) -- skipping grgl's executables,
# tests, and its own _grgl Python module (we provide our own pybind11 build).
set(PYTHON_SUPPORT OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_CHECKERS OFF CACHE BOOL "" FORCE)
add_subdirectory(${GRGL_ROOT} grgl_build EXCLUDE_FROM_ALL)

# Use grgl's bundled pybind11 (2.14.0.dev1) so our extension shares the same
# pybind11 internals tag as pygrgl's _grgl module. Cross-module type sharing
# requires matching pybind11 versions; the smoke test confirmed this works.
add_subdirectory(${GRGL_ROOT}/third-party/pybind11 pybind11_build EXCLUDE_FROM_ALL)

pybind11_add_module(_grg_recomb_native
    src/bindings.cpp
    src/recombination.cpp
)
target_link_libraries(_grg_recomb_native PRIVATE grgl vbyte)
target_include_directories(_grg_recomb_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GRGL_ROOT}/include
    ${GRGL_ROOT}/third-party/libvbyte
)
target_compile_definitions(_grg_recomb_native PRIVATE
    COMPACT_NODE_IDS
    VCF_GZ_SUPPORT
)
target_compile_options(_grg_recomb_native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
)
