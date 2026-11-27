# ASC-COMM kernels: AIV and CCU data planes for the L0 transport baseline.
#
# STATUS (read this before changing anything):
#   * host/ (launcher.cpp, registration.cpp) is complete C++ and needs no CANN API.
#   * aiv/ and ccu/ contain the device-program specification and the build wiring,
#     but the device code is NOT written, NOT compiled and NOT verified.  The
#     project therefore reports BLOCKED, and every AIV/CCU transport case reports
#     dependency_missing / not_implemented instead of a number.
#   * No vendor API signature is invented here.  The device sources must be written
#     against the headers actually installed under ${ASCEND_HOME_PATH} on the target
#     machine; the checklist is in aiv/README.md and ccu/README.md.
#
# Build output that the benchmark's capability probe consumes:
#   build/manifest.json          {"engines": {"AIV": {...}, "CCU": {...}}}
#   build/libasc_comm_kernels.so
#
# The probe refuses to use a library that does not export the ABI symbols declared
# in common/abi.h, so a partial build cannot be mistaken for a working data plane.

cmake_minimum_required(VERSION 3.16)
project(asc_comm_kernels CXX)

set(ASC_COMM_ABI_VERSION 1)

# ---------------------------------------------------------------- toolchain --
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR
    "ASCEND_HOME_PATH is not set.  The ASC-COMM kernels need the CANN toolkit "
    "(toolkit headers, the Ascend C compiler and the CCU DSL frontend).  Set it to "
    "the installed toolkit, e.g. /usr/local/Ascend/ascend-toolkit/latest, and "
    "re-run.  Nothing is built otherwise: an unbuilt kernel must stay a reported "
    "blocker, not a silent fallback to the host NBI path.")
endif()
set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
message(STATUS "ASC-COMM: CANN toolkit at ${ASCEND_HOME}")

option(ASC_COMM_BUILD_AIV "Build the AIV write_nbi data plane" ON)
option(ASC_COMM_BUILD_CCU "Build the CCU data plane" ON)
option(ASC_COMM_STRICT "Fail the configure step when a device source is missing" OFF)

set(ASC_COMM_SOC_NAME "unknown" CACHE STRING "SoC the kernels were built for")
string(TIMESTAMP ASC_COMM_BUILD_ID "%Y%m%dT%H%M%SZ" UTC)

set(ASC_COMM_SOURCES host/launcher.cpp host/registration.cpp)
set(ASC_COMM_DEFINITIONS ASC_COMM_ABI_VERSION=${ASC_COMM_ABI_VERSION}
                         ASC_COMM_SOC_NAME="${ASC_COMM_SOC_NAME}"
                         ASC_COMM_BUILD_ID="${ASC_COMM_BUILD_ID}")

#: Whether the AIV *kernel* really compiled, as opposed to merely being requested:
#: the manifest must not claim an engine whose device sources are absent.
set(ASC_COMM_AIV_KERNEL OFF)
if(ASC_COMM_BUILD_AIV)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/aiv/write_nbi/kernel.cpp")
    list(APPEND ASC_COMM_SOURCES aiv/write_nbi/kernel.cpp)
    list(APPEND ASC_COMM_DEFINITIONS ASC_COMM_BUILD_AIV)
    set(ASC_COMM_AIV_KERNEL ON)
  elseif(ASC_COMM_STRICT)
    message(FATAL_ERROR "AIV requested but aiv/write_nbi/kernel.cpp does not exist")
  else()
    message(WARNING
      "AIV requested but aiv/write_nbi/kernel.cpp does not exist yet: the AIV data "
      "plane will report ASC_COMM_ERR_UNSUPPORTED at run time.")
  endif()
endif()

set(ASC_COMM_CCU_KERNEL OFF)
if(ASC_COMM_BUILD_CCU)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ccu/dsl/kernel.cpp")
    list(APPEND ASC_COMM_SOURCES ccu/dsl/kernel.cpp)
    list(APPEND ASC_COMM_DEFINITIONS ASC_COMM_BUILD_CCU)
    set(ASC_COMM_CCU_KERNEL ON)
  elseif(ASC_COMM_STRICT)
    message(FATAL_ERROR "CCU requested but ccu/dsl/kernel.cpp does not exist")
  else()
    message(WARNING
      "CCU requested but ccu/dsl/kernel.cpp does not exist yet: the CCU data plane "
      "will report ASC_COMM_ERR_UNSUPPORTED at run time.")
  endif()
endif()

add_library(asc_comm_kernels SHARED ${ASC_COMM_SOURCES})
target_include_directories(asc_comm_kernels PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}" "${ASCEND_HOME}/include")
target_compile_definitions(asc_comm_kernels PRIVATE ${ASC_COMM_DEFINITIONS})
set_target_properties(asc_comm_kernels PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# The manifest is what the benchmark probe reads.  Writing it from the build keeps
# "the library exists" and "the build succeeded" from drifting apart.
add_custom_command(TARGET asc_comm_kernels POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -DLIBRARY=$<TARGET_FILE:asc_comm_kernels>
          -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/manifest.json
          -DAIV=$<BOOL:${ASC_COMM_BUILD_AIV}>
          -DCCU=$<BOOL:${ASC_COMM_BUILD_CCU}>
          -DAIV_KERNEL=$<BOOL:${ASC_COMM_AIV_KERNEL}>
          -DCCU_KERNEL=$<BOOL:${ASC_COMM_CCU_KERNEL}>
          -DSOC=${ASC_COMM_SOC_NAME}
          -DBUILD_ID=${ASC_COMM_BUILD_ID}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_manifest.cmake"
  COMMENT "Writing the ASC-COMM build manifest")
