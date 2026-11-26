cmake_minimum_required(VERSION 3.16)
project(l0_transport_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH must point at the CANN installation")
endif()
set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}")
message(STATUS "CANN root: ${CANN_ROOT}")

add_library(l0_transport_bridge SHARED l0_transport_bridge.cpp)
target_include_directories(l0_transport_bridge PRIVATE
    "${CANN_ROOT}/include"
    "${CANN_ROOT}/include/hccl"
    "${CANN_ROOT}/include/hcomm")

find_library(HCCL_LIBRARY NAMES hccl PATHS "${CANN_ROOT}/lib64" "${CANN_ROOT}/lib" NO_DEFAULT_PATH)
find_library(HCOMM_LIBRARY NAMES hcomm PATHS "${CANN_ROOT}/lib64" "${CANN_ROOT}/lib" NO_DEFAULT_PATH)
find_library(ACL_RT_LIBRARY NAMES acl_rt PATHS "${CANN_ROOT}/lib64" "${CANN_ROOT}/lib" "${CANN_ROOT}/runtime/lib64" NO_DEFAULT_PATH)

foreach(name HCCL_LIBRARY HCOMM_LIBRARY ACL_RT_LIBRARY)
    if(NOT ${name})
        message(WARNING "${name} was not found under ${CANN_ROOT}; the bridge will build but "
                        "report the missing dependency at run time")
    endif()
endforeach()

target_link_libraries(l0_transport_bridge PRIVATE
    ${HCCL_LIBRARY} ${HCOMM_LIBRARY} ${ACL_RT_LIBRARY})
target_compile_options(l0_transport_bridge PRIVATE -Wall -Wextra -Wno-unused-parameter)

install(TARGETS l0_transport_bridge LIBRARY DESTINATION lib)
