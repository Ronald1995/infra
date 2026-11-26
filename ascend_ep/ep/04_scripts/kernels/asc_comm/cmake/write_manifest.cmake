# Writes build/manifest.json, the file the benchmark's capability probe reads.
#
# Deliberately a CMake script rather than a shell heredoc: it runs on every platform
# the build runs on, and it is invoked only after the library has linked, so a
# manifest can never claim a kernel that failed to build.

if(NOT DEFINED LIBRARY OR NOT EXISTS "${LIBRARY}")
  message(FATAL_ERROR "write_manifest: the kernel library does not exist: '${LIBRARY}'")
endif()

function(json_engine name enabled)
  if(enabled)
    set(built "true")
  else()
    set(built "false")
  endif()
  if(NOT enabled)
    set(reason "the ${name} data plane was not enabled in this build")
  else()
    set(reason "null")
  endif()
  if(reason STREQUAL "null")
    set(entry "\"${name}\": {\"built\": ${built}, \"library\": \"${LIBRARY}\", \"soc\": \"${SOC}\", \"build_id\": \"${BUILD_ID}\"}")
  else()
    set(entry "\"${name}\": {\"built\": ${built}, \"library\": null, \"reason\": \"${reason}\"}")
  endif()
  set(ASC_COMM_MANIFEST_ENTRIES "${ASC_COMM_MANIFEST_ENTRIES}${entry}," PARENT_SCOPE)
endfunction()

json_engine(AIV "${AIV}")
json_engine(CCU "${CCU}")
string(REGEX REPLACE ",$" "" ENTRIES "${ASC_COMM_MANIFEST_ENTRIES}")

file(WRITE "${OUTPUT}" "{\n  \"schema_version\": 1,\n  \"soc\": \"${SOC}\",\n  \"build_id\": \"${BUILD_ID}\",\n  \"library\": \"${LIBRARY}\",\n  \"engines\": {${ENTRIES}}\n}\n")
message(STATUS "write_manifest: ${OUTPUT}")
