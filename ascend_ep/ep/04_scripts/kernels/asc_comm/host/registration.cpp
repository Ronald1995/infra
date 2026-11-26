// asc_comm/host/registration.cpp -- symmetric-buffer registration for ASC-COMM.
//
// A remote write needs a destination address that the peer has published.  The
// benchmark registers two buffers per rank (a source and a destination) through the
// HCOMM bridge; this file declares the registration record the kernels expect and
// the checks that must pass before an address may be used.
//
// It intentionally contains no vendor API call: the registration call itself is
// made by the HCOMM bridge in the benchmark, which is the component that owns the
// channel.  What the kernel needs is only the resulting (address, length, tag)
// triple, and that triple is validated here.

#include "common/abi.h"

#include <cstddef>
#include <string>

namespace asc_comm {

//: One registered symmetric buffer, as published to the peer.
struct BufferRegistration {
  const char* tag;    // "src" or "dst"
  uint64_t address;   // device virtual address
  uint64_t bytes;     // registered length
  uint32_t peer;      // rank the buffer belongs to
};

//: Reject a registration the data plane cannot safely use.
inline int validate_registration(const BufferRegistration& buffer) {
  if (buffer.tag == nullptr) return ASC_COMM_ERR_ARGUMENT;
  if (buffer.address == 0) return ASC_COMM_ERR_ARGUMENT;
  if (buffer.bytes == 0) return ASC_COMM_ERR_ARGUMENT;
  const std::string tag(buffer.tag);
  if (tag != "src" && tag != "dst") return ASC_COMM_ERR_ARGUMENT;
  return ASC_COMM_OK;
}

//: A transfer is only admissible when both ends are inside registered buffers.
inline int validate_bounds(const BufferRegistration& source,
                           const BufferRegistration& destination,
                           const asc_comm_transfer& transfer) {
  if (validate_registration(source) != ASC_COMM_OK) return ASC_COMM_ERR_ARGUMENT;
  if (validate_registration(destination) != ASC_COMM_OK) return ASC_COMM_ERR_ARGUMENT;
  if (transfer.src < source.address ||
      transfer.src + transfer.bytes > source.address + source.bytes) {
    return ASC_COMM_ERR_ARGUMENT;
  }
  if (transfer.dst < destination.address ||
      transfer.dst + transfer.bytes > destination.address + destination.bytes) {
    return ASC_COMM_ERR_ARGUMENT;
  }
  return ASC_COMM_OK;
}

}  // namespace asc_comm
