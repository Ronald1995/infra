/*
 * L0 native transport bridge.
 *
 * One C ABI shared library that wraps the HCOMM/HCCL control plane and the
 * host-side HCOMM data plane so the Python benchmark can drive them without
 * linking against CANN itself.  Everything here is optional: when the CANN
 * headers are absent the library still builds and reports every capability as
 * unavailable through `l0_bridge_describe`, which is how the Python side turns
 * a missing dependency into `dependency_missing` instead of a fabricated result.
 *
 * The bridge never decides which protocol was used.  It reports what it asked
 * for and what the runtime answered:
 *
 *   - `HcclRankGraphGetLinks` -> `CommLink.linkAttr.linkProtocol` (rank graph)
 *   - the `channelProtocol` handed to `HcclChannelAcquire`
 *   - the `CommEngine` handed to `HcclChannelAcquire`
 *   - the submission mode the data path actually accepted
 *
 * Control plane: HCCL channel acquisition (`HcclChannelAcquire`).
 * Data plane:    HCOMM primitives (`HcommWrite*` / `HcommRead*` / fence / drain).
 *
 * Status: this file is compiled and exercised only on an Ascend host.  See
 * `baselines/l0/transport/README.md` for the supported CANN versions and the
 * exact build command.  Nothing in the repository claims it has been run.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#if __has_include(<hccl/hccl.h>)
#include <hccl/hccl.h>
#include <hccl/hccl_channel.h>
#include <hccl/hccl_rank_graph.h>
#define L0_BRIDGE_HAS_HCCL 1
#else
#define L0_BRIDGE_HAS_HCCL 0
#endif

#if __has_include(<hcomm/hcomm_res.h>)
#include <hcomm/hcomm_res.h>
#include <hcomm/hcomm_primitives.h>
#define L0_BRIDGE_HAS_HCOMM 1
#else
#define L0_BRIDGE_HAS_HCOMM 0
#endif

#if __has_include(<acl/acl.h>)
#include <acl/acl.h>
#define L0_BRIDGE_HAS_ACL 1
#else
#define L0_BRIDGE_HAS_ACL 0
#endif

#define L0_API extern "C" __attribute__((visibility("default")))
#define L0_BRIDGE_SCHEMA 1
#define L0_MAX_TEXT 8192
#define L0_MAX_PEERS 256
#define L0_MAX_MEMS 8

// HCOMM success is 0 in every released header set; the named constant moved
// between releases, so compare against the documented value.
constexpr int32_t kHcommSuccess = 0;

namespace {

std::string g_lastError;
std::mutex g_mutex;

void SetError(const std::string& text)
{
    std::lock_guard<std::mutex> guard(g_mutex);
    g_lastError = text;
}

std::string Escape(const std::string& value)
{
    std::string out;
    out.reserve(value.size() + 8);
    for (char character : value) {
        switch (character) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20) {
                    char buffer[8];
                    snprintf(buffer, sizeof(buffer), "\\u%04x", character & 0xFF);
                    out += buffer;
                } else {
                    out += character;
                }
        }
    }
    return out;
}

int CopyOut(const std::string& text, char* buffer, uint32_t capacity)
{
    if (text.size() + 1 > capacity) {
        SetError("output buffer too small");
        return -2;
    }
    memcpy(buffer, text.data(), text.size());
    buffer[text.size()] = '\0';
    return static_cast<int>(text.size());
}

std::string ProtocolName(int protocol)
{
    switch (protocol) {
        case 0: return "HCCS";
        case 1: return "ROCE";
        case 2: return "PCIE";
        case 3: return "SIO";
        case 4: return "UB_CTP";
        case 5: return "UBC_TP";
        case 6: return "UB_MEM";
        case 7: return "UBOE";
        case 8: return "HCCS_ONLY";
        case 9: return "UB_RTP";
        default: return "RESERVED";
    }
}

std::string EngineName(int engine)
{
    switch (engine) {
        case 0: return "CPU";
        case 1: return "CPU_TS";
        case 2: return "AICPU";
        case 3: return "AICPU_TS";
        case 4: return "AIV";
        case 5: return "CCU";
        default: return "RESERVED";
    }
}

int ProtocolValue(const std::string& name)
{
    if (name == "HCCS") return 0;
    if (name == "ROCE") return 1;
    if (name == "PCIE") return 2;
    if (name == "SIO") return 3;
    if (name == "UB_CTP" || name == "UBC_CTP") return 4;
    if (name == "UBC_TP") return 5;
    if (name == "UB_MEM") return 6;
    if (name == "UBOE") return 7;
    if (name == "HCCS_ONLY") return 8;
    if (name == "UB_RTP" || name == "UBG") return 9;
    return -1;
}

int EngineValue(const std::string& name)
{
    if (name == "CPU") return 0;
    if (name == "CPU_TS") return 1;
    if (name == "AICPU") return 2;
    if (name == "AICPU_TS") return 3;
    if (name == "AIV") return 4;
    if (name == "CCU") return 5;
    return -1;
}

struct Session {
    bool aclReady = false;
    bool commReady = false;
    int device = 0;
    int rank = 0;
    int world = 1;
    void* stream = nullptr;
#if L0_BRIDGE_HAS_HCCL
    HcclComm comm = nullptr;
#endif
#if L0_BRIDGE_HAS_HCOMM
    EndpointHandle endpoint = nullptr;
    ThreadHandle thread = 0;
#endif
    struct Peer {
        int rank = -1;
        uint64_t channel = 0;
        std::string protocolFromGraph;
        std::string topoType;
        uint32_t hop = 0;
        std::vector<uint64_t> remoteAddrs;
        std::vector<std::string> remoteTags;
    };
    std::vector<Peer> peers;
    std::vector<CommMem> localMems;
    std::vector<std::string> localTags;
#if L0_BRIDGE_HAS_HCCL
    std::vector<HcclMemHandle> localHandles;
#endif
    std::string submission = "none";
};

Session* g_session = nullptr;

}  // namespace

L0_API int l0_bridge_describe(char* buffer, uint32_t capacity)
{
    std::string json = "{\"schema_version\":" + std::to_string(L0_BRIDGE_SCHEMA);
    json += ",\"bridge\":\"l0_transport_bridge\"";
#if L0_BRIDGE_HAS_HCCL
    json += ",\"hccl\":true";
#else
    json += ",\"hccl\":false";
#endif
#if L0_BRIDGE_HAS_HCOMM
    json += ",\"hcomm\":true";
#else
    json += ",\"hcomm\":false";
#endif
    json += ",\"engines\":[\"CPU\",\"CPU_TS\",\"AICPU\",\"AICPU_TS\",\"AIV\",\"CCU\"]";
    json += ",\"protocols\":[\"HCCS\",\"ROCE\",\"PCIE\",\"SIO\",\"UB_CTP\",\"UBC_TP\","
            "\"UB_MEM\",\"UBOE\",\"HCCS_ONLY\",\"UB_RTP\"]";
    json += ",\"features\":[\"rank_graph\",\"channel_acquire\",\"remote_mems\","
            "\"host_write\",\"endpoint_probe\"]";
    json += ",\"root_info_bytes\":";
#if L0_BRIDGE_HAS_HCCL
    {
        HcclRootInfo root{};
        json += std::to_string(sizeof(root));
    }
#else
    json += "0";
#endif
    json += ",\"channels\":[";
    bool first = true;
    // Protocols each channel class is selected for, taken from the HCOMM
    // dispatch in Channel::CreateChannel (engine + remoteEndpoint.protocol):
    // AIV + UB_CTP/UBC_TP/UB_RTP -> AivUrmaChannel, AIV + anything else ->
    // AivUbMemChannel, AICPU_TS + UB_RTP -> AicpuTsUbRtpChannel, CCU requires
    // UB_CTP/UBC_TP -> CcuUrmaChannel.
    const struct { const char* name; const char* engine; const char* protocols; } channels[] = {
        {"AivUbMemChannel", "AIV", "UB_MEM"},
        {"AivUrmaChannel", "AIV", "UB_CTP,UBC_TP,UB_RTP"},
        {"AivRoceHostLocChannel", "AIV", "ROCE"},
        {"AicpuTsUrmaChannel", "AICPU_TS", "UB_CTP,UBC_TP"},
        {"AicpuTsHccsChannel", "AICPU_TS", "HCCS"},
        {"AicpuTsUbRtpChannel", "AICPU_TS", "UB_RTP"},
        {"AicpuTsUboeChannel", "AICPU_TS", "UBOE"},
        {"AicpuTsRoceHostLocChannel", "AICPU_TS", "ROCE"},
        {"AicpuTsP2pChannel", "AICPU_TS", "PCIE"},
        {"CcuUrmaChannel", "CCU", "UB_CTP,UBC_TP"},
        {"HostCpuUrmaChannel", "CPU", "UB_CTP,UBC_TP"},
        {"HostCpuRoceChannel", "CPU", "ROCE"},
    };
    for (const auto& channel : channels) {
#if L0_BRIDGE_HAS_HCCL && L0_BRIDGE_HAS_HCOMM
        const bool compiled = true;
#else
        const bool compiled = false;
#endif
        if (!first) {
            json += ",";
        }
        first = false;
        json += "{\"channel\":\"" + std::string(channel.name) + "\"";
        json += ",\"engine\":\"" + std::string(channel.engine) + "\"";
        json += ",\"mechanisms\":[";
        const std::string protocols(channel.protocols);
        size_t start = 0;
        bool firstProtocol = true;
        while (start <= protocols.size()) {
            const size_t comma = protocols.find(',', start);
            const std::string name = protocols.substr(
                start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!name.empty()) {
                if (!firstProtocol) {
                    json += ",";
                }
                firstProtocol = false;
                json += "\"" + name + "\"";
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        json += "]";
        json += ",\"compiled\":";
        json += compiled ? "true" : "false";
        json += "}";
    }
    json += "]";
    json += ",\"status\":\"";
    json += (L0_BRIDGE_HAS_HCCL && L0_BRIDGE_HAS_HCOMM) ? "compiled" : "missing_headers";
    json += "\"}";
    return CopyOut(json, buffer, capacity);
}

L0_API int l0_bridge_last_error(char* buffer, uint32_t capacity)
{
    std::lock_guard<std::mutex> guard(g_mutex);
    return CopyOut(g_lastError, buffer, capacity);
}

/*
 * Create and immediately destroy one endpoint for a protocol/location pair.
 * This is the only capability probe that needs no peer, and it is authoritative:
 * the runtime either accepts the protocol for that endpoint location or not.
 */
L0_API int l0_bridge_probe_endpoint(const char* protocol, int locType, int deviceId,
                                    char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCOMM
    if (protocol == nullptr) {
        return CopyOut("{\"ok\":false,\"reason\":\"protocol is null\"}", buffer, capacity);
    }
    const int value = ProtocolValue(protocol);
    if (value < 0) {
        return CopyOut("{\"ok\":false,\"reason\":\"unknown protocol\"}", buffer, capacity);
    }
    EndpointDesc desc{};
    EndpointDescInit(&desc, 1);
    desc.protocol = static_cast<CommProtocol>(value);
    desc.loc.locType = (locType == 1) ? ENDPOINT_LOC_TYPE_HOST : ENDPOINT_LOC_TYPE_DEVICE;
    if (desc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
        desc.loc.device.devPhyId = static_cast<uint32_t>(deviceId);
        desc.loc.device.superDevId = static_cast<uint32_t>(deviceId);
    } else {
        desc.loc.host.id = static_cast<uint32_t>(deviceId);
    }
    EndpointHandle handle = nullptr;
    const HcommResult result = HcommEndpointCreate(&desc, &handle);
    if (result == kHcommSuccess && handle != nullptr) {
        (void)HcommEndpointDestroy(handle);
    }
    std::string json = "{\"ok\":";
    json += (result == kHcommSuccess) ? "true" : "false";
    json += ",\"protocol\":\"" + std::string(protocol) + "\"";
    json += ",\"loc\":\"";
    json += (locType == 1) ? "HOST" : "DEVICE";
    json += "\",\"result\":" + std::to_string(result) + "}";
    return CopyOut(json, buffer, capacity);
#else
    (void)protocol;
    (void)locType;
    (void)deviceId;
    return CopyOut("{\"ok\":false,\"reason\":\"hcomm headers were not available at build time\"}",
                   buffer, capacity);
#endif
}

/*
 * Session lifecycle.  `options` is a small JSON object:
 *   {"rank":0,"world":8,"device":0,"deviceId":3,"rootInfo":"<base64 omitted>"}
 * Root-info bytes are passed separately to keep base64 out of the C++ side.
 */
L0_API int64_t l0_session_open(int world, int rank, int device, char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCCL && L0_BRIDGE_HAS_HCOMM
    if (g_session != nullptr) {
        SetError("a session is already open");
        return -1;
    }
    auto* session = new Session();
    session->world = world;
    session->rank = rank;
    session->device = device;
    const int aclResult = aclrtSetDevice(device);
    if (aclResult != 0) {
        SetError("aclrtSetDevice failed: " + std::to_string(aclResult));
        delete session;
        return -1;
    }
    session->aclReady = true;
    g_session = session;
    std::string json = "{\"ok\":true,\"rank\":" + std::to_string(rank) +
                       ",\"world\":" + std::to_string(world) + "}";
    CopyOut(json, buffer, capacity);
    return 1;
#else
    (void)world;
    (void)rank;
    (void)device;
    (void)buffer;
    (void)capacity;
    SetError("bridge was built without HCCL/HCOMM headers");
    return -1;
#endif
}

L0_API int l0_session_comm_init(int64_t handle, const void* rootInfo, uint32_t rootInfoBytes,
                                char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCCL
    Session* session = (handle == 1) ? g_session : nullptr;
    if (session == nullptr) {
        return CopyOut("{\"ok\":false,\"reason\":\"session is not open\"}", buffer, capacity);
    }
    if (session->commReady) {
        return CopyOut("{\"ok\":true,\"note\":\"already initialized\"}", buffer, capacity);
    }
    HcclComm comm = nullptr;
    const HcclResult result = HcclCommInitRootInfo(
        static_cast<uint32_t>(session->world),
        reinterpret_cast<const HcclRootInfo*>(rootInfo), static_cast<uint32_t>(session->rank), &comm);
    if (result != HCCL_SUCCESS || comm == nullptr) {
        SetError("HcclCommInitRootInfo failed: " + std::to_string(result) +
                 " rootInfoBytes=" + std::to_string(rootInfoBytes));
        std::string json = "{\"ok\":false,\"result\":" + std::to_string(result) + "}";
        return CopyOut(json, buffer, capacity);
    }
    session->comm = comm;
    session->commReady = true;
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    (void)HcclGetRankId(comm, &rankId);
    (void)HcclGetRankSize(comm, &rankSize);
    std::string json = "{\"ok\":true,\"rank\":" + std::to_string(rankId) +
                       ",\"rank_size\":" + std::to_string(rankSize) + "}";
    return CopyOut(json, buffer, capacity);
#else
    (void)handle;
    (void)rootInfo;
    (void)rootInfoBytes;
    return CopyOut("{\"ok\":false,\"reason\":\"hccl headers unavailable\"}", buffer, capacity);
#endif
}

/*
 * Rank graph: the runtime's own description of how ranks are connected.
 * `l0_session_rank_graph` returns a JSON array with one entry per peer; the
 * `link_protocol` values are copied from `CommLink.linkAttr.linkProtocol`.
 */
L0_API int l0_session_rank_graph(int64_t handle, char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCCL
    Session* session = (handle == 1) ? g_session : nullptr;
    if (session == nullptr || !session->commReady) {
        return CopyOut("{\"ok\":false,\"reason\":\"comm is not initialized\"}", buffer, capacity);
    }
    uint32_t* layers = nullptr;
    uint32_t layerCount = 0;
    const HcclResult layerResult = HcclRankGraphGetLayers(session->comm, &layers, &layerCount);
    std::string json = "{\"ok\":";
    json += (layerResult == HCCL_SUCCESS) ? "true" : "false";
    json += ",\"result\":" + std::to_string(layerResult);
    json += ",\"layers\":[";
    for (uint32_t index = 0; index < layerCount; ++index) {
        if (index != 0) {
            json += ",";
        }
        CommTopo topo = {};
        const HcclResult topoResult = HcclRankGraphGetTopoTypeByLayer(session->comm, layers[index], &topo);
        json += "{\"layer\":" + std::to_string(layers[index]);
        json += ",\"topo\":";
        json += (topoResult == HCCL_SUCCESS) ? std::to_string(static_cast<int>(topo)) : "-1";
        json += ",\"links\":[";
        bool firstLink = true;
        for (int peer = 0; peer < session->world; ++peer) {
            if (peer == session->rank) {
                continue;
            }
            CommLink* links = nullptr;
            uint32_t linkCount = 0;
            const HcclResult linkResult = HcclRankGraphGetLinks(
                session->comm, layers[index], static_cast<uint32_t>(session->rank),
                static_cast<uint32_t>(peer), &links, &linkCount);
            if (linkResult != HCCL_SUCCESS || links == nullptr) {
                continue;
            }
            for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
                if (!firstLink) {
                    json += ",";
                }
                firstLink = false;
                json += "{\"peer\":" + std::to_string(peer);
                json += ",\"link_protocol\":\"" +
                        ProtocolName(static_cast<int>(links[linkIndex].linkAttr.linkProtocol)) + "\"";
                json += ",\"hop\":" + std::to_string(links[linkIndex].linkAttr.hop);
                json += "}";
            }
        }
        json += "]}";
    }
    json += "]}";
    return CopyOut(json, buffer, capacity);
#else
    (void)handle;
    return CopyOut("{\"ok\":false,\"reason\":\"hccl headers unavailable\"}", buffer, capacity);
#endif
}

/*
 * Register one local device buffer as symmetric memory for this communicator.
 * Returns a tag index used by later transfer calls.
 */
L0_API int l0_session_register(int64_t handle, void* address, uint64_t bytes, const char* tag,
                               char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCCL
    Session* session = (handle == 1) ? g_session : nullptr;
    if (session == nullptr || !session->commReady) {
        return CopyOut("{\"ok\":false,\"reason\":\"comm is not initialized\"}", buffer, capacity);
    }
    if (session->localMems.size() >= L0_MAX_MEMS) {
        return CopyOut("{\"ok\":false,\"reason\":\"too many registered buffers\"}", buffer, capacity);
    }
    CommMem mem{};
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = address;
    mem.size = bytes;
    HcclMemHandle memHandle = nullptr;
    const HcclResult result = HcclCommMemReg(session->comm, tag, &mem, &memHandle);
    if (result != HCCL_SUCCESS) {
        std::string json = "{\"ok\":false,\"result\":" + std::to_string(result) + "}";
        return CopyOut(json, buffer, capacity);
    }
    session->localMems.push_back(mem);
    session->localTags.emplace_back(tag == nullptr ? "" : tag);
    session->localHandles.push_back(memHandle);
    const size_t index = session->localMems.size() - 1;
    std::string json = "{\"ok\":true,\"mem_index\":" + std::to_string(index) +
                       ",\"handle\":" + std::to_string(reinterpret_cast<uintptr_t>(memHandle)) + "}";
    return CopyOut(json, buffer, capacity);
#else
    (void)handle;
    (void)address;
    (void)bytes;
    (void)tag;
    return CopyOut("{\"ok\":false,\"reason\":\"hccl headers unavailable\"}", buffer, capacity);
#endif
}

/*
 * Acquire one channel per peer for the requested engine and protocol, using the
 * endpoints the rank graph reports for each link.  The response names, per peer,
 * the protocol the rank graph advertised and the protocol that was requested.
 */
L0_API int l0_session_channels(int64_t handle, const char* engineName, const char* protocolName,
                               const int* peers, int peerCount, char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCCL
    Session* session = (handle == 1) ? g_session : nullptr;
    if (session == nullptr || !session->commReady) {
        return CopyOut("{\"ok\":false,\"reason\":\"comm is not initialized\"}", buffer, capacity);
    }
    const int engineValue = EngineValue(engineName == nullptr ? "" : engineName);
    const int protocolValue = ProtocolValue(protocolName == nullptr ? "" : protocolName);
    if (engineValue < 0 || protocolValue < 0) {
        return CopyOut("{\"ok\":false,\"reason\":\"unknown engine or protocol\"}", buffer, capacity);
    }
    if (peerCount <= 0 || peerCount > L0_MAX_PEERS) {
        return CopyOut("{\"ok\":false,\"reason\":\"invalid peer count\"}", buffer, capacity);
    }
    session->peers.clear();
    session->peers.resize(static_cast<size_t>(peerCount));
    std::vector<HcclChannelDesc> descs(static_cast<size_t>(peerCount));
    for (int index = 0; index < peerCount; ++index) {
        HcclChannelDescInit(&descs[static_cast<size_t>(index)], 1);
        Session::Peer& peer = session->peers[static_cast<size_t>(index)];
        peer.rank = peers[index];
        CommLink* links = nullptr;
        uint32_t linkCount = 0;
        const HcclResult linkResult = HcclRankGraphGetLinks(
            session->comm, 0, static_cast<uint32_t>(session->rank),
            static_cast<uint32_t>(peers[index]), &links, &linkCount);
        if (linkResult != HCCL_SUCCESS || links == nullptr || linkCount == 0) {
            peer.protocolFromGraph = "UNKNOWN";
            continue;
        }
        peer.protocolFromGraph = ProtocolName(static_cast<int>(links[0].linkAttr.linkProtocol));
        peer.topoType = "layer0";
        peer.hop = links[0].linkAttr.hop;
        descs[static_cast<size_t>(index)].remoteRank = static_cast<uint32_t>(peers[index]);
        descs[static_cast<size_t>(index)].channelProtocol = static_cast<CommProtocol>(protocolValue);
        descs[static_cast<size_t>(index)].localEndpoint = links[0].srcEndpointDesc;
        descs[static_cast<size_t>(index)].remoteEndpoint = links[0].dstEndpointDesc;
        descs[static_cast<size_t>(index)].notifyNum = 4;
        descs[static_cast<size_t>(index)].memHandles = session->localHandles.data();
        descs[static_cast<size_t>(index)].memHandleNum = static_cast<uint32_t>(session->localHandles.size());
    }
    std::vector<ChannelHandle> channels(static_cast<size_t>(peerCount));
    const HcclResult acquire = HcclChannelAcquire(
        session->comm, static_cast<CommEngine>(engineValue), descs.data(),
        static_cast<uint32_t>(peerCount), channels.data());
    std::string json = "{\"ok\":";
    json += (acquire == HCCL_SUCCESS) ? "true" : "false";
    json += ",\"result\":" + std::to_string(acquire);
    json += ",\"engine\":\"" + std::string(engineName) + "\"";
    json += ",\"requested_protocol\":\"" + std::string(protocolName) + "\"";
    json += ",\"peers\":[";
    for (int index = 0; index < peerCount; ++index) {
        if (index != 0) {
            json += ",";
        }
        Session::Peer& peer = session->peers[static_cast<size_t>(index)];
        peer.channel = channels[static_cast<size_t>(index)];
        json += "{\"rank\":" + std::to_string(peer.rank);
        json += ",\"channel\":" + std::to_string(peer.channel);
        json += ",\"link_protocol_from_graph\":\"" + Escape(peer.protocolFromGraph) + "\"";
        json += ",\"hop\":" + std::to_string(peer.hop);
        json += ",\"remote_mems\":[";
        if (acquire == HCCL_SUCCESS && peer.channel != 0) {
            uint32_t memCount = 0;
            CommMem* remoteMems = nullptr;
            char** remoteTags = nullptr;
            const HcclResult memResult = HcclChannelGetRemoteMems(
                session->comm, peer.channel, &memCount, &remoteMems, &remoteTags);
            if (memResult == HCCL_SUCCESS && remoteMems != nullptr) {
                for (uint32_t memIndex = 0; memIndex < memCount; ++memIndex) {
                    if (memIndex != 0) {
                        json += ",";
                    }
                    const uint64_t address = reinterpret_cast<uint64_t>(remoteMems[memIndex].addr);
                    peer.remoteAddrs.push_back(address);
                    peer.remoteTags.push_back(
                        remoteTags != nullptr && remoteTags[memIndex] != nullptr ? remoteTags[memIndex] : "");
                    json += "{\"tag\":\"";
                    json += Escape(peer.remoteTags.back());
                    json += "\",\"addr\":" + std::to_string(address);
                    json += ",\"bytes\":" + std::to_string(remoteMems[memIndex].size) + "}";
                }
            }
        }
        json += "]}";
    }
    json += "]}";
    return CopyOut(json, buffer, capacity);
#else
    (void)handle;
    (void)engineName;
    (void)protocolName;
    (void)peers;
    (void)peerCount;
    return CopyOut("{\"ok\":false,\"reason\":\"hccl headers unavailable\"}", buffer, capacity);
#endif
}

/*
 * Submit one transfer.  `mode` is "write" or "read"; the bridge tries the plain
 * NBI overload first and falls back to the per-thread overload, recording which
 * one the runtime accepted so the caller can report the real submission path.
 */
L0_API int l0_session_transfer(int64_t handle, int peerIndex, const char* mode,
                               uint32_t localMemIndex, uint32_t remoteMemIndex,
                               uint64_t localOffset, uint64_t remoteOffset, uint64_t bytes,
                               char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCOMM && L0_BRIDGE_HAS_HCCL
    Session* session = (handle == 1) ? g_session : nullptr;
    if (session == nullptr || !session->commReady) {
        return CopyOut("{\"ok\":false,\"reason\":\"comm is not initialized\"}", buffer, capacity);
    }
    if (peerIndex < 0 || peerIndex >= static_cast<int>(session->peers.size())) {
        return CopyOut("{\"ok\":false,\"reason\":\"peer index out of range\"}", buffer, capacity);
    }
    Session::Peer& peer = session->peers[static_cast<size_t>(peerIndex)];
    if (peer.remoteAddrs.empty() || session->localMems.empty()) {
        return CopyOut("{\"ok\":false,\"reason\":\"no registered remote memory\"}", buffer, capacity);
    }
    if (localMemIndex >= session->localMems.size() || remoteMemIndex >= peer.remoteAddrs.size()) {
        return CopyOut("{\"ok\":false,\"reason\":\"memory index out of range\"}", buffer, capacity);
    }
    void* remote = reinterpret_cast<void*>(peer.remoteAddrs[remoteMemIndex] + remoteOffset);
    void* local = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(session->localMems[localMemIndex].addr) + localOffset);
    int32_t result = -1;
    std::string submission;
    if (strcmp(mode, "read") == 0) {
        result = HcommReadNbi(peer.channel, local, remote, bytes);
        submission = "plain_read_nbi";
    } else {
        result = HcommWriteNbi(peer.channel, remote, local, bytes);
        submission = "plain_write_nbi";
    }
    if (result != 0 && session->thread != 0) {
        if (strcmp(mode, "read") == 0) {
            result = HcommReadNbiOnThread(session->thread, peer.channel, local, remote, bytes);
            submission = "thread_read_nbi";
        } else {
            result = HcommWriteNbiOnThread(session->thread, peer.channel, remote, local, bytes);
            submission = "thread_write_nbi";
        }
    }
    if (result == 0) {
        session->submission = submission;
    }
    std::string json = "{\"ok\":";
    json += (result == 0) ? "true" : "false";
    json += ",\"result\":" + std::to_string(result);
    json += ",\"submission\":\"" + Escape(submission) + "\"";
    json += ",\"bytes\":" + std::to_string(bytes) + "}";
    return CopyOut(json, buffer, capacity);
#else
    (void)handle;
    (void)peerIndex;
    (void)mode;
    (void)localOffset;
    (void)remoteOffset;
    (void)bytes;
    return CopyOut("{\"ok\":false,\"reason\":\"hcomm/hccl headers unavailable\"}", buffer, capacity);
#endif
}

L0_API int l0_session_thread_alloc(int64_t handle, const char* engineName, uint32_t notifyNum,
                                   char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCOMM
    Session* session = (handle == 1) ? g_session : nullptr;
    if (session == nullptr) {
        return CopyOut("{\"ok\":false,\"reason\":\"session is not open\"}", buffer, capacity);
    }
    const int engineValue = EngineValue(engineName == nullptr ? "" : engineName);
    if (engineValue < 0) {
        return CopyOut("{\"ok\":false,\"reason\":\"unknown engine\"}", buffer, capacity);
    }
    if (session->thread != 0) {
        (void)HcommThreadFree(&session->thread, 1);
        session->thread = 0;
    }
    const uint32_t notify = notifyNum == 0 ? 1 : notifyNum;
    ThreadHandle thread = 0;
    const HcommResult result = HcommThreadAlloc(
        static_cast<CommEngine>(engineValue), 1, &notify, &thread);
    if (result == kHcommSuccess) {
        session->thread = thread;
    }
    std::string json = "{\"ok\":";
    json += (result == kHcommSuccess) ? "true" : "false";
    json += ",\"result\":" + std::to_string(result) + "}";
    return CopyOut(json, buffer, capacity);
#else
    (void)handle;
    (void)engineName;
    (void)notifyNum;
    return CopyOut("{\"ok\":false,\"reason\":\"hcomm headers unavailable\"}", buffer, capacity);
#endif
}

L0_API int l0_session_quiet(int64_t handle, char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCOMM
    Session* session = (handle == 1) ? g_session : nullptr;
    if (session == nullptr) {
        return CopyOut("{\"ok\":false,\"reason\":\"session is not open\"}", buffer, capacity);
    }
    int32_t lastResult = 0;
    for (Session::Peer& peer : session->peers) {
        if (peer.channel == 0) {
            continue;
        }
        int32_t result = -1;
        if (session->thread != 0) {
            result = HcommChannelDrainOnThread(session->thread, peer.channel);
        } else {
            result = HcommChannelFence(peer.channel);
        }
        if (result != 0) {
            lastResult = result;
        }
    }
    std::string json = "{\"ok\":";
    json += (lastResult == 0) ? "true" : "false";
    json += ",\"result\":" + std::to_string(lastResult) + "}";
    return CopyOut(json, buffer, capacity);
#else
    (void)handle;
    return CopyOut("{\"ok\":false,\"reason\":\"hcomm headers unavailable\"}", buffer, capacity);
#endif
}

/*
 * Produce this rank's root information blob.  The Python driver gathers these
 * blobs through the existing torch.distributed rendezvous; HCOMM/HCCL never
 * needs to know how they travelled.
 */
L0_API int l0_bridge_root_info(char* buffer, uint32_t capacity)
{
#if L0_BRIDGE_HAS_HCCL
    HcclRootInfo root{};
    const HcclResult result = HcclGetRootInfo(&root);
    if (result != HCCL_SUCCESS) {
        SetError("HcclGetRootInfo failed: " + std::to_string(result));
        return -1;
    }
    if (sizeof(root) > capacity) {
        SetError("root info buffer too small");
        return -2;
    }
    memcpy(buffer, &root, sizeof(root));
    return static_cast<int>(sizeof(root));
#else
    (void)buffer;
    (void)capacity;
    SetError("bridge was built without HCCL headers");
    return -1;
#endif
}

L0_API int l0_session_close(int64_t handle)
{
    Session* session = (handle == 1) ? g_session : nullptr;
    if (session == nullptr) {
        return 0;
    }
#if L0_BRIDGE_HAS_HCCL
    if (session->commReady && !session->peers.empty()) {
        std::vector<ChannelHandle> channels;
        for (Session::Peer& peer : session->peers) {
            if (peer.channel != 0) {
                channels.push_back(peer.channel);
            }
        }
        if (!channels.empty()) {
            (void)HcclChannelDestroy(session->comm, channels.data(),
                                     static_cast<uint32_t>(channels.size()));
        }
    }
    if (session->commReady) {
        (void)HcclCommDestroy(session->comm);
    }
#endif
#if L0_BRIDGE_HAS_HCOMM
    if (session->thread != 0) {
        (void)HcommThreadFree(&session->thread, 1);
    }
    if (session->endpoint != nullptr) {
        (void)HcommEndpointDestroy(session->endpoint);
    }
#endif
    if (session->aclReady) {
        (void)aclrtResetDevice(session->device);
    }
    delete session;
    g_session = nullptr;
    return 0;
}
