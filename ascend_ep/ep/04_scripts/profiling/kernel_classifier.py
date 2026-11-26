"""Conservative classifier for visible device activity."""


def classify(name, category=""):
    text = (name + " " + category).casefold().replace("_", "")
    if any(key in text for key in ("megamoe", "alltoallvgroupedmatmul", "groupedmatmulalltoallv",
                                  "dispatchffncombine", "fuseddeepmoe", "combineaddrmsnorm")):
        return "fused"
    if any(key in text for key in ("hccl", "allreduce", "allgather", "reducescatter", "alltoall",
                                  "shmem", "distribut dispatch", "moedistributedispatch",
                                  "moedistributecombine", "dispatchkernel", "combinekernel",
                                  "moonep", "cammoe", "epdispatch", "epcombine")):
        return "communication"
    if any(key in text for key in ("matmul", "gemm", "gmm", "swiglu")):
        return "compute"
    if any(key in text for key in ("memcpy", "tensormove")):
        return "copy"
    return "other"
