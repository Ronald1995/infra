Superseded by the consolidated implementation in [#20865](https://github.com/vllm-project/vllm-ascend/pull/20865), following RFC #18920 revision v6.

The consolidated PR directly inherits GPUWorker, calls the real parent constructor,
preserves all existing NPU Worker methods, and explicitly overrides the 29 other
newly visible methods/properties with NotImplementedError until individually
validated. Only the two required constructor hooks have NPU implementations.

This draft is closed without merging. Its shutdown migration and selective
interface enablement are deferred to focused follow-ups; no successor stack
is required. See #20865 for the current code, dependency and validation status.

- vLLM main: https://github.com/vllm-project/vllm/commit/24420b7a341af37a9bc3186e6fc6062512e0aa58

