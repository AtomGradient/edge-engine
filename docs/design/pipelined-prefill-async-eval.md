# Pipelined Prefill: async_eval between chunks

**Status:** Implemented in `1.0.0-rc123`
**Upstream reference:** Apple mlx-swift-lm commit `4676fb9`
**Date:** 2026-05-21

## Implementation Status

Implemented for both text-only LLM generation and VLM text-token prefill:

- `LLMEngine.runCmlxLazyPrefill()` now schedules intermediate chunks with `prefillAsync`.
- `VLMEngine.runNativeVLMCmlxTokenPrefill()` now schedules intermediate text chunks with `prefillCmlxTokensAsync`.
- `QwenVLMNativeContainer` exposes the Cmlx async token prefill path.

Initial verification:

- `edge-engine`: `swift build -c debug` passed.
- `edge-kit`: `swift package edit edge-engine --path ../edge-engine && swift build -c debug` passed against the local engine change.

## Problem

Chunked prefill in edge-engine currently uses blocking `eval()` for all intermediate chunks and only uses `async_eval()` for the final chunk. The upstream fix (commit 4676fb9) showed that switching intermediate chunks from `eval()` to `async_eval()` yields ~10x prefill speedup on GDN models, because the CPU can build chunk N+1's computation graph while the GPU is still executing chunk N.

## Current State

### Call chain (LLM cmlx lazy path -- the primary path)

```
LLMEngine.runCmlxLazyPrefill()                           [edge-kit LLMEngine.swift:1482]
  for each intermediate chunk:
    session.prefill(tokenIDs: chunk)                      [edge-kit LLMEngine.swift:1554]
      QwenCmlxLazyDecodeSession.prefill(tokenIDs:)        [edge-engine QwenCmlxLazyDecodeSession.swift:267]
        EdgeMLXQwen35Session.prefill(tokenIDs:)           [edge-engine EdgeMLXQwen35Session.swift:885]
          edge_cmlx_qwen35_session_prefill()              [edge-engine shim.cpp:7319]
            qwen35_session_advance_tokens()               [shim.cpp:7258]
              qwen35_session_advance_token_indices()      [shim.cpp:7234]
                qwen35_session_advance_hidden()           [shim.cpp:7209]
                  qwen35_session_advance_hidden_with_state() [shim.cpp:6840]
                    >>> async_schedule=false <<<
                    eval(eval_outputs)                    [shim.cpp:7158]  <-- BLOCKING
  for the last chunk:
    session.prefillAsync(tokenIDs: chunk)                 [edge-kit LLMEngine.swift:1547]
      edge_cmlx_qwen35_session_prefill_async()            [shim.cpp:7378]
        qwen35_session_advance_token_indices(..., async_schedule=true, ...)
          qwen35_session_advance_hidden_with_state()
            >>> async_schedule=true <<<
            async_eval(eval_outputs)                      [shim.cpp:7154]
    session.nextToken()                                   -- materializes pending token
```

Key observation: intermediate chunks call `edge_cmlx_qwen35_session_prefill` which passes `async_schedule=false` to `qwen35_session_advance_token_indices`. This causes `eval()` (blocking) at shim.cpp:7158. The CPU stalls waiting for the GPU to finish chunk N before it can even begin building chunk N+1's graph.

### Call chain (VLM text-only prefill)

```
VLMEngine.runNativeVLMCmlxTokenPrefill()                  [edge-kit VLMEngine.swift:829]
  for each chunk (including last):
    container.prefillCmlxTokens(tokenIDs:)                [edge-engine QwenVLMNativeContainer.swift:276]
      cmlxDecoderSession.prefill(tokenIDs:)               -- always sync (blocking eval)
```

VLM currently uses blocking `prefill()` for ALL chunks, including the last. Even worse than LLM.

### Upstream pattern (commit 4676fb9)

```swift
while y.tokens.size > prefillStepSize {
    let input = y[.newAxis, ..<prefillStepSize]
    _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
    asyncEval(cache)       // <-- async for ALL intermediate chunks
    y = y[prefillStepSize...]
}
eval(cache)                // <-- single sync after the loop
```

The upstream approach: `asyncEval(cache)` after each intermediate model forward pass, then one `eval(cache)` after the loop exits to flush remaining async work. The final chunk still goes through the normal `TokenIterator` which does its own eval.

## Target State

Change intermediate chunked prefill from blocking to async:

```
for each intermediate chunk (NOT the last):
    session.prefillAsync(tokenIDs: chunk)     // was: session.prefill()
for the last chunk:
    session.prefillAsync(tokenIDs: chunk)     // unchanged
    session.nextToken()                       // materializes everything
```

This is simpler than upstream because edge-engine already has `prefillAsync()` wired end-to-end. The fix is a one-line change at each call site.

## Key Constraints

### 1. GDN recurrent state must remain correct

GDN layers (75% of Qwen3.5-4B's 36 layers) maintain conv state and recurrent state in `gdn_conv_states` / `gdn_recurrent_states` maps on `EdgeCmlxQwen35Session` (shim_internal.h:62-63). These are updated inside `qwen35_session_advance_hidden_with_state` at shim.cpp:6975-6978 via `insert_or_assign`. With `async_schedule=true`, these array objects are scheduled but not yet evaluated -- they hold lazy computation graphs, not materialized data. This is safe because:

- MLX arrays are immutable values with reference semantics to lazy computation graphs
- `insert_or_assign` replaces the map entry with the new lazy array
- The next chunk's GDN layer reads the lazy array from the map, which extends the computation graph
- `async_eval(eval_outputs)` at shim.cpp:7154 enqueues the entire batch (including stored states) for GPU execution
- All intermediate state is eventually materialized by the final `eval()` or `nextToken()`

This is exactly how upstream MLX KV cache works with `asyncEval(cache)`.

### 2. DSR attention eviction must not run mid-prefill

DSR eviction reads `attention_dsr_tokens_since_eviction` and `attention_active_lengths` to decide when to evict. With async intermediate chunks, the `decoded_token_count` is incremented at shim.cpp:7199 after each chunk's `async_eval`. Eviction logic in `qwen35_full_attention_decode_array` checks against `eviction_interval`. Since prefill processes all tokens in a single `qwen35_session_advance_hidden_with_state` call per chunk (not per-token), DSR eviction fires at chunk boundaries, not token boundaries. This behavior is unchanged -- the same code path runs regardless of `async_schedule`.

### 3. Prompt cache must not break

The prompt cache operates at the Swift layer in `LLMEngine.runCmlxLazyGenerate()` (LLMEngine.swift:1036-1094). It determines `prefillTokens` before calling `runCmlxLazyPrefill()`. The chunked prefill loop only processes the tokens it receives. The async change does not affect which tokens are prefilled or the token-level bookkeeping in `nativeDecodeSessionTokenIds`.

### 4. VLM media features path unaffected

VLM image/audio prefill goes through `prefillMediaFeatures()` / `prefillImageFeatures()`, which always call `edge_cmlx_qwen35_session_prefill_media_features` (sync). These are NOT chunked -- they process all media features in one call. The optimization targets only the text-token chunked prefill that follows. The VLM text chunk path in `runNativeVLMCmlxTokenPrefill` (VLMEngine.swift:829) should be updated in the same way.

### 5. prefill_fp16_attention_materialized_pending_clear

After shim.cpp:7193, when `clear_prefill_fp16_cache` is true, `mlx::core::clear_cache()` is called. With `async_schedule=true`, this clear happens after `async_eval`, not `eval`. The arrays are not yet materialized. This is safe: `clear_cache()` evicts unused cached allocations; it does not invalidate pending computation graphs. The in-flight async work holds references to its allocations. The pending_clear flag is only set during the first prefill chunk (when attention cache materializes from FP16 to quantized), so it fires once per session start.

### 6. Single command buffer prefill mode

When `useSingleCommandBufferPrefill` is enabled in the Metal configuration, the entire forward pass for a chunk is batched into one command buffer. With `async_eval`, this command buffer is submitted to the GPU but the CPU does not wait for completion. The next chunk's command buffer can be prepared immediately. This is the primary source of the 10x speedup -- GPU and CPU work overlap.

## Implementation Plan

### Change 1: LLMEngine intermediate chunks (edge-kit)

**File:** `edge-kit/Sources/EdgeInference/NativeDefault/LLMEngine.swift`
**Function:** `runCmlxLazyPrefill()` (line 1482)
**Line 1554:** Change `_ = try session.prefill(tokenIDs: chunk)` to `try session.prefillAsync(tokenIDs: chunk)`

Before:
```swift
_ = try session.prefill(tokenIDs: chunk)
```

After:
```swift
try session.prefillAsync(tokenIDs: chunk)
```

That is the entire LLM change. The sampled path already uses `prefillSampledAsync` for the last chunk. For intermediate chunks, the sampled path is not hit (intermediate chunks do not sample). The `prefillAsync` call builds the computation graph and enqueues it without blocking.

### Change 2: VLMEngine intermediate chunks (edge-kit)

**File:** `edge-kit/Sources/EdgeInference/NativeDefault/VLMEngine.swift`
**Function:** `runNativeVLMCmlxTokenPrefill()` (line 829)

This requires adding a new async prefill path. Currently VLM uses `container.prefillCmlxTokens()` which always calls the sync `session.prefill()`.

Option A (minimal): Add `prefillCmlxTokensAsync()` to `QwenVLMNativeContainer` that calls `session.prefillAsync()`. Use it for intermediate chunks.

Option B (deferred): Leave VLM as-is. VLM text-only prefill chunks are typically small because most tokens are media features. Defer to a follow-up.

Recommendation: Option A is low risk since the plumbing already exists.

**File:** `edge-engine/Sources/EdgeEngine/Models/Qwen/QwenVLMNativeContainer.swift`
Add:
```swift
public func prefillCmlxTokensAsync(tokenIDs: [Int]) throws {
    try cmlxDecoderSession.prefillAsync(tokenIDs: tokenIDs)
}
```

Then in `VLMEngine.swift:858`, change:
```swift
_ = try container.prefillCmlxTokens(tokenIDs: chunk)
```
to:
```swift
try container.prefillCmlxTokensAsync(tokenIDs: chunk)
```

### Change 3: Diagnostic logging update (edge-kit)

Update the diagnostic log at LLMEngine.swift:1555-1557 to indicate async mode:
```swift
diagnosticSink?(
    "cmlx_lazy_prefill_chunk_done index=\(chunkIndex) tokens=\(chunk.count) final=false async=true"
)
```

Same for VLMEngine.swift:859-861.

### No changes needed in edge-engine C++ layer

The C shim already has `edge_cmlx_qwen35_session_prefill_async` (shim.cpp:7378) which passes `async_schedule=true`. The Swift wrapper `EdgeMLXQwen35Session.prefillAsync()` (EdgeMLXQwen35Session.swift:1096) and `QwenCmlxLazyDecodeSession.prefillAsync()` (QwenCmlxLazyDecodeSession.swift:276) already exist. No new C++ code is needed.

## Files Changed Summary

| Repository | File | Change |
|------------|------|--------|
| edge-kit | `Sources/EdgeInference/NativeDefault/LLMEngine.swift` | Line 1554: `prefill` -> `prefillAsync` |
| edge-kit | `Sources/EdgeInference/NativeDefault/VLMEngine.swift` | Line 858: use async for intermediate chunks |
| edge-engine | `Sources/EdgeEngine/Models/Qwen/QwenVLMNativeContainer.swift` | Add `prefillCmlxTokensAsync()` |

## Acceptance Criteria

### Performance targets

| Metric | Baseline (current) | Target | Measurement |
|--------|-------------------|--------|-------------|
| TTFT (first turn, 500 tokens, GDN model) | Measured via `cmlx_lazy_phase_timings prefillMs` | 5-10x reduction | Device test, Qwen3.5-4B-4bit, iPhone 15 PM |
| TTFT (first turn, 500 tokens, FA-only model) | Baseline | ~1x (no regression, minor improvement possible) | Device test |
| Decode TPS | Baseline | No regression (within 5%) | Device test, 20-turn conversation |
| Memory peak during prefill | Baseline | No regression (within 50MB) | `phys_footprint` monitoring |

### Functional correctness

- [ ] Chunked prefill with 2+ chunks produces identical first token as single-chunk prefill for the same prompt
- [ ] Multi-turn conversation: prompt cache hit rate unchanged
- [ ] DSR eviction timing unchanged (verify via `cmlx_lazy_dsr_policy_updated` diagnostics)
- [ ] VLM image+text prefill produces coherent output
- [ ] GDN state continuity: 20-turn deep conversation shows no quality degradation vs baseline
- [ ] Prompt cache suffix reuse still works after pipelined prefill

### Test plan

1. **Unit verification:** Run existing edge-engine Swift tests (`swift test`)
2. **Smoke test:** `tests/smoke_test/run_smoke.sh llm` with Qwen3.5-4B-4bit
3. **Device test A/B:** Run `tests/device_test/run_device_test.sh` with EDGE_CMLX_EVAL_PROFILE=1, compare `prefillMs` between baseline and patched builds
4. **VLM smoke test:** `tests/smoke_test/run_smoke.sh vlm` -- verify no regression
5. **Long conversation:** Device test `deep` level, 20 turns, verify no forgetting score regression

## Risks

### Risk 1: Peak memory spike during overlapped prefill (Medium)

With async intermediate chunks, the GPU may be working on chunk N while the CPU has already built the graph for chunk N+1. This means two chunks' worth of intermediate activations may be resident simultaneously. For a 512-token chunk on Qwen3.5-4B (hidden_size=2560, 36 layers), each chunk's activations are ~2560 * 512 * 2 bytes * 36 layers = ~94MB. Two overlapped chunks = ~188MB peak vs ~94MB sequential.

**Mitigation:** On iOS, the existing `prefillStepSize` policy already reduces chunk size under memory pressure (64/128/256 based on thermal state, see InferencePolicy.swift:311-322). The doubled peak is within typical iOS headroom since the model itself is ~2.5GB and jetsam limits are typically 3-4GB. Monitor `phys_footprint` in device test to confirm.

### Risk 2: GDN state evaluation ordering (Low)

GDN conv_state and recurrent_state are stored as lazy MLX arrays in the session's maps. When chunk N+1 reads them, it extends the graph. MLX guarantees correct evaluation order within a single `eval()` or `async_eval()` call -- all outputs are evaluated respecting their dependencies. Since each chunk's `async_eval` includes the stored states in `eval_outputs`, the GPU will evaluate them in order.

The risk would only materialize if someone reads `.data<>()` on a GDN state between chunks (forcing premature evaluation). No code path does this.

### Risk 3: prefill_fp16_attention clear_cache interaction (Low)

`clear_cache()` after `async_eval` at shim.cpp:7193-7195 only evicts unreferenced allocations. In-flight async work holds references. However, if `clear_cache()` is called between two async chunks, it could evict allocations that were freed by the GPU completing chunk N, before chunk N+1 needs similar-sized allocations. This could cause re-allocation rather than reuse.

**Mitigation:** The `prefill_fp16_attention_materialized_pending_clear` flag is only set once (first prefill chunk). After the clear, it is set to false. Subsequent chunks do not trigger this path. No action needed.

### Risk 4: Online calibrator interference (Low)

`OnlineCalibrator` records per-turn metrics including `prefillMs`. The pipelined prefill will dramatically reduce `prefillMs`, which could cause the calibrator to make different decisions about `maxOpsPerBuffer` and `prefillStepSize`. This is desirable behavior -- faster prefill means the calibrator should not need to reduce step size.

**Mitigation:** Monitor calibrator decisions in device test logs. If the calibrator over-aggressively increases step sizes, add a guard.
