# EdgeEngine

EdgeEngine is AtomGradient's narrow native runtime for EdgeKit.

The package is intentionally not a general MLX replacement. It owns only the
runtime, tensor, kernel, and fixed model-family code needed by EdgeKit's
Qwen3.5/Qwen3.6, ASR, and TTS paths.

## Scope

- Native Metal command scheduling with EdgeEngine-owned memory controls.
- Minimal tensor/storage abstractions for inference.
- Model-family implementations required by EdgeKit.
- Development-time parity tests against the current MLX-based runtime.

## Non-goals

- No dependency on `mlx-swift`, `mlx-swift-lm`, or `mlx-audio-swift`.
- No public generic tensor framework API.
- No autograd or training runtime.

## License

EdgeEngine is released under the MIT License. See `LICENSE`.
