// swift-tools-version: 6.1
// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import PackageDescription

let cmlxExcludes: [String] = [
    "json",
    "metal-cpp/README.md",
    "mlx/3rdparty/.clang-format",
    "mlx/CMakeLists.txt",
    "mlx/backend/common/CMakeLists.txt",
    "mlx/backend/cpu/CMakeLists.txt",
    "mlx/backend/cpu/gemms/simd_bf16.cpp",
    "mlx/backend/cpu/gemms/simd_fp16.cpp",
    "mlx/backend/cuda/CMakeLists.txt",
    "mlx/backend/cuda/allocator.cpp",
    "mlx/backend/cuda/arange.cu",
    "mlx/backend/cuda/arg_reduce.cu",
    "mlx/backend/cuda/binary",
    "mlx/backend/cuda/binary_two.cu",
    "mlx/backend/cuda/compiled.cpp",
    "mlx/backend/cuda/conv",
    "mlx/backend/cuda/conv.cpp",
    "mlx/backend/cuda/copy",
    "mlx/backend/cuda/copy.cu",
    "mlx/backend/cuda/cublas_utils.cpp",
    "mlx/backend/cuda/cudnn_utils.cpp",
    "mlx/backend/cuda/custom_kernel.cpp",
    "mlx/backend/cuda/delayload.cpp",
    "mlx/backend/cuda/device",
    "mlx/backend/cuda/device.cpp",
    "mlx/backend/cuda/device_info.cpp",
    "mlx/backend/cuda/distributed.cu",
    "mlx/backend/cuda/eval.cpp",
    "mlx/backend/cuda/event.cu",
    "mlx/backend/cuda/fence.cpp",
    "mlx/backend/cuda/fft.cu",
    "mlx/backend/cuda/gemms",
    "mlx/backend/cuda/hadamard.cu",
    "mlx/backend/cuda/indexing.cpp",
    "mlx/backend/cuda/jit_module.cpp",
    "mlx/backend/cuda/kernel_utils.cu",
    "mlx/backend/cuda/layer_norm.cu",
    "mlx/backend/cuda/load.cpp",
    "mlx/backend/cuda/logsumexp.cu",
    "mlx/backend/cuda/matmul.cpp",
    "mlx/backend/cuda/primitives.cpp",
    "mlx/backend/cuda/quantized",
    "mlx/backend/cuda/random.cu",
    "mlx/backend/cuda/reduce",
    "mlx/backend/cuda/reduce.cu",
    "mlx/backend/cuda/rms_norm.cu",
    "mlx/backend/cuda/rope.cu",
    "mlx/backend/cuda/scan.cu",
    "mlx/backend/cuda/scaled_dot_product_attention.cpp",
    "mlx/backend/cuda/scaled_dot_product_attention.cu",
    "mlx/backend/cuda/slicing.cpp",
    "mlx/backend/cuda/softmax.cu",
    "mlx/backend/cuda/sort.cu",
    "mlx/backend/cuda/steel",
    "mlx/backend/cuda/ternary.cu",
    "mlx/backend/cuda/unary",
    "mlx/backend/cuda/utils.cpp",
    "mlx/backend/cuda/worker.cpp",
    "mlx/backend/gpu/CMakeLists.txt",
    "mlx/backend/metal/CMakeLists.txt",
    "mlx/backend/metal/kernels",
    "mlx/backend/metal/make_compiled_preamble.sh",
    "mlx/backend/metal/no_metal.cpp",
    "mlx/backend/metal/nojit_kernels.cpp",
    "mlx/backend/no_cpu",
    "mlx/backend/no_gpu",
    "mlx/distributed/CMakeLists.txt",
    "mlx/distributed/jaccl/CMakeLists.txt",
    "mlx/distributed/jaccl/jaccl.cpp",
    "mlx/distributed/jaccl/lib",
    "mlx/distributed/mpi/CMakeLists.txt",
    "mlx/distributed/mpi/mpi.cpp",
    "mlx/distributed/nccl/CMakeLists.txt",
    "mlx/distributed/nccl/nccl.cpp",
    "mlx/distributed/ring/CMakeLists.txt",
    "mlx/distributed/ring/ring.cpp",
    "mlx/io/CMakeLists.txt",
    "mlx/io/gguf.cpp",
    "mlx/io/gguf_quants.cpp",
    "mlx/io/no_safetensors.cpp",
]

let package = Package(
    name: "edge-engine",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "EdgeEngine",
            targets: ["EdgeEngine"]
        ),
        .executable(
            name: "edge-smoke",
            targets: ["edge-smoke"]
        ),
        .executable(
            name: "edge-speech-preflight",
            targets: ["edge-speech-preflight"]
        ),
        .executable(
            name: "edge-asr",
            targets: ["edge-asr"]
        ),
        .executable(
            name: "edge-tts",
            targets: ["edge-tts"]
        ),
        .executable(
            name: "edge-vlm-preflight",
            targets: ["edge-vlm-preflight"]
        ),
        .executable(
            name: "edge-vlm",
            targets: ["edge-vlm"]
        ),
        .executable(
            name: "edge-llm",
            targets: ["edge-llm"]
        ),
        .executable(
            name: "edge-bench",
            targets: ["edge-bench"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.2"),
    ],
    targets: [
        .target(
            name: "EdgeEngine",
            dependencies: [
                "EdgeCmlx",
                "EdgeCmlxResources",
                "CmlxMetalKernels",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("ImageIO"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
            ]
        ),
        .target(
            name: "EdgeCmlx",
            path: "Sources/Cmlx",
            exclude: cmlxExcludes,
            publicHeadersPath: "shim/include",
            cSettings: [
                .headerSearchPath("."),
            ],
            cxxSettings: [
                .headerSearchPath("."),
                .headerSearchPath("fmt/include"),
                .headerSearchPath("json/single_include/nlohmann"),
                .headerSearchPath("metal-cpp"),
                .define("_METAL_"),
                .define("ACCELERATE_NEW_LAPACK"),
                .define("CA", to: "EdgeCA"),
                .define("EDGE_METALCPP_PRIVATE_C_SYMBOLS"),
                .define("fmt", to: "edge_fmt"),
                .define("METAL_PATH", to: "\"default.metallib\""),
                .define("mlx", to: "edge_mlx"),
                .define("MLX_USE_ACCELERATE"),
                .define("MLX_VERSION", to: "\"0.32.0\""),
                .define("MLX_STATIC"),
                .define("MTL", to: "EdgeMTL"),
                .define("MTL4", to: "EdgeMTL4"),
                .define("MTL4FX", to: "EdgeMTL4FX"),
                .define("MTLFX", to: "EdgeMTLFX"),
                .define("NS", to: "EdgeNS"),
                .define("SWIFTPM_BUNDLE", to: "\"edge-engine_EdgeCmlxResources\""),
                .define("SWIFTPM_BUNDLE_ALT", to: "\"edge_engine_EdgeCmlxResources\""),
                .define("get_kernel_preamble", to: "edge_mlx_get_kernel_preamble"),
                .define("get_prebuilt_preamble", to: "edge_mlx_get_prebuilt_preamble"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("QuartzCore"),
            ]
        ),
        .target(
            name: "EdgeCmlxResources",
            path: "Sources/EdgeCmlxResources",
            resources: [
                .copy("default.metallib"),
            ]
        ),
        .target(
            name: "CmlxMetalKernels",
            path: "Sources/CmlxMetalKernels",
            resources: [
                .copy("kernels"),
            ]
        ),
        .testTarget(
            name: "EdgeEngineTests",
            dependencies: ["EdgeEngine"]
        ),
        .executableTarget(
            name: "edge-smoke",
            dependencies: ["EdgeEngine"]
        ),
        .executableTarget(
            name: "edge-speech-preflight",
            dependencies: ["EdgeEngine"]
        ),
        .executableTarget(
            name: "edge-asr",
            dependencies: [
                "EdgeEngine",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
            ]
        ),
        .executableTarget(
            name: "edge-tts",
            dependencies: [
                "EdgeEngine",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "edge-vlm-preflight",
            dependencies: ["EdgeEngine"]
        ),
        .executableTarget(
            name: "edge-vlm",
            dependencies: [
                "EdgeEngine",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "edge-llm",
            dependencies: [
                "EdgeEngine",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "edge-bench",
            dependencies: ["EdgeEngine"]
        ),
    ],
    swiftLanguageModes: [.v6],
    cxxLanguageStandard: .cxx20
)
