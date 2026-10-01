# OpenCL Kernel Profiler

`opencl-kernel-profiler` is a perfetto-based OpenCL kernel profiler using the layering capability of the [OpenCL-ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader#about-layers)

# Legal

`opencl-kernel-profiler` is licensed under the terms of the [Apache 2.0 license](LICENSE).

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

# Dependencies

`opencl-kernel-profiler` depends on the following:

* [OpenCL-ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader)
* [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers)
* [perfetto](https://github.com/google/perfetto)
* [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools) (optional: to disassemble SPIR-V IL when `SPIRV_DISASSEMBLY` is enabled)

`opencl-kernel-profiler` also (obviously) depends on a OpenCL implementation.

# Building

`opencl-kernel-profiler` uses CMake for its build system.

To compile it, run:
```bash
cmake -B <build_dir> -S <path-to-opencl-kernel-profiler> \
  -DOPENCL_HEADER_PATH=<path-to-opencl-header> \
  -DPERFETTO_SDK_PATH=<path-to-perfetto-sdk>
cmake --build <build_dir>
```

For real-world examples, see:
- ChromeOS [ebuild](https://chromium.googlesource.com/chromiumos/overlays/chromiumos-overlay/+/main/dev-libs/opencl-kernel-profiler/opencl-kernel-profiler-0.0.1.ebuild)
- GitHub presubmit [configuration](https://github.com/rjodinchr/opencl-kernel-profiler/blob/main/.github/workflows/presubmit.yml)

# Build Options

* `PERFETTO_SDK_PATH` (REQUIRED): Path to [perfetto](https://github.com/google/perfetto) SDK (expects `perfetto.cc` and `perfetto.h` in this directory).
* `PERFETTO_LIBRARY`: Name of an existing perfetto library to link against (avoids compiling `perfetto.cc`).
* `OPENCL_HEADER_PATH`: Path to [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers).
* `OPENCL_LIBRARY_PATH`: Path to the directory containing `libOpenCL.so` (used to link `clkp-runner` on Linux/Android).
* `BACKEND`: Perfetto backend to use:
  * `InProcess` (default): The application generates the traces directly ([Perfetto In-Process Mode](https://perfetto.dev/docs/instrumentation/tracing-sdk#in-process-mode)).
  * `System`: The system-wide Perfetto daemon (`traced`) collects the traces ([Perfetto System Mode](https://perfetto.dev/docs/instrumentation/tracing-sdk#system-mode)).
* `TRACE_MAX_SIZE` (InProcess only): Default maximum trace buffer size in KB. Can be overridden at runtime. (Default: `1024`).
* `TRACE_DEST` (InProcess only): Default file path for the trace. Can be overridden at runtime. (Default: `opencl-kernel-profiler.trace`).
* `SPIRV_DISASSEMBLY` (optional): Enables SPIR-V disassembly in traces. Requires [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools).

# Running with OpenCL Kernel Profiler

To run an application with the `opencl-kernel-profiler`, one need to ensure the following point

* The application will link with the [OpenCL-ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader). If not the case, one can override `LD_LIBRARY_PATH` to point to where the `libOpenCL.so` coming from the ICD Loader is.
* The ICD Loader is build with [layers enable](https://github.com/KhronosGroup/OpenCL-ICD-Loader#about-layers) (`ENABLE_OPENCL_LAYERS=ON`).
* The ICD Loader is using the correct [OpenCL implementation](https://github.com/KhronosGroup/OpenCL-ICD-Loader#about-layers). If not the case, one can override `OCL_ICD_FILENAMES` to point to the appropriate OpenCL implementation library.

## On ChromeOS

Make sure to have emerged and deployed the `opencl-icd-loader` as well as the `opencl-kernel-profiler`.

Then run the application using `opencl-kernel-profiler.sh`. This script will take care of setting all the environment variables needed to run with the `opencl-kernel-profiler`.

## On Android

* Clone the project under `<aosp>/external/opencl-kernel-profiler`
* Compile the project:
  ```bash
  m opencl-kernel-profiler clkp-extractor clkp-runner
  ```
* Push the library, tools, and the `.lay` file to the device. Note that `/vendor` partition is usually read-only, so you may need to remount it first:
  ```bash
  adb root
  adb disable-verity
  adb reboot
  # Wait for the device to reboot, then:
  adb root
  adb remount
  adb shell mkdir -p /vendor/etc/Khronos/OpenCL/layers/
  adb push $OUT/vendor/lib64/opencl-kernel-profiler.so /vendor/lib64/
  adb push $OUT/vendor/bin/clkp-extractor /vendor/bin/
  adb push $OUT/vendor/bin/clkp-runner /vendor/bin/
  adb push $OUT/vendor/etc/Khronos/OpenCL/layers/opencl-kernel-profiler.lay /vendor/etc/Khronos/OpenCL/layers/
  adb shell chcon u:object_r:same_process_hal_file:s0 /vendor/lib64/opencl-kernel-profiler.so
  ```

Any application using the `OpenCL-ICD-Loader` will go through the `opencl-kernel-profiler`.

# Environment Variables

The profiler can be configured at runtime using the following environment variables:

* `CLKP_TRACE_DEST` (InProcess backend only): File path where the Perfetto trace will be saved. (Default: `opencl-kernel-profiler.trace`).
* `CLKP_TRACE_MAX_SIZE` (InProcess backend only): Maximum size of the trace buffer in KB. (Default: `1024`).
* `CLKP_KERNEL_DIR`: Directory path where kernel sources, binaries, and IL will be dumped. If not set, dumping is disabled.
* `CLKP_DISABLE_PROGRAM_BINARY_CACHE`: When set to `1`, reports zero size for `CL_PROGRAM_BINARY_SIZES` in `clGetProgramInfo` so applications fall back to `clCreateProgramWithSource` or `clCreateProgramWithIL`.

# Using the Trace

Once traces have been generated, one can view them using the [Perfetto Trace Viewer](https://ui.perfetto.dev).

It is also possible to make SQL queries using the [trace_processor](https://perfetto.dev/docs/analysis/trace-processor) tool.
See the [Perfetto SQL Analysis Quickstart](https://perfetto.dev/docs/quickstart/trace-analysis).

Here is a simple example to extract all kernel sources from a trace:
```bash
echo "SELECT EXTRACT_ARG(arg_set_id, 'debug.string') FROM slice WHERE slice.name='clCreateProgramWithSource-args'" \
  | ./trace_processor -q /dev/stdin <opencl-kernel-profiler.trace>
```

# Trace Extraction and Replay (`clkp-extractor` & `clkp-runner`)

On Linux and Android, `opencl-kernel-profiler` builds two standalone tools to inspect, filter, and replay captured OpenCL Perfetto traces:

## `clkp-extractor`

`clkp-extractor` inspects Perfetto traces and extracts self-contained `.perfetto-trace` files containing only the OpenCL (`clkp`) events (optionally trimmed to a dispatch or call window while preserving all required setup and resource creation events).

```bash
# List all recorded kernel dispatches and command buffer executions in a trace
clkp-extractor --list-dispatches -i <input.perfetto-trace>

# Extract all OpenCL events into a compact trace
clkp-extractor -i <input.perfetto-trace> -o <extracted.perfetto-trace>

# Extract a specific dispatch window [start_dispatch, end_dispatch]
clkp-extractor -i <input.perfetto-trace> -o <windowed.perfetto-trace> \
  --start-dispatch 10 --end-dispatch 25
```

Options:
* `-i, --input <file>`: Input Perfetto trace file.
* `-o, --output <file>`: Output filtered Perfetto trace file.
* `-l, --list-dispatches`: List all OpenCL dispatches and command buffer executions.
* `--start-dispatch <id>` / `--end-dispatch <id>`: Inclusive `dispatch_id` window to extract.
* `--start-call <id>` / `--end-call <id>`: Inclusive `call_id` window to extract.

## `clkp-runner`

`clkp-runner` replays a raw or extracted `.perfetto-trace` against an OpenCL driver, separating one-time resource/program/command-buffer setup from a timed execution loop and reporting GPU profiling statistics (aggregated by target by default, or per dispatch with `--by-dispatch`).

```bash
# Replay a trace with 5 warmup iterations and 20 measured iterations
clkp-runner -i <extracted.perfetto-trace> -m 5 -n 20 --json results.json

# Report profiling statistics per dispatch_id instead of aggregating by target
clkp-runner -i <extracted.perfetto-trace> -m 5 -n 20 --by-dispatch
```

Options:
* `-i, --input <file>`: Input Perfetto trace file.
* `-p, --platform <idx>`: OpenCL platform index (default: `0`).
* `-d, --device <idx>`: OpenCL device index (default: `0`).
* `-m, --warmup <count>`: Number of warmup iterations (default: `0`).
* `-n, --iterations <count>`: Number of measured iterations (default: `1`).
* `--start-dispatch <id>` / `--end-dispatch <id>`: Inclusive `dispatch_id` range to replay.
* `--use-command-buffer`: Promote standard queue dispatches into a `cl_command_buffer_khr`.
* `--emulate-command-buffer`: Emulate `cl_khr_command_buffer` calls via standard queue enqueues.
* `--allow-lws-fallback`: Fall back to `NULL` `local_work_size` if the replay device rejects the captured work-group size.
* `--by-dispatch`: Aggregate profiling results by `dispatch_id` instead of by target.
* `--json <file>`: Write profiling results to a JSON file.
* `-v, --verbose`: Verbose logging of replayed OpenCL calls.

# Dumping Kernel Sources to Disk

If `CLKP_KERNEL_DIR` is set, the profiler dumps all programs/kernels to disk:
* OpenCL C sources are saved with a `.cl` extension.
* Compiled binaries are saved with a `.bin` extension.
* Intermediate Language (IL) is saved with `.spv` (for SPIR-V) or `.il` extension.
* SPIR-V disassembly is saved with `.spvasm` extension (if `SPIRV_DISASSEMBLY` is enabled).

If `CLKP_KERNEL_DIR` is not set, no files are written. This dumping occurs independently of Perfetto tracing.

# How it Works

The layer intercepts OpenCL APIs (including `cl_khr_command_buffer` and `cl_khr_command_buffer_mutable_dispatch` extensions) to instrument execution, record state for replay, and dump resources:

* `clCreateCommandQueue` / `clCreateCommandQueueWithProperties`: Forces `CL_QUEUE_PROFILING_ENABLE` to ensure hardware timestamps are available.
* `clCreateProgramWithSource`: Emits the source code to the trace (as chunked instant events) and dumps it to `CLKP_KERNEL_DIR` if configured.
* `clCreateProgramWithBinary`: Emits binary chunks to the trace and dumps the binary to `CLKP_KERNEL_DIR` if configured.
* `clCreateProgramWithIL`: Emits IL chunks and SPIR-V disassembly (if enabled) to the trace and dumps IL/disassembly to `CLKP_KERNEL_DIR` if configured.
* `clCreateContext*`, `clBuildProgram`, `clCompileProgram`, `clLinkProgram`, `clCreateKernel*`, `clSetKernelArg`, `clCreateBuffer*`, `clCreateImage*`, `clCreateSampler*`, `clCreateCommandBufferKHR`, `clCommand*KHR`, `clUpdateMutableCommandsKHR`: Tracks object relationships and records parameters in the trace for `clkp-extractor` and `clkp-runner`.
* `clEnqueueNDRangeKernel` / `clEnqueueTask` / `clEnqueueCommandBufferKHR`: Enqueues the dispatch or command buffer and registers a completion callback. The callback retrieves GPU start/end timestamps via `clGetEventProfilingInfo` and emits a corresponding Perfetto slice.
* `clReleaseCommandQueue`: Cleans up the background helper thread and resources associated with the queue.

Every intercepted host API call also generates a host-side Perfetto slice.
