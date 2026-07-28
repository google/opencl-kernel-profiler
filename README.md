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
  m opencl-kernel-profiler
  ```
* Push the library and the `.lay` file to the device. Note that `/vendor` partition is usually read-only, so you may need to remount it first:
  ```bash
  adb root
  adb disable-verity
  adb reboot
  # Wait for the device to reboot, then:
  adb root
  adb remount
  adb push $OUT/vendor/lib64/opencl-kernel-profiler.so /vendor/lib64/
  adb push $OUT/vendor/etc/Khronos/OpenCL/layers/opencl-kernel-profiler.lay /vendor/etc/Khronos/OpenCL/layers/
  ```

Any application using the `OpenCL-ICD-Loader` will go through the `opencl-kernel-profiler`.

# Environment Variables

The profiler can be configured at runtime using the following environment variables:

* `CLKP_TRACE_DEST` (InProcess backend only): File path where the Perfetto trace will be saved. (Default: `opencl-kernel-profiler.trace`).
* `CLKP_TRACE_MAX_SIZE` (InProcess backend only): Maximum size of the trace buffer in KB. (Default: `1024`).
* `CLKP_KERNEL_DIR`: Directory path where kernel sources, binaries, and IL will be dumped. If not set, dumping is disabled.

# Using the Trace

Once traces have been generated, one can view them using the [Perfetto Trace Viewer](https://ui.perfetto.dev).

It is also possible to make SQL queries using the [trace_processor](https://perfetto.dev/docs/analysis/trace-processor) tool.
See the [Perfetto SQL Analysis Quickstart](https://perfetto.dev/docs/quickstart/trace-analysis).

Here is a simple example to extract all kernel sources from a trace:
```bash
echo "SELECT EXTRACT_ARG(arg_set_id, 'debug.string') FROM slice WHERE slice.name='clCreateProgramWithSource-args'" \
  | ./trace_processor -q /dev/stdin <opencl-kernel-profiler.trace>
```

# Dumping Kernel Sources to Disk

If `CLKP_KERNEL_DIR` is set, the profiler dumps all programs/kernels to disk:
* OpenCL C sources are saved with a `.cl` extension.
* Compiled binaries are saved with a `.bin` extension.
* Intermediate Language (IL) is saved with `.spv` (for SPIR-V) or `.il` extension.
* SPIR-V disassembly is saved with `.spvasm` extension (if `SPIRV_DISASSEMBLY` is enabled).

If `CLKP_KERNEL_DIR` is not set, no files are written. This dumping occurs independently of Perfetto tracing.

# How it Works

The layer intercepts the following OpenCL APIs to instrument execution and dump resources:

* `clCreateCommandQueue` / `clCreateCommandQueueWithProperties`: Forces `CL_QUEUE_PROFILING_ENABLE` to ensure hardware timestamps are available.
* `clCreateProgramWithSource`: Emits the source code to the trace (as an instant event) and dumps it to `CLKP_KERNEL_DIR` if configured.
* `clCreateProgramWithBinary`: Dumps the binary to `CLKP_KERNEL_DIR` if configured.
* `clCreateProgramWithIL`: Emits SPIR-V disassembly (if enabled) to the trace and dumps IL/disassembly to `CLKP_KERNEL_DIR` if configured.
* `clCreateKernel`: Tracks kernel-to-program relationships and kernel names.
* `clEnqueueNDRangeKernel`: Enqueues the kernel and registers a completion callback. The callback retrieves GPU start/end timestamps via `clGetEventProfilingInfo` and emits a corresponding Perfetto slice.
* `clReleaseCommandQueue`: Cleans up the background helper thread and resources associated with the queue.

Every intercepted host API call also generates a host-side Perfetto slice.
