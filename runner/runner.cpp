// Copyright 2026 The OpenCL Kernel Profiler authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define CL_TARGET_OPENCL_VERSION 300
#define CL_ENABLE_BETA_EXTENSIONS
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include "clkp_common.hpp"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

struct ExtensionFunctions {
    clCreateCommandBufferKHR_fn clCreateCommandBufferKHR = nullptr;
    clFinalizeCommandBufferKHR_fn clFinalizeCommandBufferKHR = nullptr;
    clRetainCommandBufferKHR_fn clRetainCommandBufferKHR = nullptr;
    clReleaseCommandBufferKHR_fn clReleaseCommandBufferKHR = nullptr;
    clEnqueueCommandBufferKHR_fn clEnqueueCommandBufferKHR = nullptr;
    clCommandBarrierWithWaitListKHR_fn clCommandBarrierWithWaitListKHR = nullptr;
    clCommandCopyBufferKHR_fn clCommandCopyBufferKHR = nullptr;
    clCommandCopyBufferRectKHR_fn clCommandCopyBufferRectKHR = nullptr;
    clCommandCopyBufferToImageKHR_fn clCommandCopyBufferToImageKHR = nullptr;
    clCommandCopyImageKHR_fn clCommandCopyImageKHR = nullptr;
    clCommandCopyImageToBufferKHR_fn clCommandCopyImageToBufferKHR = nullptr;
    clCommandFillBufferKHR_fn clCommandFillBufferKHR = nullptr;
    clCommandFillImageKHR_fn clCommandFillImageKHR = nullptr;
    clCommandNDRangeKernelKHR_fn clCommandNDRangeKernelKHR = nullptr;
    clUpdateMutableCommandsKHR_fn clUpdateMutableCommandsKHR = nullptr;

    void load(cl_platform_id platform)
    {
        auto get_ext = [&](const char *name) -> void * {
            void *fn = clGetExtensionFunctionAddressForPlatform(platform, name);
            if (!fn) {
                fn = clGetExtensionFunctionAddress(name);
            }
            return fn;
        };

#define LOAD_EXT(fn) this->fn = reinterpret_cast<fn##_fn>(get_ext(#fn))
        LOAD_EXT(clCreateCommandBufferKHR);
        LOAD_EXT(clFinalizeCommandBufferKHR);
        LOAD_EXT(clRetainCommandBufferKHR);
        LOAD_EXT(clReleaseCommandBufferKHR);
        LOAD_EXT(clEnqueueCommandBufferKHR);
        LOAD_EXT(clCommandBarrierWithWaitListKHR);
        LOAD_EXT(clCommandCopyBufferKHR);
        LOAD_EXT(clCommandCopyBufferRectKHR);
        LOAD_EXT(clCommandCopyBufferToImageKHR);
        LOAD_EXT(clCommandCopyImageKHR);
        LOAD_EXT(clCommandCopyImageToBufferKHR);
        LOAD_EXT(clCommandFillBufferKHR);
        LOAD_EXT(clCommandFillImageKHR);
        LOAD_EXT(clCommandNDRangeKernelKHR);
        LOAD_EXT(clUpdateMutableCommandsKHR);
#undef LOAD_EXT
    }

    bool supports_command_buffer() const
    {
        return clCreateCommandBufferKHR != nullptr && clFinalizeCommandBufferKHR != nullptr
            && clEnqueueCommandBufferKHR != nullptr && clCommandNDRangeKernelKHR != nullptr;
    }
};

struct RunnerOptions {
    std::string input_path;
    std::string json_output_path;
    uint32_t platform_index = 0;
    uint32_t device_index = 0;
    uint32_t warmup_iterations = 0;
    uint32_t iterations = 1;
    bool has_start_dispatch = false;
    bool has_end_dispatch = false;
    uint64_t start_dispatch = 0;
    uint64_t end_dispatch = UINT64_MAX;
    bool use_command_buffer = false;
    bool emulate_command_buffer = false;
    bool allow_lws_fallback = false;
    bool by_dispatch = false;
    bool verbose = false;
};

void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog << " [options] -i <trace.perfetto-trace>\n"
              << "\nOptions:\n"
              << "  -i, --input <file>           Input Perfetto trace file\n"
              << "  -p, --platform <idx>         OpenCL platform index (default: 0)\n"
              << "  -d, --device <idx>           OpenCL device index (default: 0)\n"
              << "  -m, --warmup <count>         Number of warmup iterations (default: 0)\n"
              << "  -n, --iterations <count>     Number of measured iterations (default: 1)\n"
              << "  --start-dispatch <id>        Start replay at dispatch_id <id> (inclusive)\n"
              << "  --end-dispatch <id>          End replay at dispatch_id <id> (inclusive)\n"
              << "  --use-command-buffer         Promote queue dispatches into a cl_command_buffer_khr\n"
              << "  --emulate-command-buffer     Emulate cl_khr_command_buffer via standard queue enqueues\n"
              << "  --allow-lws-fallback         Fallback to NULL local_work_size if device rejects captured LWS\n"
              << "  --by-dispatch                Aggregate profiling results by dispatch_id instead of by target\n"
              << "  --json <file>                Write profiling results to JSON file\n"
              << "  -v, --verbose                Verbose logging of replayed OpenCL calls\n"
              << "  -h, --help                   Show this help message\n";
}

bool parse_args(int argc, char **argv, RunnerOptions &opts)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            opts.input_path = argv[++i];
        } else if ((arg == "-p" || arg == "--platform") && i + 1 < argc) {
            opts.platform_index = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            opts.device_index = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if ((arg == "-m" || arg == "--warmup") && i + 1 < argc) {
            opts.warmup_iterations = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if ((arg == "-n" || arg == "--iterations") && i + 1 < argc) {
            opts.iterations = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--start-dispatch" && i + 1 < argc) {
            opts.has_start_dispatch = true;
            opts.start_dispatch = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--end-dispatch" && i + 1 < argc) {
            opts.has_end_dispatch = true;
            opts.end_dispatch = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--use-command-buffer") {
            opts.use_command_buffer = true;
        } else if (arg == "--emulate-command-buffer") {
            opts.emulate_command_buffer = true;
        } else if (arg == "--allow-lws-fallback") {
            opts.allow_lws_fallback = true;
        } else if (arg == "--by-dispatch") {
            opts.by_dispatch = true;
        } else if ((arg == "--json" || arg == "--json-output") && i + 1 < argc) {
            opts.json_output_path = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else if (arg[0] != '-' && opts.input_path.empty()) {
            opts.input_path = arg;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            return false;
        }
    }
    if (opts.input_path.empty()) {
        std::cerr << "Error: missing input trace file (-i <file>).\n";
        return false;
    }
    return true;
}

struct KernelArgSpec {
    cl_uint arg_index = 0;
    size_t arg_size = 0;
    std::string arg_kind;
    uint64_t mem_id = 0;
    uint64_t sampler_id = 0;
    std::vector<uint8_t> value_bytes;
};

struct MutableConfigSpec {
    uint64_t mutable_cmd_id = 0;
    cl_uint work_dim = 0;
    std::vector<size_t> gwo;
    std::vector<size_t> gws;
    std::vector<size_t> lws;
    std::vector<KernelArgSpec> args;
};

struct EmulatedCommand {
    std::string api_name;
    uint64_t queue_id = 0;
    uint64_t kernel_id = 0;
    uint64_t mutable_cmd_id = 0;
    uint64_t src_mem_id = 0;
    uint64_t dst_mem_id = 0;
    uint64_t mem_id = 0;
    size_t src_offset = 0;
    size_t dst_offset = 0;
    size_t offset = 0;
    size_t size = 0;
    std::vector<size_t> src_origin;
    std::vector<size_t> dst_origin;
    std::vector<size_t> origin;
    std::vector<size_t> region;
    size_t src_row_pitch = 0;
    size_t src_slice_pitch = 0;
    size_t dst_row_pitch = 0;
    size_t dst_slice_pitch = 0;
    std::vector<uint8_t> pattern;
    cl_uint work_dim = 1;
    std::vector<size_t> gwo;
    std::vector<size_t> gws;
    std::vector<size_t> lws;
    std::map<cl_uint, KernelArgSpec> kernel_args_snapshot;
};

struct DispatchMetric {
    uint64_t dispatch_id = 0;
    uint64_t call_id = 0;
    std::string label;
    std::string gws;
    std::string lws;
    std::vector<uint64_t> durations_ns;
};

struct TargetMetric {
    std::string label;
    uint64_t dispatch_count = 0;
    std::vector<uint64_t> durations_ns;
    double total_iter_us = 0.0;
};

struct TraceData {
    std::vector<clkp::ParsedEvent> primary_events;
    std::map<uint64_t, std::vector<clkp::ParsedEvent>> aux_by_call_id;
    std::map<std::string, std::vector<clkp::ParsedEvent>> source_chunks_by_prog;
    std::map<std::string, std::vector<clkp::ParsedEvent>> il_chunks_by_prog;
    std::map<std::string, std::vector<clkp::ParsedEvent>> bin_chunks_by_prog;
};

std::vector<size_t> parse_size_list(const std::string &str)
{
    std::vector<uint64_t> u64 = clkp::string_to_uint64_list(str);
    std::vector<size_t> out(u64.size());
    for (size_t i = 0; i < u64.size(); ++i) {
        out[i] = static_cast<size_t>(u64[i]);
    }
    return out;
}

KernelArgSpec parse_kernel_arg_from_event(const clkp::ParsedEvent &ev)
{
    KernelArgSpec spec;
    spec.arg_index = static_cast<cl_uint>(ev.get_uint("arg_index", 0));
    spec.arg_size = static_cast<size_t>(ev.get_uint("arg_size", 0));
    spec.arg_kind = ev.get_string("arg_kind", "VALUE");
    spec.mem_id = ev.get_uint("mem_id", 0);
    spec.sampler_id = ev.get_uint("sampler_id", 0);
    spec.value_bytes = clkp::hex_to_bytes(ev.get_string("value_hex", ""));
    return spec;
}

bool is_command_recording_call(const std::string &name)
{
    static const std::set<std::string> kCmdRecordingNames = {
        "clCommandBarrierWithWaitListKHR",
        "clCommandCopyBufferKHR",
        "clCommandCopyBufferRectKHR",
        "clCommandCopyBufferToImageKHR",
        "clCommandCopyImageKHR",
        "clCommandCopyImageToBufferKHR",
        "clCommandFillBufferKHR",
        "clCommandFillImageKHR",
        "clCommandNDRangeKernelKHR",
    };
    return kCmdRecordingNames.count(name) > 0;
}

bool is_setup_call(const std::string &name)
{
    static const std::set<std::string> kSetupNames = {
        "clCreateContext",
        "clCreateContextFromType",
        "clCreateCommandQueue",
        "clCreateCommandQueueWithProperties",
        "clCreateProgramWithSource",
        "clCreateProgramWithIL",
        "clCreateProgramWithBinary",
        "clBuildProgram",
        "clCompileProgram",
        "clLinkProgram",
        "clSetProgramSpecializationConstant",
        "clCreateKernel",
        "clCreateKernelsInProgram",
        "clCloneKernel",
        "clCreateBuffer",
        "clCreateBufferWithProperties",
        "clCreateSubBuffer",
        "clCreateImage",
        "clCreateImageWithProperties",
        "clCreateImage2D",
        "clCreateImage3D",
        "clCreateSampler",
        "clCreateSamplerWithProperties",
        "clCreateCommandBufferKHR",
        "clFinalizeCommandBufferKHR",
    };
    return kSetupNames.count(name) > 0 || is_command_recording_call(name);
}

bool is_execution_call(const std::string &name)
{
    static const std::set<std::string> kExecNames = {
        "clSetKernelArg",
        "clEnqueueNDRangeKernel",
        "clEnqueueTask",
        "clEnqueueReadBuffer",
        "clEnqueueWriteBuffer",
        "clEnqueueReadBufferRect",
        "clEnqueueWriteBufferRect",
        "clEnqueueCopyBuffer",
        "clEnqueueCopyBufferRect",
        "clEnqueueFillBuffer",
        "clEnqueueReadImage",
        "clEnqueueWriteImage",
        "clEnqueueCopyImage",
        "clEnqueueFillImage",
        "clEnqueueCopyImageToBuffer",
        "clEnqueueCopyBufferToImage",
        "clEnqueueMapBuffer",
        "clEnqueueMapImage",
        "clEnqueueUnmapMemObject",
        "clEnqueueMigrateMemObjects",
        "clEnqueueMarkerWithWaitList",
        "clEnqueueBarrierWithWaitList",
        "clEnqueueMarker",
        "clEnqueueBarrier",
        "clEnqueueWaitForEvents",
        "clFlush",
        "clFinish",
        "clWaitForEvents",
        "clCreateUserEvent",
        "clSetUserEventStatus",
        "clEnqueueCommandBufferKHR",
        "clUpdateMutableCommandsKHR",
    };
    return kExecNames.count(name) > 0;
}

bool is_internal_or_gpu_slice(const std::string &name)
{
    return name == "writeProgramOnDisk" || name == "clkp-callback" || name == "clkp_wait"
        || name.rfind("clkp_p", 0) == 0 || name.rfind("clkp_cmdbuf_", 0) == 0;
}

void classify_instant_event(clkp::ParsedEvent &&ev, TraceData &td)
{
    if (ev.name == "clCreateProgramWithSource-args") {
        td.source_chunks_by_prog[ev.get_string("program")].push_back(std::move(ev));
    } else if (ev.name == "clCreateProgramWithIL-args") {
        td.il_chunks_by_prog[ev.get_string("program")].push_back(std::move(ev));
    } else if (ev.name == "clCreateProgramWithBinary-args") {
        td.bin_chunks_by_prog[ev.get_string("program")].push_back(std::move(ev));
    } else if (ev.name == "clCreateKernelsInProgram-kernel" || ev.name == "clUpdateMutableCommandsKHR-config"
        || ev.name == "clUpdateMutableCommandsKHR-arg") {
        td.aux_by_call_id[ev.get_uint("call_id", 0)].push_back(std::move(ev));
    }
}

void parse_trace_into_replay_stream(const std::vector<uint8_t> &trace_bytes, TraceData &td)
{
    clkp::TraceIncrementState inc_state;
    uint64_t synthetic_call_counter = 1;
    perfetto::protos::pbzero::Trace_Decoder trace_dec(trace_bytes.data(), trace_bytes.size());
    size_t pkt_idx = 0;

    for (auto pkt_it = trace_dec.packet(); pkt_it; ++pkt_it, ++pkt_idx) {
        protozero::ConstBytes pkt_bytes = *pkt_it;
        perfetto::protos::pbzero::TracePacket_Decoder pkt_dec(pkt_bytes.data, pkt_bytes.size);
        inc_state.update_from_packet(pkt_dec);

        clkp::ParsedEvent ev;
        if (!inc_state.decode_track_event(pkt_dec, pkt_idx, ev) || !ev.has_category(CLKP_PERFETTO_CATEGORY)) {
            continue;
        }

        if (ev.event_type == 3) {
            classify_instant_event(std::move(ev), td);
        } else if (ev.event_type == 1 && !is_internal_or_gpu_slice(ev.name)) {
            if (!ev.has_key("call_id")) {
                clkp::ParsedAnnotation ann;
                ann.name = "call_id";
                ann.type = clkp::ParsedAnnotation::ValueType::kUint;
                ann.uint_val = synthetic_call_counter++;
                ev.annotations.push_back(std::move(ann));
            } else {
                synthetic_call_counter = std::max(synthetic_call_counter, ev.get_uint("call_id") + 1);
            }
            td.primary_events.push_back(std::move(ev));
        }
    }

    std::sort(
        td.primary_events.begin(), td.primary_events.end(), [](const clkp::ParsedEvent &a, const clkp::ParsedEvent &b) {
            return a.get_uint("call_id") < b.get_uint("call_id");
        });
}

bool select_platform_and_device(const RunnerOptions &opts, cl_platform_id &platform, cl_device_id &device,
    std::string &platform_name, std::string &device_name)
{
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "Error: no OpenCL platforms found.\n";
        return false;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (opts.platform_index >= num_platforms) {
        std::cerr << "Error: platform index " << opts.platform_index << " out of range (found " << num_platforms
                  << ").\n";
        return false;
    }
    platform = platforms[opts.platform_index];

    cl_uint num_devices = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices) != CL_SUCCESS || num_devices == 0) {
        std::cerr << "Error: no OpenCL devices found on platform " << opts.platform_index << ".\n";
        return false;
    }
    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
    if (opts.device_index >= num_devices) {
        std::cerr << "Error: device index " << opts.device_index << " out of range (found " << num_devices << ").\n";
        return false;
    }
    device = devices[opts.device_index];

    char pname[256] = { 0 };
    char dname[256] = { 0 };
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pname) - 1, pname, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname) - 1, dname, nullptr);
    platform_name = pname;
    device_name = dname;

    std::cout << "[clkp-runner] Platform: " << platform_name << "\n";
    std::cout << "[clkp-runner] Device:   " << device_name << "\n";
    return true;
}

bool resolve_dispatch_window(const RunnerOptions &opts, const std::vector<clkp::ParsedEvent> &primary_events,
    uint64_t &window_start_call, uint64_t &window_end_call)
{
    window_start_call = 0;
    window_end_call = UINT64_MAX;
    if (!opts.has_start_dispatch && !opts.has_end_dispatch) {
        return true;
    }

    uint64_t min_c = UINT64_MAX;
    uint64_t max_c = 0;
    for (const auto &ev : primary_events) {
        if (ev.has_key("dispatch_id")) {
            uint64_t did = ev.get_uint("dispatch_id");
            if (did >= opts.start_dispatch && did <= opts.end_dispatch) {
                uint64_t cid = ev.get_uint("call_id");
                min_c = std::min(min_c, cid);
                max_c = std::max(max_c, cid);
            }
        }
    }
    if (min_c == UINT64_MAX) {
        std::cerr << "Error: no dispatches found in requested dispatch range.\n";
        return false;
    }
    window_start_call = min_c;
    window_end_call = max_c;
    return true;
}

class ReplayEngine {
public:
    ReplayEngine(const RunnerOptions &opts, cl_platform_id platform, cl_device_id device, TraceData &td)
        : opts_(opts)
        , platform_(platform)
        , device_(device)
        , td_(td)
        , host_scratch_buffer_(64 * 1024, 0)
    {
        ext_.load(platform_);
    }

    ~ReplayEngine() { release_resources(); }

    bool run_setup_phase(
        uint64_t window_start_call, uint64_t window_end_call, std::vector<const clkp::ParsedEvent *> &hot_loop_events)
    {
        for (const auto &ev : td_.primary_events) {
            uint64_t cid = ev.get_uint("call_id", 0);
            if (cid > window_end_call) {
                continue;
            }
            if (opts_.verbose && is_setup_call(ev.name)) {
                std::cout << "[Setup] call_id=" << cid << " " << ev.name << "\n";
            }
            if (!dispatch_setup_event(ev, cid, window_start_call, window_end_call, hot_loop_events)) {
                return false;
            }
        }

        for (auto &[qid, q] : queues_) {
            clFinish(q);
        }

        if (opts_.use_command_buffer && !build_promoted_command_buffer(hot_loop_events)) {
            return false;
        }
        return true;
    }

    bool run_iterations(const std::vector<const clkp::ParsedEvent *> &hot_loop_events,
        std::map<uint64_t, DispatchMetric> &metrics, std::vector<uint64_t> &iter_total_ns)
    {
        uint32_t total_iters = opts_.warmup_iterations + opts_.iterations;
        iter_total_ns.assign(opts_.iterations, 0);

        for (uint32_t iter = 0; iter < total_iters; ++iter) {
            bool is_measured = (iter >= opts_.warmup_iterations);
            uint32_t measured_idx = is_measured ? (iter - opts_.warmup_iterations) : 0;
            if (!run_single_iteration(hot_loop_events, is_measured, measured_idx, metrics, iter_total_ns)) {
                return false;
            }
        }
        return true;
    }

private:
    const RunnerOptions &opts_;
    cl_platform_id platform_;
    cl_device_id device_;
    ExtensionFunctions ext_;
    TraceData &td_;

    std::map<uint64_t, cl_context> contexts_;
    std::map<uint64_t, cl_command_queue> queues_;
    std::map<uint64_t, cl_program> programs_;
    std::map<std::string, cl_program> programs_by_name_;
    std::map<uint64_t, cl_kernel> kernels_;
    std::map<std::pair<std::string, std::string>, cl_kernel> kernels_by_prog_and_name_;
    std::map<uint64_t, cl_mem> mems_;
    std::map<uint64_t, size_t> mem_sizes_;
    std::map<uint64_t, cl_sampler> samplers_;
    std::map<uint64_t, cl_command_buffer_khr> cmdbufs_;
    std::map<uint64_t, uint64_t> cmdbuf_default_queue_;
    std::map<uint64_t, cl_mutable_command_khr> mutable_cmds_;
    std::map<uint64_t, std::map<cl_uint, cl_sync_point_khr>> sync_points_;
    std::map<uint64_t, std::map<cl_uint, KernelArgSpec>> current_kernel_args_;
    std::map<uint64_t, std::vector<EmulatedCommand>> emulated_cmdbufs_;
    std::map<uint64_t, void *> active_maps_;
    std::vector<uint8_t> host_scratch_buffer_;

    cl_command_buffer_khr promoted_cmdbuf_ = nullptr;
    cl_command_queue promoted_queue_ = nullptr;

    cl_context get_default_context()
    {
        if (!contexts_.empty()) {
            return contexts_.begin()->second;
        }
        cl_int err = CL_SUCCESS;
        cl_context ctx = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        contexts_[1] = ctx;
        return ctx;
    }

    cl_context get_context_or_default(uint64_t ctx_id)
    {
        return contexts_.count(ctx_id) ? contexts_[ctx_id] : get_default_context();
    }

    cl_command_queue get_default_queue()
    {
        if (!queues_.empty()) {
            return queues_.begin()->second;
        }
        cl_context ctx = get_default_context();
        cl_int err = CL_SUCCESS;
        const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        cl_command_queue q = clCreateCommandQueueWithProperties(ctx, device_, props, &err);
        queues_[1] = q;
        return q;
    }

    cl_command_queue get_queue_or_default(uint64_t q_id)
    {
        return queues_.count(q_id) ? queues_[q_id] : get_default_queue();
    }

    cl_program resolve_program(uint64_t prog_id, const std::string &prog_str)
    {
        return programs_.count(prog_id) ? programs_[prog_id] : programs_by_name_[prog_str];
    }

    cl_kernel resolve_kernel(uint64_t kern_id, const std::string &prog_str, const std::string &kname)
    {
        return kernels_.count(kern_id) ? kernels_[kern_id] : kernels_by_prog_and_name_[{ prog_str, kname }];
    }

    cl_int apply_kernel_arg(cl_kernel k, const KernelArgSpec &spec)
    {
        if (!k) {
            return CL_INVALID_KERNEL;
        }
        if (spec.arg_kind == "LOCAL") {
            return clSetKernelArg(k, spec.arg_index, spec.arg_size, nullptr);
        }
        if (spec.arg_kind == "MEM") {
            cl_mem m = mems_.count(spec.mem_id) ? mems_[spec.mem_id] : nullptr;
            return clSetKernelArg(k, spec.arg_index, sizeof(cl_mem), &m);
        }
        if (spec.arg_kind == "SAMPLER") {
            cl_sampler s = samplers_.count(spec.sampler_id) ? samplers_[spec.sampler_id] : nullptr;
            return clSetKernelArg(k, spec.arg_index, sizeof(cl_sampler), &s);
        }
        const void *ptr = spec.value_bytes.empty() ? nullptr : spec.value_bytes.data();
        return clSetKernelArg(k, spec.arg_index, spec.arg_size, ptr);
    }

    bool has_explicit_build_event(uint64_t prog_id, const std::string &prog_str) const
    {
        for (const auto &other : td_.primary_events) {
            bool is_build = (other.name == "clBuildProgram" || other.name == "clCompileProgram");
            bool matches_prog = (other.get_uint("program_id") == prog_id || other.get_string("program") == prog_str);
            if (is_build && matches_prog) {
                return true;
            }
        }
        return false;
    }

    void register_program(uint64_t prog_id, const std::string &prog_str, cl_program prog, bool build_if_legacy)
    {
        if (prog_id != 0) {
            programs_[prog_id] = prog;
        }
        if (!prog_str.empty()) {
            programs_by_name_[prog_str] = prog;
        }
        if (build_if_legacy && !has_explicit_build_event(prog_id, prog_str)) {
            clBuildProgram(prog, 1, &device_, nullptr, nullptr, nullptr);
        }
    }

    bool setup_create_context(const clkp::ParsedEvent &ev)
    {
        uint64_t ctx_id = ev.get_uint("context_id", 1);
        cl_int err = CL_SUCCESS;
        cl_context ctx = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS || !ctx) {
            std::cerr << "Error: clCreateContext failed (" << err << ")\n";
            return false;
        }
        contexts_[ctx_id] = ctx;
        return true;
    }

    bool setup_create_command_queue(const clkp::ParsedEvent &ev)
    {
        uint64_t ctx_id = ev.get_uint("context_id", 1);
        uint64_t q_id = ev.get_uint("queue_id", 1);
        uint64_t props_bits = ev.get_uint("properties", 0) | CL_QUEUE_PROFILING_ENABLE;
        cl_context ctx = get_context_or_default(ctx_id);
        const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(props_bits), 0 };
        cl_int err = CL_SUCCESS;
        cl_command_queue q = clCreateCommandQueueWithProperties(ctx, device_, props, &err);
        if (err != CL_SUCCESS || !q) {
            std::cerr << "Error: clCreateCommandQueue failed (" << err << ")\n";
            return false;
        }
        queues_[q_id] = q;
        return true;
    }

    bool setup_create_program_with_source(const clkp::ParsedEvent &ev)
    {
        uint64_t ctx_id = ev.get_uint("context_id", 1);
        uint64_t prog_id = ev.get_uint("program_id", 0);
        std::string prog_str = ev.get_string("program", "");
        cl_context ctx = get_context_or_default(ctx_id);

        std::map<uint64_t, std::map<uint64_t, std::string>> strings_map;
        auto it = td_.source_chunks_by_prog.find(prog_str);
        if (it != td_.source_chunks_by_prog.end()) {
            for (const auto &chunk_ev : it->second) {
                strings_map[chunk_ev.get_uint("string_index", 0)][chunk_ev.get_uint("chunk_index", 0)]
                    = chunk_ev.get_string("string", "");
            }
        }

        std::vector<std::string> full_strings;
        for (const auto &[s_idx, chunks] : strings_map) {
            std::string combined;
            for (const auto &[c_idx, part] : chunks) {
                combined += part;
            }
            full_strings.push_back(std::move(combined));
        }

        std::vector<const char *> c_strs;
        std::vector<size_t> lengths;
        for (const auto &s : full_strings) {
            c_strs.push_back(s.c_str());
            lengths.push_back(s.size());
        }

        cl_int err = CL_SUCCESS;
        cl_program prog
            = clCreateProgramWithSource(ctx, static_cast<cl_uint>(c_strs.size()), c_strs.data(), lengths.data(), &err);
        if (err != CL_SUCCESS || !prog) {
            std::cerr << "Error: clCreateProgramWithSource failed (" << err << ") for " << prog_str << "\n";
            return false;
        }
        register_program(prog_id, prog_str, prog, true);
        return true;
    }

    bool setup_create_program_with_il(const clkp::ParsedEvent &ev)
    {
        uint64_t ctx_id = ev.get_uint("context_id", 1);
        uint64_t prog_id = ev.get_uint("program_id", 0);
        std::string prog_str = ev.get_string("program", "");
        cl_context ctx = get_context_or_default(ctx_id);

        std::map<uint64_t, std::string> chunks;
        auto it = td_.il_chunks_by_prog.find(prog_str);
        if (it != td_.il_chunks_by_prog.end()) {
            for (const auto &chunk_ev : it->second) {
                if (chunk_ev.has_key("il_hex")) {
                    chunks[chunk_ev.get_uint("chunk_index", 0)] = chunk_ev.get_string("il_hex", "");
                }
            }
        }
        std::string full_hex;
        for (const auto &[c_idx, part] : chunks) {
            full_hex += part;
        }
        std::vector<uint8_t> il_bytes = clkp::hex_to_bytes(full_hex);

        cl_int err = CL_SUCCESS;
        cl_program prog = clCreateProgramWithIL(ctx, il_bytes.data(), il_bytes.size(), &err);
        if (err != CL_SUCCESS || !prog) {
            std::cerr << "Error: clCreateProgramWithIL failed (" << err << ") for " << prog_str << "\n";
            return false;
        }
        register_program(prog_id, prog_str, prog, true);
        return true;
    }

    bool setup_create_program_with_binary(const clkp::ParsedEvent &ev)
    {
        uint64_t ctx_id = ev.get_uint("context_id", 1);
        uint64_t prog_id = ev.get_uint("program_id", 0);
        std::string prog_str = ev.get_string("program", "");
        cl_context ctx = get_context_or_default(ctx_id);

        std::map<uint64_t, std::string> chunks;
        auto it = td_.bin_chunks_by_prog.find(prog_str);
        if (it != td_.bin_chunks_by_prog.end()) {
            for (const auto &chunk_ev : it->second) {
                if (chunk_ev.get_uint("device_index", 0) == 0) {
                    chunks[chunk_ev.get_uint("chunk_index", 0)] = chunk_ev.get_string("binary_hex", "");
                }
            }
        }
        std::string full_hex;
        for (const auto &[c_idx, part] : chunks) {
            full_hex += part;
        }
        std::vector<uint8_t> bin_bytes = clkp::hex_to_bytes(full_hex);
        size_t len = bin_bytes.size();
        const unsigned char *bin_ptr = bin_bytes.data();
        cl_int bin_status = CL_SUCCESS;
        cl_int err = CL_SUCCESS;
        cl_program prog = clCreateProgramWithBinary(ctx, 1, &device_, &len, &bin_ptr, &bin_status, &err);
        if (err != CL_SUCCESS || !prog) {
            std::cerr << "Error: clCreateProgramWithBinary failed (" << err << ") for " << prog_str << "\n";
            return false;
        }
        register_program(prog_id, prog_str, prog, false);
        return true;
    }

    bool setup_specialization_constant(const clkp::ParsedEvent &ev)
    {
        cl_program prog = resolve_program(ev.get_uint("program_id", 0), ev.get_string("program"));
        if (prog) {
            cl_uint spec_id = static_cast<cl_uint>(ev.get_uint("spec_id", 0));
            std::vector<uint8_t> val = clkp::hex_to_bytes(ev.get_string("value_hex", ""));
            clSetProgramSpecializationConstant(prog, spec_id, val.size(), val.data());
        }
        return true;
    }

    bool setup_build_program(const clkp::ParsedEvent &ev)
    {
        std::string prog_str = ev.get_string("program", "");
        cl_program prog = resolve_program(ev.get_uint("program_id", 0), prog_str);
        std::string options = ev.get_string("options", "");
        cl_int err = clBuildProgram(prog, 1, &device_, options.empty() ? nullptr : options.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(prog, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string build_log(log_size, '\0');
            if (log_size > 0) {
                clGetProgramBuildInfo(prog, device_, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
            }
            std::cerr << "Error: clBuildProgram failed (" << err << ") for " << prog_str << ":\n" << build_log << "\n";
            return false;
        }
        return true;
    }

    bool setup_compile_program(const clkp::ParsedEvent &ev)
    {
        cl_program prog = resolve_program(ev.get_uint("program_id", 0), ev.get_string("program"));
        std::string options = ev.get_string("options", "");
        std::vector<uint64_t> hdr_ids = clkp::string_to_uint64_list(ev.get_string("header_program_ids"));
        std::vector<std::string> hdr_names = clkp::joined_to_string_list(ev.get_string("header_include_names"), ';');
        size_t count = std::min(hdr_ids.size(), hdr_names.size());
        std::vector<cl_program> hdr_progs(count);
        std::vector<const char *> hdr_cstrs(count);
        for (size_t i = 0; i < count; ++i) {
            hdr_progs[i] = programs_[hdr_ids[i]];
            hdr_cstrs[i] = hdr_names[i].c_str();
        }
        cl_int err = clCompileProgram(prog, 1, &device_, options.empty() ? nullptr : options.c_str(),
            static_cast<cl_uint>(count), count ? hdr_progs.data() : nullptr, count ? hdr_cstrs.data() : nullptr,
            nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: clCompileProgram failed (" << err << ")\n";
            return false;
        }
        return true;
    }

    bool setup_link_program(const clkp::ParsedEvent &ev)
    {
        cl_context ctx = get_context_or_default(ev.get_uint("context_id", 1));
        uint64_t prog_id = ev.get_uint("program_id", 0);
        std::string prog_str = ev.get_string("program", "");
        std::string options = ev.get_string("options", "");
        std::vector<uint64_t> in_ids = clkp::string_to_uint64_list(ev.get_string("input_program_ids"));
        std::vector<cl_program> in_progs;
        for (uint64_t id : in_ids) {
            if (programs_.count(id)) {
                in_progs.push_back(programs_[id]);
            }
        }
        cl_int err = CL_SUCCESS;
        cl_program linked = clLinkProgram(ctx, 1, &device_, options.empty() ? nullptr : options.c_str(),
            static_cast<cl_uint>(in_progs.size()), in_progs.data(), nullptr, nullptr, &err);
        if (err != CL_SUCCESS || !linked) {
            std::cerr << "Error: clLinkProgram failed (" << err << ")\n";
            return false;
        }
        register_program(prog_id, prog_str, linked, false);
        return true;
    }

    bool setup_create_kernel(const clkp::ParsedEvent &ev)
    {
        std::string prog_str = ev.get_string("program", "");
        uint64_t kern_id = ev.get_uint("kernel_id", 0);
        std::string kname = ev.get_string("kernel_name", "");
        cl_program prog = resolve_program(ev.get_uint("program_id", 0), prog_str);
        cl_int err = CL_SUCCESS;
        cl_kernel k = clCreateKernel(prog, kname.c_str(), &err);
        if (err != CL_SUCCESS || !k) {
            std::cerr << "Error: clCreateKernel('" << kname << "') failed (" << err << ")\n";
            return false;
        }
        kernels_[kern_id] = k;
        kernels_by_prog_and_name_[{ prog_str, kname }] = k;
        return true;
    }

    bool setup_create_kernels_in_program(const clkp::ParsedEvent &ev, uint64_t cid)
    {
        std::string prog_str = ev.get_string("program", "");
        cl_program prog = resolve_program(ev.get_uint("program_id", 0), prog_str);
        for (const auto &kev : td_.aux_by_call_id[cid]) {
            if (kev.name != "clCreateKernelsInProgram-kernel") {
                continue;
            }
            uint64_t kern_id = kev.get_uint("kernel_id", 0);
            std::string kname = kev.get_string("kernel_name", "");
            cl_int err = CL_SUCCESS;
            cl_kernel k = clCreateKernel(prog, kname.c_str(), &err);
            if (err == CL_SUCCESS && k) {
                kernels_[kern_id] = k;
                kernels_by_prog_and_name_[{ prog_str, kname }] = k;
            }
        }
        return true;
    }

    bool setup_clone_kernel(const clkp::ParsedEvent &ev)
    {
        uint64_t src_id = ev.get_uint("src_kernel_id", 0);
        uint64_t dst_id = ev.get_uint("kernel_id", 0);
        if (kernels_.count(src_id)) {
            cl_int err = CL_SUCCESS;
            cl_kernel cloned = clCloneKernel(kernels_[src_id], &err);
            if (err == CL_SUCCESS && cloned) {
                kernels_[dst_id] = cloned;
                current_kernel_args_[dst_id] = current_kernel_args_[src_id];
            }
        }
        return true;
    }

    bool setup_create_buffer(const clkp::ParsedEvent &ev)
    {
        cl_context ctx = get_context_or_default(ev.get_uint("context_id", 1));
        uint64_t mem_id = ev.get_uint("mem_id", 0);
        cl_mem_flags flags = static_cast<cl_mem_flags>(ev.get_uint("flags", CL_MEM_READ_WRITE));
        flags &= ~(CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR);
        if ((flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY)) == 0) {
            flags |= CL_MEM_READ_WRITE;
        }
        size_t size = std::max<size_t>(static_cast<size_t>(ev.get_uint("size", 4096)), 4);
        cl_int err = CL_SUCCESS;
        cl_mem buf = clCreateBuffer(ctx, flags, size, nullptr, &err);
        if (err != CL_SUCCESS || !buf) {
            std::cerr << "Error: clCreateBuffer(size=" << size << ") failed (" << err << ")\n";
            return false;
        }
        mems_[mem_id] = buf;
        mem_sizes_[mem_id] = size;

        uint32_t zero_pattern = 0;
        clEnqueueFillBuffer(
            get_default_queue(), buf, &zero_pattern, sizeof(zero_pattern), 0, size, 0, nullptr, nullptr);
        return true;
    }

    bool setup_create_sub_buffer(const clkp::ParsedEvent &ev)
    {
        uint64_t parent_id = ev.get_uint("parent_mem_id", 0);
        uint64_t mem_id = ev.get_uint("mem_id", 0);
        if (!mems_.count(parent_id)) {
            return true;
        }
        cl_mem_flags flags = static_cast<cl_mem_flags>(ev.get_uint("flags", CL_MEM_READ_WRITE));
        flags &= ~(CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR);
        cl_buffer_region region = {
            static_cast<size_t>(ev.get_uint("origin", 0)),
            static_cast<size_t>(ev.get_uint("size", 0)),
        };
        cl_int err = CL_SUCCESS;
        cl_mem sub = clCreateSubBuffer(mems_[parent_id], flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        if (err == CL_SUCCESS && sub) {
            mems_[mem_id] = sub;
            mem_sizes_[mem_id] = region.size;
        }
        return true;
    }

    bool setup_create_image(const clkp::ParsedEvent &ev)
    {
        cl_context ctx = get_context_or_default(ev.get_uint("context_id", 1));
        uint64_t mem_id = ev.get_uint("mem_id", 0);
        cl_mem_flags flags = static_cast<cl_mem_flags>(ev.get_uint("flags", CL_MEM_READ_WRITE));
        flags &= ~(CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR);
        cl_image_format fmt = {
            static_cast<cl_channel_order>(ev.get_uint("channel_order", CL_RGBA)),
            static_cast<cl_channel_type>(ev.get_uint("channel_data_type", CL_FLOAT)),
        };
        cl_image_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.image_type = static_cast<cl_mem_object_type>(ev.get_uint("image_type", CL_MEM_OBJECT_IMAGE2D));
        desc.image_width = static_cast<size_t>(ev.get_uint("width", 1));
        desc.image_height = static_cast<size_t>(ev.get_uint("height", 1));
        desc.image_depth = static_cast<size_t>(ev.get_uint("depth", 1));
        desc.image_array_size = static_cast<size_t>(ev.get_uint("array_size", 0));
        desc.num_mip_levels = static_cast<cl_uint>(ev.get_uint("num_mip_levels", 0));
        desc.num_samples = static_cast<cl_uint>(ev.get_uint("num_samples", 0));
        uint64_t parent_id = ev.get_uint("parent_mem_id", 0);
        if (parent_id != 0 && mems_.count(parent_id)) {
            desc.mem_object = mems_[parent_id];
            desc.image_row_pitch = static_cast<size_t>(ev.get_uint("row_pitch", 0));
            desc.image_slice_pitch = static_cast<size_t>(ev.get_uint("slice_pitch", 0));
        }

        cl_int err = CL_SUCCESS;
        cl_mem img = clCreateImage(ctx, flags, &fmt, &desc, nullptr, &err);
        if (err != CL_SUCCESS || !img) {
            std::cerr << "Error: clCreateImage failed (" << err << ")\n";
            return false;
        }
        mems_[mem_id] = img;
        return true;
    }

    bool setup_create_sampler(const clkp::ParsedEvent &ev)
    {
        cl_context ctx = get_context_or_default(ev.get_uint("context_id", 1));
        uint64_t samp_id = ev.get_uint("sampler_id", 0);
        cl_bool norm = static_cast<cl_bool>(ev.get_uint("normalized_coords", CL_TRUE));
        cl_addressing_mode addr = static_cast<cl_addressing_mode>(ev.get_uint("addressing_mode", CL_ADDRESS_CLAMP));
        cl_filter_mode filt = static_cast<cl_filter_mode>(ev.get_uint("filter_mode", CL_FILTER_NEAREST));
        cl_int err = CL_SUCCESS;
        cl_sampler s = clCreateSampler(ctx, norm, addr, filt, &err);
        if (err == CL_SUCCESS && s) {
            samplers_[samp_id] = s;
        }
        return true;
    }

    bool setup_create_command_buffer(const clkp::ParsedEvent &ev)
    {
        uint64_t cb_id = ev.get_uint("cmdbuf_id", 0);
        std::vector<uint64_t> q_ids = clkp::string_to_uint64_list(ev.get_string("queue_ids"));
        std::vector<cl_command_queue> q_vec;
        for (uint64_t qid : q_ids) {
            if (queues_.count(qid)) {
                q_vec.push_back(queues_[qid]);
            }
        }
        if (q_vec.empty()) {
            q_vec.push_back(get_default_queue());
        }
        cmdbuf_default_queue_[cb_id] = q_ids.empty() ? 1 : q_ids[0];

        if (opts_.emulate_command_buffer || !ext_.supports_command_buffer()) {
            emulated_cmdbufs_[cb_id] = {};
            return true;
        }

        std::vector<uint64_t> raw_props = clkp::string_to_uint64_list(ev.get_string("properties"));
        std::vector<cl_command_buffer_properties_khr> props(raw_props.begin(), raw_props.end());
        if (!props.empty()) {
            props.push_back(0);
        }
        cl_int err = CL_SUCCESS;
        cl_command_buffer_khr cb = ext_.clCreateCommandBufferKHR(
            static_cast<cl_uint>(q_vec.size()), q_vec.data(), props.empty() ? nullptr : props.data(), &err);
        if (err != CL_SUCCESS || !cb) {
            std::cerr << "Error: clCreateCommandBufferKHR failed (" << err << ")\n";
            return false;
        }
        cmdbufs_[cb_id] = cb;
        return true;
    }

    cl_int invoke_native_command_recording(const clkp::ParsedEvent &ev, cl_command_buffer_khr cb,
        cl_command_queue explicit_q, const std::vector<cl_sync_point_khr> &wait_sps, cl_sync_point_khr *out_sp,
        cl_mutable_command_khr *out_mut)
    {
        const std::string &name = ev.name;
        cl_uint num_wait = static_cast<cl_uint>(wait_sps.size());
        const cl_sync_point_khr *wait_ptr = wait_sps.empty() ? nullptr : wait_sps.data();

        if (name == "clCommandNDRangeKernelKHR") {
            cl_kernel k = kernels_[ev.get_uint("kernel_id", 0)];
            cl_uint work_dim = static_cast<cl_uint>(ev.get_uint("work_dim", 1));
            auto gwo = parse_size_list(ev.get_string("gwo"));
            auto gws = parse_size_list(ev.get_string("gws"));
            auto lws = parse_size_list(ev.get_string("lws"));
            auto raw_props = clkp::string_to_uint64_list(ev.get_string("properties"));
            std::vector<cl_command_properties_khr> props(raw_props.begin(), raw_props.end());
            if (!props.empty()) {
                props.push_back(0);
            }
            const cl_command_properties_khr *props_ptr = props.empty() ? nullptr : props.data();
            cl_int err = ext_.clCommandNDRangeKernelKHR(cb, explicit_q, props_ptr, k, work_dim,
                gwo.empty() ? nullptr : gwo.data(), gws.data(), lws.empty() ? nullptr : lws.data(), num_wait, wait_ptr,
                out_sp, out_mut);
            if (err == CL_INVALID_WORK_GROUP_SIZE && opts_.allow_lws_fallback) {
                err = ext_.clCommandNDRangeKernelKHR(cb, explicit_q, props_ptr, k, work_dim,
                    gwo.empty() ? nullptr : gwo.data(), gws.data(), nullptr, num_wait, wait_ptr, out_sp, out_mut);
            }
            return err;
        }
        if (name == "clCommandCopyBufferKHR" && ext_.clCommandCopyBufferKHR) {
            return ext_.clCommandCopyBufferKHR(cb, explicit_q, nullptr, mems_[ev.get_uint("src_mem_id")],
                mems_[ev.get_uint("dst_mem_id")], static_cast<size_t>(ev.get_uint("src_offset")),
                static_cast<size_t>(ev.get_uint("dst_offset")), static_cast<size_t>(ev.get_uint("size")), num_wait,
                wait_ptr, out_sp, out_mut);
        }
        if (name == "clCommandCopyBufferRectKHR" && ext_.clCommandCopyBufferRectKHR) {
            auto so = parse_size_list(ev.get_string("src_origin"));
            auto do_ = parse_size_list(ev.get_string("dst_origin"));
            auto reg = parse_size_list(ev.get_string("region"));
            return ext_.clCommandCopyBufferRectKHR(cb, explicit_q, nullptr, mems_[ev.get_uint("src_mem_id")],
                mems_[ev.get_uint("dst_mem_id")], so.data(), do_.data(), reg.data(),
                static_cast<size_t>(ev.get_uint("src_row_pitch")), static_cast<size_t>(ev.get_uint("src_slice_pitch")),
                static_cast<size_t>(ev.get_uint("dst_row_pitch")), static_cast<size_t>(ev.get_uint("dst_slice_pitch")),
                num_wait, wait_ptr, out_sp, out_mut);
        }
        if (name == "clCommandCopyBufferToImageKHR" && ext_.clCommandCopyBufferToImageKHR) {
            auto do_ = parse_size_list(ev.get_string("dst_origin"));
            auto reg = parse_size_list(ev.get_string("region"));
            return ext_.clCommandCopyBufferToImageKHR(cb, explicit_q, nullptr, mems_[ev.get_uint("src_mem_id")],
                mems_[ev.get_uint("dst_mem_id")], static_cast<size_t>(ev.get_uint("src_offset")), do_.data(),
                reg.data(), num_wait, wait_ptr, out_sp, out_mut);
        }
        if (name == "clCommandCopyImageKHR" && ext_.clCommandCopyImageKHR) {
            auto so = parse_size_list(ev.get_string("src_origin"));
            auto do_ = parse_size_list(ev.get_string("dst_origin"));
            auto reg = parse_size_list(ev.get_string("region"));
            return ext_.clCommandCopyImageKHR(cb, explicit_q, nullptr, mems_[ev.get_uint("src_mem_id")],
                mems_[ev.get_uint("dst_mem_id")], so.data(), do_.data(), reg.data(), num_wait, wait_ptr, out_sp,
                out_mut);
        }
        if (name == "clCommandCopyImageToBufferKHR" && ext_.clCommandCopyImageToBufferKHR) {
            auto so = parse_size_list(ev.get_string("src_origin"));
            auto reg = parse_size_list(ev.get_string("region"));
            return ext_.clCommandCopyImageToBufferKHR(cb, explicit_q, nullptr, mems_[ev.get_uint("src_mem_id")],
                mems_[ev.get_uint("dst_mem_id")], so.data(), reg.data(), static_cast<size_t>(ev.get_uint("dst_offset")),
                num_wait, wait_ptr, out_sp, out_mut);
        }
        if (name == "clCommandFillBufferKHR" && ext_.clCommandFillBufferKHR) {
            auto pat = clkp::hex_to_bytes(ev.get_string("pattern_hex"));
            return ext_.clCommandFillBufferKHR(cb, explicit_q, nullptr, mems_[ev.get_uint("mem_id")], pat.data(),
                pat.size(), static_cast<size_t>(ev.get_uint("offset")), static_cast<size_t>(ev.get_uint("size")),
                num_wait, wait_ptr, out_sp, out_mut);
        }
        if (name == "clCommandFillImageKHR" && ext_.clCommandFillImageKHR) {
            auto col = clkp::hex_to_bytes(ev.get_string("fill_color_hex"));
            auto orig = parse_size_list(ev.get_string("origin"));
            auto reg = parse_size_list(ev.get_string("region"));
            return ext_.clCommandFillImageKHR(cb, explicit_q, nullptr, mems_[ev.get_uint("mem_id")], col.data(),
                orig.data(), reg.data(), num_wait, wait_ptr, out_sp, out_mut);
        }
        if (name == "clCommandBarrierWithWaitListKHR" && ext_.clCommandBarrierWithWaitListKHR) {
            return ext_.clCommandBarrierWithWaitListKHR(cb, explicit_q, nullptr, num_wait, wait_ptr, out_sp, out_mut);
        }
        return CL_SUCCESS;
    }

    void record_emulated_command(const clkp::ParsedEvent &ev, uint64_t cb_id, uint64_t q_id, uint64_t mut_id)
    {
        EmulatedCommand ecmd;
        ecmd.api_name = ev.name;
        ecmd.queue_id = q_id != 0 ? q_id : cmdbuf_default_queue_[cb_id];
        ecmd.kernel_id = ev.get_uint("kernel_id", 0);
        ecmd.mutable_cmd_id = mut_id;
        ecmd.src_mem_id = ev.get_uint("src_mem_id", 0);
        ecmd.dst_mem_id = ev.get_uint("dst_mem_id", 0);
        ecmd.mem_id = ev.get_uint("mem_id", 0);
        ecmd.src_offset = static_cast<size_t>(ev.get_uint("src_offset", 0));
        ecmd.dst_offset = static_cast<size_t>(ev.get_uint("dst_offset", 0));
        ecmd.offset = static_cast<size_t>(ev.get_uint("offset", 0));
        ecmd.size = static_cast<size_t>(ev.get_uint("size", 0));
        ecmd.src_origin = parse_size_list(ev.get_string("src_origin"));
        ecmd.dst_origin = parse_size_list(ev.get_string("dst_origin"));
        ecmd.origin = parse_size_list(ev.get_string("origin"));
        ecmd.region = parse_size_list(ev.get_string("region"));
        ecmd.src_row_pitch = static_cast<size_t>(ev.get_uint("src_row_pitch", 0));
        ecmd.src_slice_pitch = static_cast<size_t>(ev.get_uint("src_slice_pitch", 0));
        ecmd.dst_row_pitch = static_cast<size_t>(ev.get_uint("dst_row_pitch", 0));
        ecmd.dst_slice_pitch = static_cast<size_t>(ev.get_uint("dst_slice_pitch", 0));
        ecmd.pattern = clkp::hex_to_bytes(
            ev.has_key("pattern_hex") ? ev.get_string("pattern_hex") : ev.get_string("fill_color_hex"));
        ecmd.work_dim = static_cast<cl_uint>(ev.get_uint("work_dim", 1));
        ecmd.gwo = parse_size_list(ev.get_string("gwo"));
        ecmd.gws = parse_size_list(ev.get_string("gws"));
        ecmd.lws = parse_size_list(ev.get_string("lws"));
        ecmd.kernel_args_snapshot = current_kernel_args_[ecmd.kernel_id];
        emulated_cmdbufs_[cb_id].push_back(std::move(ecmd));
    }

    bool setup_command_recording(const clkp::ParsedEvent &ev)
    {
        uint64_t cb_id = ev.get_uint("cmdbuf_id", 0);
        uint64_t q_id = ev.get_uint("queue_id", 0);
        uint64_t traced_ret_sp = ev.get_uint("ret_sync_point", 0);
        uint64_t mut_id = ev.get_uint("mutable_cmd_id", 0);

        if (opts_.emulate_command_buffer || !ext_.supports_command_buffer() || !cmdbufs_.count(cb_id)) {
            record_emulated_command(ev, cb_id, q_id, mut_id);
            return true;
        }

        cl_command_queue explicit_q = (q_id != 0 && queues_.count(q_id)) ? queues_[q_id] : nullptr;
        std::vector<cl_sync_point_khr> wait_sps;
        for (uint64_t sp : clkp::string_to_uint64_list(ev.get_string("wait_sync_points"))) {
            auto &sp_map = sync_points_[cb_id];
            if (sp_map.count(static_cast<cl_uint>(sp))) {
                wait_sps.push_back(sp_map[static_cast<cl_uint>(sp)]);
            }
        }

        cl_sync_point_khr out_sp = 0;
        cl_mutable_command_khr out_mut = nullptr;
        cl_int err = invoke_native_command_recording(
            ev, cmdbufs_[cb_id], explicit_q, wait_sps, traced_ret_sp ? &out_sp : nullptr, mut_id ? &out_mut : nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: " << ev.name << " failed (" << err << ")\n";
            return false;
        }
        if (traced_ret_sp != 0) {
            sync_points_[cb_id][static_cast<cl_uint>(traced_ret_sp)] = out_sp;
        }
        if (mut_id != 0 && out_mut != nullptr) {
            mutable_cmds_[mut_id] = out_mut;
        }
        return true;
    }

    bool setup_finalize_command_buffer(const clkp::ParsedEvent &ev)
    {
        uint64_t cb_id = ev.get_uint("cmdbuf_id", 0);
        if (!opts_.emulate_command_buffer && ext_.supports_command_buffer() && cmdbufs_.count(cb_id)) {
            cl_int err = ext_.clFinalizeCommandBufferKHR(cmdbufs_[cb_id]);
            if (err != CL_SUCCESS) {
                std::cerr << "Error: clFinalizeCommandBufferKHR failed (" << err << ")\n";
                return false;
            }
        }
        return true;
    }

    bool dispatch_setup_event(const clkp::ParsedEvent &ev, uint64_t cid, uint64_t window_start_call,
        uint64_t window_end_call, std::vector<const clkp::ParsedEvent *> &hot_loop_events)
    {
        const std::string &name = ev.name;
        if (name == "clCreateContext" || name == "clCreateContextFromType") {
            return setup_create_context(ev);
        }
        if (name == "clCreateCommandQueue" || name == "clCreateCommandQueueWithProperties") {
            return setup_create_command_queue(ev);
        }
        if (name == "clCreateProgramWithSource") {
            return setup_create_program_with_source(ev);
        }
        if (name == "clCreateProgramWithIL") {
            return setup_create_program_with_il(ev);
        }
        if (name == "clCreateProgramWithBinary") {
            return setup_create_program_with_binary(ev);
        }
        if (name == "clSetProgramSpecializationConstant") {
            return setup_specialization_constant(ev);
        }
        if (name == "clBuildProgram") {
            return setup_build_program(ev);
        }
        if (name == "clCompileProgram") {
            return setup_compile_program(ev);
        }
        if (name == "clLinkProgram") {
            return setup_link_program(ev);
        }
        if (name == "clCreateKernel") {
            return setup_create_kernel(ev);
        }
        if (name == "clCreateKernelsInProgram") {
            return setup_create_kernels_in_program(ev, cid);
        }
        if (name == "clCloneKernel") {
            return setup_clone_kernel(ev);
        }
        if (name == "clCreateBuffer" || name == "clCreateBufferWithProperties") {
            return setup_create_buffer(ev);
        }
        if (name == "clCreateSubBuffer") {
            return setup_create_sub_buffer(ev);
        }
        if (name == "clCreateImage" || name == "clCreateImageWithProperties" || name == "clCreateImage2D"
            || name == "clCreateImage3D") {
            return setup_create_image(ev);
        }
        if (name == "clCreateSampler" || name == "clCreateSamplerWithProperties") {
            return setup_create_sampler(ev);
        }
        if (name == "clSetKernelArg") {
            uint64_t kern_id = ev.get_uint("kernel_id", 0);
            KernelArgSpec spec = parse_kernel_arg_from_event(ev);
            current_kernel_args_[kern_id][spec.arg_index] = spec;
            if (kernels_.count(kern_id)) {
                apply_kernel_arg(kernels_[kern_id], spec);
            }
            if (cid >= window_start_call && cid <= window_end_call) {
                hot_loop_events.push_back(&ev);
            }
            return true;
        }
        if (name == "clCreateCommandBufferKHR") {
            return setup_create_command_buffer(ev);
        }
        if (is_command_recording_call(name)) {
            return setup_command_recording(ev);
        }
        if (name == "clFinalizeCommandBufferKHR") {
            return setup_finalize_command_buffer(ev);
        }
        if (cid >= window_start_call && cid <= window_end_call && is_execution_call(name)) {
            hot_loop_events.push_back(&ev);
        }
        return true;
    }

    bool promote_event_to_command_buffer(const clkp::ParsedEvent &ev)
    {
        if (ev.name == "clSetKernelArg") {
            uint64_t kern_id = ev.get_uint("kernel_id", 0);
            if (kernels_.count(kern_id)) {
                apply_kernel_arg(kernels_[kern_id], parse_kernel_arg_from_event(ev));
            }
            return true;
        }
        if (ev.name == "clEnqueueNDRangeKernel") {
            cl_kernel k
                = resolve_kernel(ev.get_uint("kernel_id", 0), ev.get_string("program"), ev.get_string("kernel_name"));
            cl_uint work_dim = static_cast<cl_uint>(ev.get_uint("work_dim", 1));
            auto gwo = parse_size_list(ev.get_string("gwo"));
            auto gws = parse_size_list(ev.get_string("gws"));
            if (gws.empty()) {
                gws = { static_cast<size_t>(ev.get_uint("gidX", 1)), static_cast<size_t>(ev.get_uint("gidY", 1)),
                    static_cast<size_t>(ev.get_uint("gidZ", 1)) };
                work_dim = 3;
            }
            auto lws = parse_size_list(ev.get_string("lws"));
            cl_int err = ext_.clCommandNDRangeKernelKHR(promoted_cmdbuf_, nullptr, nullptr, k, work_dim,
                gwo.empty() ? nullptr : gwo.data(), gws.data(), lws.empty() ? nullptr : lws.data(), 0, nullptr, nullptr,
                nullptr);
            if (err == CL_INVALID_WORK_GROUP_SIZE && opts_.allow_lws_fallback) {
                err = ext_.clCommandNDRangeKernelKHR(promoted_cmdbuf_, nullptr, nullptr, k, work_dim,
                    gwo.empty() ? nullptr : gwo.data(), gws.data(), nullptr, 0, nullptr, nullptr, nullptr);
            }
            if (err != CL_SUCCESS) {
                std::cerr << "Error: clCommandNDRangeKernelKHR failed (" << err << ")\n";
                return false;
            }
            return true;
        }
        if (ev.name == "clEnqueueCopyBuffer" && ext_.clCommandCopyBufferKHR) {
            ext_.clCommandCopyBufferKHR(promoted_cmdbuf_, nullptr, nullptr, mems_[ev.get_uint("src_mem_id")],
                mems_[ev.get_uint("dst_mem_id")], static_cast<size_t>(ev.get_uint("src_offset")),
                static_cast<size_t>(ev.get_uint("dst_offset")), static_cast<size_t>(ev.get_uint("size")), 0, nullptr,
                nullptr, nullptr);
        } else if (ev.name == "clEnqueueFillBuffer" && ext_.clCommandFillBufferKHR) {
            auto pat = clkp::hex_to_bytes(ev.get_string("pattern_hex"));
            ext_.clCommandFillBufferKHR(promoted_cmdbuf_, nullptr, nullptr, mems_[ev.get_uint("mem_id")], pat.data(),
                pat.size(), static_cast<size_t>(ev.get_uint("offset")), static_cast<size_t>(ev.get_uint("size")), 0,
                nullptr, nullptr, nullptr);
        }
        return true;
    }

    bool build_promoted_command_buffer(const std::vector<const clkp::ParsedEvent *> &hot_loop_events)
    {
        if (!ext_.supports_command_buffer()) {
            std::cerr << "Error: --use-command-buffer requested, but device does not support cl_khr_command_buffer.\n";
            return false;
        }
        promoted_queue_ = get_default_queue();
        cl_int err = CL_SUCCESS;
        promoted_cmdbuf_ = ext_.clCreateCommandBufferKHR(1, &promoted_queue_, nullptr, &err);
        if (err != CL_SUCCESS || !promoted_cmdbuf_) {
            std::cerr << "Error: failed to create command buffer for --use-command-buffer (" << err << ").\n";
            return false;
        }
        for (const auto *ev_ptr : hot_loop_events) {
            if (!promote_event_to_command_buffer(*ev_ptr)) {
                return false;
            }
        }
        err = ext_.clFinalizeCommandBufferKHR(promoted_cmdbuf_);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: clFinalizeCommandBufferKHR failed (" << err << ")\n";
            return false;
        }
        return true;
    }

    bool exec_ndrange_kernel(const clkp::ParsedEvent &ev, cl_command_queue q, const std::vector<cl_event> &wait_list,
        uint64_t ret_ev_id, std::map<uint64_t, cl_event> &live_events,
        std::vector<std::pair<uint64_t, cl_event>> &iter_prof_events, std::map<uint64_t, DispatchMetric> &metrics)
    {
        uint64_t cid = ev.get_uint("call_id", 0);
        uint64_t did = ev.get_uint("dispatch_id", cid);
        std::string prog_str = ev.get_string("program", "");
        std::string kname = ev.get_string("kernel_name", "");
        cl_kernel k = resolve_kernel(ev.get_uint("kernel_id", 0), prog_str, kname);
        cl_uint work_dim = static_cast<cl_uint>(ev.get_uint("work_dim", 0));
        auto gwo = parse_size_list(ev.get_string("gwo"));
        auto gws = parse_size_list(ev.get_string("gws"));
        auto lws = parse_size_list(ev.get_string("lws"));
        if (gws.empty()) {
            gws = { static_cast<size_t>(ev.get_uint("gidX", 1)), static_cast<size_t>(ev.get_uint("gidY", 1)),
                static_cast<size_t>(ev.get_uint("gidZ", 1)) };
            if (work_dim == 0) {
                work_dim = (gws[2] > 1) ? 3 : ((gws[1] > 1) ? 2 : 1);
            }
        }

        cl_event prof_ev = nullptr;
        cl_int err = clEnqueueNDRangeKernel(q, k, work_dim, gwo.empty() ? nullptr : gwo.data(), gws.data(),
            lws.empty() ? nullptr : lws.data(), static_cast<cl_uint>(wait_list.size()),
            wait_list.empty() ? nullptr : wait_list.data(), &prof_ev);
        if (err == CL_INVALID_WORK_GROUP_SIZE && opts_.allow_lws_fallback) {
            err = clEnqueueNDRangeKernel(q, k, work_dim, gwo.empty() ? nullptr : gwo.data(), gws.data(), nullptr,
                static_cast<cl_uint>(wait_list.size()), wait_list.empty() ? nullptr : wait_list.data(), &prof_ev);
        }
        if (err != CL_SUCCESS) {
            std::cerr << "Error: clEnqueueNDRangeKernel('" << kname << "') failed (" << err << ")\n";
            return false;
        }
        if (ret_ev_id != 0) {
            live_events[ret_ev_id] = prof_ev;
        }
        auto &m = metrics[did];
        m.dispatch_id = did;
        m.call_id = cid;
        m.label = prog_str + ":" + kname;
        m.gws = ev.get_string("gws");
        m.lws = ev.get_string("lws", "NULL");
        iter_prof_events.push_back({ did, prof_ev });
        return true;
    }

    std::vector<MutableConfigSpec> parse_mutable_configs(uint64_t cid)
    {
        std::vector<MutableConfigSpec> cfg_specs;
        for (const auto &aev : td_.aux_by_call_id[cid]) {
            if (aev.name == "clUpdateMutableCommandsKHR-config") {
                MutableConfigSpec cs;
                cs.mutable_cmd_id = aev.get_uint("mutable_cmd_id", 0);
                cs.work_dim = static_cast<cl_uint>(aev.get_uint("work_dim", 0));
                cs.gwo = parse_size_list(aev.get_string("gwo"));
                cs.gws = parse_size_list(aev.get_string("gws"));
                cs.lws = parse_size_list(aev.get_string("lws"));
                cfg_specs.push_back(std::move(cs));
            } else if (aev.name == "clUpdateMutableCommandsKHR-arg" && !cfg_specs.empty()) {
                cfg_specs.back().args.push_back(parse_kernel_arg_from_event(aev));
            }
        }
        return cfg_specs;
    }

    void apply_native_mutable_update(uint64_t cb_id, const MutableConfigSpec &cs)
    {
        if (!mutable_cmds_.count(cs.mutable_cmd_id)) {
            return;
        }
        std::vector<cl_mutable_dispatch_arg_khr> cl_args;
        std::vector<cl_mem> mem_holders(cs.args.size(), nullptr);
        std::vector<cl_sampler> samp_holders(cs.args.size(), nullptr);
        for (size_t i = 0; i < cs.args.size(); ++i) {
            const auto &aspec = cs.args[i];
            cl_mutable_dispatch_arg_khr a = { aspec.arg_index, aspec.arg_size, nullptr };
            if (aspec.arg_kind == "MEM") {
                mem_holders[i] = mems_[aspec.mem_id];
                a.arg_size = sizeof(cl_mem);
                a.arg_value = &mem_holders[i];
            } else if (aspec.arg_kind == "SAMPLER") {
                samp_holders[i] = samplers_[aspec.sampler_id];
                a.arg_size = sizeof(cl_sampler);
                a.arg_value = &samp_holders[i];
            } else if (aspec.arg_kind != "LOCAL") {
                a.arg_value = aspec.value_bytes.data();
            }
            cl_args.push_back(a);
        }
        cl_mutable_dispatch_config_khr dcfg;
        memset(&dcfg, 0, sizeof(dcfg));
        dcfg.command = mutable_cmds_[cs.mutable_cmd_id];
        dcfg.num_args = static_cast<cl_uint>(cl_args.size());
        dcfg.arg_list = cl_args.empty() ? nullptr : cl_args.data();
        dcfg.work_dim = cs.work_dim;
        dcfg.global_work_offset = cs.gwo.empty() ? nullptr : cs.gwo.data();
        dcfg.global_work_size = cs.gws.empty() ? nullptr : cs.gws.data();
        dcfg.local_work_size = cs.lws.empty() ? nullptr : cs.lws.data();
        cl_command_buffer_update_type_khr utype = CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR;
        const void *configs_arr[1] = { &dcfg };
        ext_.clUpdateMutableCommandsKHR(cmdbufs_[cb_id], 1, &utype, configs_arr);
    }

    void apply_emulated_mutable_update(uint64_t cb_id, const MutableConfigSpec &cs)
    {
        for (auto &ecmd : emulated_cmdbufs_[cb_id]) {
            if (ecmd.mutable_cmd_id != cs.mutable_cmd_id || cs.mutable_cmd_id == 0) {
                continue;
            }
            if (!cs.gwo.empty()) {
                ecmd.gwo = cs.gwo;
            }
            if (!cs.gws.empty()) {
                ecmd.gws = cs.gws;
            }
            if (!cs.lws.empty()) {
                ecmd.lws = cs.lws;
            }
            for (const auto &aspec : cs.args) {
                ecmd.kernel_args_snapshot[aspec.arg_index] = aspec;
            }
        }
    }

    void exec_update_mutable_commands(const clkp::ParsedEvent &ev)
    {
        uint64_t cid = ev.get_uint("call_id", 0);
        uint64_t cb_id = ev.get_uint("cmdbuf_id", 0);
        auto cfg_specs = parse_mutable_configs(cid);

        if (!opts_.emulate_command_buffer && ext_.supports_command_buffer() && cmdbufs_.count(cb_id)
            && ext_.clUpdateMutableCommandsKHR) {
            for (const auto &cs : cfg_specs) {
                apply_native_mutable_update(cb_id, cs);
            }
        } else if (emulated_cmdbufs_.count(cb_id)) {
            for (const auto &cs : cfg_specs) {
                apply_emulated_mutable_update(cb_id, cs);
            }
        }
    }

    bool exec_emulated_command_buffer(uint64_t cb_id, uint64_t did, uint64_t cid, cl_command_queue target_q,
        std::vector<std::pair<uint64_t, cl_event>> &iter_prof_events, std::map<uint64_t, DispatchMetric> &metrics)
    {
        for (const auto &ecmd : emulated_cmdbufs_[cb_id]) {
            if (ecmd.api_name == "clCommandNDRangeKernelKHR") {
                cl_kernel k = kernels_[ecmd.kernel_id];
                for (const auto &[aidx, aspec] : ecmd.kernel_args_snapshot) {
                    apply_kernel_arg(k, aspec);
                }
                cl_event prof_ev = nullptr;
                cl_int err
                    = clEnqueueNDRangeKernel(target_q, k, ecmd.work_dim, ecmd.gwo.empty() ? nullptr : ecmd.gwo.data(),
                        ecmd.gws.data(), ecmd.lws.empty() ? nullptr : ecmd.lws.data(), 0, nullptr, &prof_ev);
                if (err == CL_INVALID_WORK_GROUP_SIZE && opts_.allow_lws_fallback) {
                    err = clEnqueueNDRangeKernel(target_q, k, ecmd.work_dim,
                        ecmd.gwo.empty() ? nullptr : ecmd.gwo.data(), ecmd.gws.data(), nullptr, 0, nullptr, &prof_ev);
                }
                if (err != CL_SUCCESS) {
                    std::cerr << "Error: emulated clCommandNDRangeKernelKHR failed (" << err << ")\n";
                    return false;
                }
                auto &m = metrics[did];
                m.dispatch_id = did;
                m.call_id = cid;
                m.label = "emulated_cmdbuf_" + std::to_string(cb_id);
                iter_prof_events.push_back({ did, prof_ev });
            } else if (ecmd.api_name == "clCommandCopyBufferKHR") {
                clEnqueueCopyBuffer(target_q, mems_[ecmd.src_mem_id], mems_[ecmd.dst_mem_id], ecmd.src_offset,
                    ecmd.dst_offset, ecmd.size, 0, nullptr, nullptr);
            } else if (ecmd.api_name == "clCommandFillBufferKHR") {
                clEnqueueFillBuffer(target_q, mems_[ecmd.mem_id], ecmd.pattern.data(), ecmd.pattern.size(), ecmd.offset,
                    ecmd.size, 0, nullptr, nullptr);
            } else if (ecmd.api_name == "clCommandBarrierWithWaitListKHR") {
                clEnqueueBarrierWithWaitList(target_q, 0, nullptr, nullptr);
            }
        }
        return true;
    }

    bool exec_command_buffer(const clkp::ParsedEvent &ev, const std::vector<cl_event> &wait_list, uint64_t ret_ev_id,
        std::map<uint64_t, cl_event> &live_events, std::vector<std::pair<uint64_t, cl_event>> &iter_prof_events,
        std::map<uint64_t, DispatchMetric> &metrics)
    {
        uint64_t cid = ev.get_uint("call_id", 0);
        uint64_t did = ev.get_uint("dispatch_id", cid);
        uint64_t cb_id = ev.get_uint("cmdbuf_id", 0);
        std::vector<cl_command_queue> q_vec;
        for (uint64_t qid : clkp::string_to_uint64_list(ev.get_string("queue_ids"))) {
            if (queues_.count(qid)) {
                q_vec.push_back(queues_[qid]);
            }
        }

        if (!opts_.emulate_command_buffer && ext_.supports_command_buffer() && cmdbufs_.count(cb_id)) {
            cl_event prof_ev = nullptr;
            cl_int err = ext_.clEnqueueCommandBufferKHR(static_cast<cl_uint>(q_vec.size()),
                q_vec.empty() ? nullptr : q_vec.data(), cmdbufs_[cb_id], static_cast<cl_uint>(wait_list.size()),
                wait_list.empty() ? nullptr : wait_list.data(), &prof_ev);
            if (err != CL_SUCCESS) {
                std::cerr << "Error: clEnqueueCommandBufferKHR failed (" << err << ")\n";
                return false;
            }
            if (ret_ev_id != 0) {
                live_events[ret_ev_id] = prof_ev;
            }
            auto &m = metrics[did];
            m.dispatch_id = did;
            m.call_id = cid;
            m.label = "cmdbuf_" + std::to_string(cb_id);
            iter_prof_events.push_back({ did, prof_ev });
            return true;
        }

        if (emulated_cmdbufs_.count(cb_id)) {
            cl_command_queue target_q = !q_vec.empty() ? q_vec[0] : get_queue_or_default(cmdbuf_default_queue_[cb_id]);
            return exec_emulated_command_buffer(cb_id, did, cid, target_q, iter_prof_events, metrics);
        }
        return true;
    }

    void exec_auxiliary_queue_command(const clkp::ParsedEvent &ev, cl_command_queue q,
        const std::vector<cl_event> &wait_list, uint64_t ret_ev_id, std::map<uint64_t, cl_event> &live_events)
    {
        const std::string &name = ev.name;
        cl_uint num_wait = static_cast<cl_uint>(wait_list.size());
        const cl_event *wait_ptr = wait_list.empty() ? nullptr : wait_list.data();
        cl_event out_ev = nullptr;
        cl_event *out_ptr = ret_ev_id ? &out_ev : nullptr;

        if (name == "clEnqueueCopyBuffer") {
            clEnqueueCopyBuffer(q, mems_[ev.get_uint("src_mem_id")], mems_[ev.get_uint("dst_mem_id")],
                static_cast<size_t>(ev.get_uint("src_offset")), static_cast<size_t>(ev.get_uint("dst_offset")),
                static_cast<size_t>(ev.get_uint("size")), num_wait, wait_ptr, out_ptr);
        } else if (name == "clEnqueueFillBuffer") {
            auto pat = clkp::hex_to_bytes(ev.get_string("pattern_hex"));
            clEnqueueFillBuffer(q, mems_[ev.get_uint("mem_id")], pat.data(), pat.size(),
                static_cast<size_t>(ev.get_uint("offset")), static_cast<size_t>(ev.get_uint("size")), num_wait,
                wait_ptr, out_ptr);
        } else if (name == "clEnqueueWriteBuffer" || name == "clEnqueueReadBuffer") {
            size_t sz = static_cast<size_t>(ev.get_uint("size", 0));
            size_t off = static_cast<size_t>(ev.get_uint("offset", 0));
            uint64_t mid = ev.get_uint("mem_id", 0);
            if (mems_.count(mid) && sz > 0) {
                if (host_scratch_buffer_.size() < sz) {
                    host_scratch_buffer_.resize(sz, 0);
                }
                if (name == "clEnqueueWriteBuffer") {
                    clEnqueueWriteBuffer(
                        q, mems_[mid], CL_FALSE, off, sz, host_scratch_buffer_.data(), num_wait, wait_ptr, out_ptr);
                } else {
                    clEnqueueReadBuffer(
                        q, mems_[mid], CL_FALSE, off, sz, host_scratch_buffer_.data(), num_wait, wait_ptr, out_ptr);
                }
            }
        } else if (name == "clEnqueueMapBuffer") {
            uint64_t mid = ev.get_uint("mem_id", 0);
            uint64_t map_id = ev.get_uint("map_id", 0);
            if (mems_.count(mid)) {
                cl_int err = CL_SUCCESS;
                void *ptr = clEnqueueMapBuffer(q, mems_[mid], CL_TRUE,
                    static_cast<cl_map_flags>(ev.get_uint("map_flags", CL_MAP_READ)),
                    static_cast<size_t>(ev.get_uint("offset", 0)), static_cast<size_t>(ev.get_uint("size", 0)),
                    num_wait, wait_ptr, out_ptr, &err);
                if (ptr && map_id != 0) {
                    active_maps_[map_id] = ptr;
                }
            }
        } else if (name == "clEnqueueUnmapMemObject") {
            uint64_t mid = ev.get_uint("mem_id", 0);
            uint64_t map_id = ev.get_uint("map_id", 0);
            if (mems_.count(mid) && active_maps_.count(map_id)) {
                clEnqueueUnmapMemObject(q, mems_[mid], active_maps_[map_id], num_wait, wait_ptr, out_ptr);
                active_maps_.erase(map_id);
            }
        } else if (name == "clEnqueueBarrierWithWaitList") {
            clEnqueueBarrierWithWaitList(q, num_wait, wait_ptr, out_ptr);
        } else if (name == "clEnqueueMarkerWithWaitList") {
            clEnqueueMarkerWithWaitList(q, num_wait, wait_ptr, out_ptr);
        } else if (name == "clFlush") {
            clFlush(q);
        } else if (name == "clFinish") {
            clFinish(q);
        } else if (name == "clWaitForEvents" && !wait_list.empty()) {
            clWaitForEvents(num_wait, wait_ptr);
        }

        if (ret_ev_id != 0 && out_ev != nullptr) {
            live_events[ret_ev_id] = out_ev;
        }
    }

    bool execute_hot_loop_event(const clkp::ParsedEvent &ev, std::map<uint64_t, cl_event> &live_events,
        std::vector<std::pair<uint64_t, cl_event>> &iter_prof_events, std::map<uint64_t, DispatchMetric> &metrics)
    {
        const std::string &name = ev.name;
        cl_command_queue q = get_queue_or_default(ev.get_uint("queue_id", 1));
        std::vector<cl_event> wait_list;
        for (uint64_t wid : clkp::string_to_uint64_list(ev.get_string("wait_events"))) {
            if (live_events.count(wid) && live_events[wid] != nullptr) {
                wait_list.push_back(live_events[wid]);
            }
        }
        uint64_t ret_ev_id = ev.get_uint("ret_event_id", 0);

        if (name == "clSetKernelArg") {
            uint64_t kern_id = ev.get_uint("kernel_id", 0);
            KernelArgSpec spec = parse_kernel_arg_from_event(ev);
            current_kernel_args_[kern_id][spec.arg_index] = spec;
            if (kernels_.count(kern_id)) {
                apply_kernel_arg(kernels_[kern_id], spec);
            }
            return true;
        }
        if (name == "clEnqueueNDRangeKernel" || name == "clEnqueueTask") {
            return exec_ndrange_kernel(ev, q, wait_list, ret_ev_id, live_events, iter_prof_events, metrics);
        }
        if (name == "clUpdateMutableCommandsKHR") {
            exec_update_mutable_commands(ev);
            return true;
        }
        if (name == "clEnqueueCommandBufferKHR") {
            return exec_command_buffer(ev, wait_list, ret_ev_id, live_events, iter_prof_events, metrics);
        }
        exec_auxiliary_queue_command(ev, q, wait_list, ret_ev_id, live_events);
        return true;
    }

    void collect_iteration_profiling(bool is_measured, uint32_t measured_idx,
        const std::vector<std::pair<uint64_t, cl_event>> &iter_prof_events,
        const std::map<uint64_t, cl_event> &live_events, std::map<uint64_t, DispatchMetric> &metrics,
        std::vector<uint64_t> &iter_total_ns)
    {
        std::set<cl_event> released_events;
        for (const auto &[did, ev_handle] : iter_prof_events) {
            if (!ev_handle) {
                continue;
            }
            if (is_measured) {
                cl_ulong start_ns = 0;
                cl_ulong end_ns = 0;
                if (clGetEventProfilingInfo(ev_handle, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns, nullptr)
                        == CL_SUCCESS
                    && clGetEventProfilingInfo(ev_handle, CL_PROFILING_COMMAND_END, sizeof(end_ns), &end_ns, nullptr)
                        == CL_SUCCESS
                    && end_ns >= start_ns) {
                    uint64_t dur = static_cast<uint64_t>(end_ns - start_ns);
                    metrics[did].durations_ns.push_back(dur);
                    iter_total_ns[measured_idx] += dur;
                }
            }
            if (released_events.insert(ev_handle).second) {
                clReleaseEvent(ev_handle);
            }
        }
        for (const auto &[eid, ev_handle] : live_events) {
            if (ev_handle && released_events.insert(ev_handle).second) {
                clReleaseEvent(ev_handle);
            }
        }
    }

    bool run_single_iteration(const std::vector<const clkp::ParsedEvent *> &hot_loop_events, bool is_measured,
        uint32_t measured_idx, std::map<uint64_t, DispatchMetric> &metrics, std::vector<uint64_t> &iter_total_ns)
    {
        std::vector<std::pair<uint64_t, cl_event>> iter_prof_events;
        std::map<uint64_t, cl_event> live_events;

        if (promoted_cmdbuf_ != nullptr) {
            cl_event ev_out = nullptr;
            cl_int err = ext_.clEnqueueCommandBufferKHR(0, nullptr, promoted_cmdbuf_, 0, nullptr, &ev_out);
            if (err != CL_SUCCESS) {
                std::cerr << "Error: clEnqueueCommandBufferKHR failed (" << err << ")\n";
                return false;
            }
            auto &m = metrics[0];
            m.dispatch_id = 0;
            m.call_id = 0;
            m.label = "promoted_cmdbuf";
            iter_prof_events.push_back({ 0, ev_out });
            clFinish(promoted_queue_);
        } else {
            for (const auto *ev_ptr : hot_loop_events) {
                if (!execute_hot_loop_event(*ev_ptr, live_events, iter_prof_events, metrics)) {
                    return false;
                }
            }
            for (auto &[qid, q] : queues_) {
                clFinish(q);
            }
        }

        collect_iteration_profiling(is_measured, measured_idx, iter_prof_events, live_events, metrics, iter_total_ns);
        return true;
    }

    void release_resources()
    {
        if (promoted_cmdbuf_ && ext_.clReleaseCommandBufferKHR) {
            ext_.clReleaseCommandBufferKHR(promoted_cmdbuf_);
        }
        for (auto &[id, cb] : cmdbufs_) {
            if (cb && ext_.clReleaseCommandBufferKHR) {
                ext_.clReleaseCommandBufferKHR(cb);
            }
        }
        for (auto &[id, s] : samplers_) {
            if (s) {
                clReleaseSampler(s);
            }
        }
        for (auto &[id, m] : mems_) {
            if (m) {
                clReleaseMemObject(m);
            }
        }
        for (auto &[id, k] : kernels_) {
            if (k) {
                clReleaseKernel(k);
            }
        }
        for (auto &[id, p] : programs_) {
            if (p) {
                clReleaseProgram(p);
            }
        }
        for (auto &[id, q] : queues_) {
            if (q) {
                clReleaseCommandQueue(q);
            }
        }
        for (auto &[id, c] : contexts_) {
            if (c) {
                clReleaseContext(c);
            }
        }
    }
};

std::tuple<double, double, double, double> compute_stats_us(std::vector<uint64_t> vals)
{
    if (vals.empty()) {
        return { 0.0, 0.0, 0.0, 0.0 };
    }
    std::sort(vals.begin(), vals.end());
    double min_us = static_cast<double>(vals.front()) / 1000.0;
    double max_us = static_cast<double>(vals.back()) / 1000.0;
    double sum_ns = 0.0;
    for (uint64_t v : vals) {
        sum_ns += static_cast<double>(v);
    }
    double avg_us = (sum_ns / static_cast<double>(vals.size())) / 1000.0;
    double med_us = (vals.size() % 2 == 1)
        ? (static_cast<double>(vals[vals.size() / 2]) / 1000.0)
        : (static_cast<double>(vals[vals.size() / 2 - 1] + vals[vals.size() / 2]) / 2000.0);
    return { min_us, avg_us, med_us, max_us };
}

std::vector<TargetMetric> aggregate_metrics_by_target(
    const std::map<uint64_t, DispatchMetric> &metrics, uint32_t iterations)
{
    std::vector<TargetMetric> targets;
    std::unordered_map<std::string, size_t> label_to_idx;

    for (const auto &[did, m] : metrics) {
        auto it = label_to_idx.find(m.label);
        if (it == label_to_idx.end()) {
            label_to_idx[m.label] = targets.size();
            TargetMetric tm;
            tm.label = m.label;
            targets.push_back(std::move(tm));
            it = label_to_idx.find(m.label);
        }
        auto &tm = targets[it->second];
        tm.dispatch_count++;
        tm.durations_ns.insert(tm.durations_ns.end(), m.durations_ns.begin(), m.durations_ns.end());
    }

    double denom = iterations > 0 ? (static_cast<double>(iterations) * 1000.0) : 1000.0;
    for (auto &tm : targets) {
        double sum_ns = 0.0;
        for (uint64_t v : tm.durations_ns) {
            sum_ns += static_cast<double>(v);
        }
        tm.total_iter_us = sum_ns / denom;
    }

    return targets;
}

void print_dispatch_profiling_report(const RunnerOptions &opts, const std::map<uint64_t, DispatchMetric> &metrics,
    const std::vector<uint64_t> &iter_total_ns)
{
    std::cout << "\n=== Replay Profiling Results (warmup=" << opts.warmup_iterations
              << ", iterations=" << opts.iterations << ") ===\n";
    std::cout << std::left << std::setw(12) << "DISPATCH" << std::setw(28) << "TARGET" << std::setw(14) << "MIN (us)"
              << std::setw(14) << "AVG (us)" << std::setw(14) << "MEDIAN (us)" << "MAX (us)\n";

    for (const auto &[did, m] : metrics) {
        auto [min_us, avg_us, med_us, max_us] = compute_stats_us(m.durations_ns);
        std::cout << std::left << std::setw(12) << m.dispatch_id << std::setw(28) << m.label << std::fixed
                  << std::setprecision(2) << std::setw(14) << min_us << std::setw(14) << avg_us << std::setw(14)
                  << med_us << std::setw(14) << max_us << "\n";
    }

    auto [tot_min, tot_avg, tot_med, tot_max] = compute_stats_us(iter_total_ns);
    std::cout << "--------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(12) << "TOTAL/ITER" << std::setw(28)
              << ("(" + std::to_string(metrics.size()) + " dispatches)") << std::fixed << std::setprecision(2)
              << std::setw(14) << tot_min << std::setw(14) << tot_avg << std::setw(14) << tot_med << std::setw(14)
              << tot_max << "\n";
}

void print_target_profiling_report(const RunnerOptions &opts, const std::map<uint64_t, DispatchMetric> &metrics,
    const std::vector<uint64_t> &iter_total_ns)
{
    auto targets = aggregate_metrics_by_target(metrics, opts.iterations);

    std::cout << "\n=== Replay Profiling Results (warmup=" << opts.warmup_iterations
              << ", iterations=" << opts.iterations << ") ===\n";
    std::cout << std::left << std::setw(28) << "TARGET" << std::setw(12) << "DISPATCHES" << std::setw(14) << "MIN (us)"
              << std::setw(14) << "AVG (us)" << std::setw(14) << "MEDIAN (us)" << std::setw(14) << "MAX (us)"
              << "TOTAL/ITER (us)\n";

    for (const auto &tm : targets) {
        auto [min_us, avg_us, med_us, max_us] = compute_stats_us(tm.durations_ns);
        std::cout << std::left << std::setw(28) << tm.label << std::setw(12) << tm.dispatch_count << std::fixed
                  << std::setprecision(2) << std::setw(14) << min_us << std::setw(14) << avg_us << std::setw(14)
                  << med_us << std::setw(14) << max_us << std::setw(14) << tm.total_iter_us << "\n";
    }

    auto [tot_min, tot_avg, tot_med, tot_max] = compute_stats_us(iter_total_ns);
    std::cout << "-----------------------------------------------------------------------------------------------------"
                 "-------\n";
    std::cout << std::left << std::setw(28) << "TOTAL/ITER" << std::setw(12) << metrics.size() << std::fixed
              << std::setprecision(2) << std::setw(14) << tot_min << std::setw(14) << tot_avg << std::setw(14)
              << tot_med << std::setw(14) << tot_max << std::setw(14) << tot_avg << "\n";
}

void print_profiling_report(const RunnerOptions &opts, const std::map<uint64_t, DispatchMetric> &metrics,
    const std::vector<uint64_t> &iter_total_ns)
{
    if (opts.by_dispatch) {
        print_dispatch_profiling_report(opts, metrics, iter_total_ns);
    } else {
        print_target_profiling_report(opts, metrics, iter_total_ns);
    }
}

void write_json_report(const RunnerOptions &opts, const std::string &platform_name, const std::string &device_name,
    const std::map<uint64_t, DispatchMetric> &metrics, const std::vector<uint64_t> &iter_total_ns)
{
    if (opts.json_output_path.empty()) {
        return;
    }
    std::ofstream jf(opts.json_output_path, std::ios::out | std::ios::trunc);
    if (!jf.is_open()) {
        return;
    }

    auto [tot_min, tot_avg, tot_med, tot_max] = compute_stats_us(iter_total_ns);
    auto targets = aggregate_metrics_by_target(metrics, opts.iterations);

    jf << "{\n"
       << "  \"platform\": \"" << platform_name << "\",\n"
       << "  \"device\": \"" << device_name << "\",\n"
       << "  \"warmup_iterations\": " << opts.warmup_iterations << ",\n"
       << "  \"iterations\": " << opts.iterations << ",\n"
       << "  \"total_iteration_us\": { \"min\": " << tot_min << ", \"avg\": " << tot_avg << ", \"median\": " << tot_med
       << ", \"max\": " << tot_max << " },\n"
       << "  \"targets\": [\n";
    size_t tidx = 0;
    for (const auto &tm : targets) {
        auto [min_us, avg_us, med_us, max_us] = compute_stats_us(tm.durations_ns);
        jf << "    { \"target\": \"" << tm.label << "\", \"dispatches\": " << tm.dispatch_count
           << ", \"min_us\": " << min_us << ", \"avg_us\": " << avg_us << ", \"median_us\": " << med_us
           << ", \"max_us\": " << max_us << ", \"total_iter_us\": " << tm.total_iter_us << " }"
           << (++tidx < targets.size() ? "," : "") << "\n";
    }
    jf << "  ],\n"
       << "  \"dispatches\": [\n";
    size_t idx = 0;
    for (const auto &[did, m] : metrics) {
        auto [min_us, avg_us, med_us, max_us] = compute_stats_us(m.durations_ns);
        jf << "    { \"dispatch_id\": " << m.dispatch_id << ", \"call_id\": " << m.call_id << ", \"label\": \""
           << m.label << "\", \"min_us\": " << min_us << ", \"avg_us\": " << avg_us << ", \"median_us\": " << med_us
           << ", \"max_us\": " << max_us << " }" << (++idx < metrics.size() ? "," : "") << "\n";
    }
    jf << "  ]\n}\n";
}

} // namespace

int main(int argc, char **argv)
{
    RunnerOptions opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<uint8_t> trace_bytes;
    if (!clkp::read_file_bytes(opts.input_path, trace_bytes)) {
        std::cerr << "Error: failed to read trace file '" << opts.input_path << "'\n";
        return 1;
    }

    TraceData trace_data;
    parse_trace_into_replay_stream(trace_bytes, trace_data);

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    std::string platform_name;
    std::string device_name;
    if (!select_platform_and_device(opts, platform, device, platform_name, device_name)) {
        return 1;
    }

    uint64_t window_start_call = 0;
    uint64_t window_end_call = UINT64_MAX;
    if (!resolve_dispatch_window(opts, trace_data.primary_events, window_start_call, window_end_call)) {
        return 1;
    }

    ReplayEngine engine(opts, platform, device, trace_data);
    std::vector<const clkp::ParsedEvent *> hot_loop_events;
    if (!engine.run_setup_phase(window_start_call, window_end_call, hot_loop_events)) {
        return 1;
    }

    std::map<uint64_t, DispatchMetric> metrics;
    std::vector<uint64_t> iter_total_ns;
    if (!engine.run_iterations(hot_loop_events, metrics, iter_total_ns)) {
        return 1;
    }

    print_profiling_report(opts, metrics, iter_total_ns);
    write_json_report(opts, platform_name, device_name, metrics, iter_total_ns);
    return 0;
}
