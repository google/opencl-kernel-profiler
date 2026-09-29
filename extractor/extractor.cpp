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

#include "clkp_common.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using TrackDescriptorMap = std::map<uint64_t, std::pair<const uint8_t *, size_t>>;
using SequenceTrackKey = std::pair<uint32_t, uint64_t>;

struct SliceContext {
    bool is_clkp = false;
    uint64_t call_id = 0;
    uint64_t dispatch_id = 0;
    uint64_t cmdbuf_id = 0;
    uint64_t begin_timestamp = 0;
    std::string name;
    size_t begin_event_index = 0;
};

struct DispatchSummary {
    uint64_t dispatch_id = 0;
    uint64_t call_id = 0;
    std::string api_name;
    std::string program;
    std::string kernel_name;
    uint64_t cmdbuf_id = 0;
    std::string gws;
    std::string lws;
    uint64_t gpu_duration_ns = 0;
    bool has_gpu_duration = false;
};

struct ExtractorOptions {
    std::string input_path;
    std::string output_path;
    bool list_dispatches = false;
    bool has_start_dispatch = false;
    bool has_end_dispatch = false;
    uint64_t start_dispatch = 0;
    uint64_t end_dispatch = UINT64_MAX;
    bool has_start_call = false;
    bool has_end_call = false;
    uint64_t start_call = 0;
    uint64_t end_call = UINT64_MAX;

    bool has_window() const { return has_start_dispatch || has_end_dispatch || has_start_call || has_end_call; }
};

void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog << " [options] -i <input.perfetto-trace> [-o <output.perfetto-trace>]\n"
              << "\nOptions:\n"
              << "  -i, --input <file>           Input Perfetto trace file\n"
              << "  -o, --output <file>          Output filtered Perfetto trace file\n"
              << "  -l, --list-dispatches        List all OpenCL dispatches and command buffer executions\n"
              << "  --start-dispatch <id>        Start extraction window at dispatch_id <id> (inclusive)\n"
              << "  --end-dispatch <id>          End extraction window at dispatch_id <id> (inclusive)\n"
              << "  --start-call <id>            Start extraction window at call_id <id> (inclusive)\n"
              << "  --end-call <id>              End extraction window at call_id <id> (inclusive)\n"
              << "  -h, --help                   Show this help message\n";
}

bool parse_args(int argc, char **argv, ExtractorOptions &opts)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            opts.input_path = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            opts.output_path = argv[++i];
        } else if (arg == "-l" || arg == "--list-dispatches") {
            opts.list_dispatches = true;
        } else if (arg == "--start-dispatch" && i + 1 < argc) {
            opts.has_start_dispatch = true;
            opts.start_dispatch = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--end-dispatch" && i + 1 < argc) {
            opts.has_end_dispatch = true;
            opts.end_dispatch = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--start-call" && i + 1 < argc) {
            opts.has_start_call = true;
            opts.start_call = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--end-call" && i + 1 < argc) {
            opts.has_end_call = true;
            opts.end_call = std::strtoull(argv[++i], nullptr, 10);
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
    if (!opts.list_dispatches && opts.output_path.empty()) {
        std::cerr << "Error: missing output trace file (-o <file>) or --list-dispatches.\n";
        return false;
    }
    return true;
}

bool is_creation_or_setup_event(const std::string &name)
{
    static const std::unordered_set<std::string> kSetupEvents = {
        "clCreateContext",
        "clCreateContextFromType",
        "clCreateCommandQueue",
        "clCreateCommandQueueWithProperties",
        "clCreateCommandQueue-properties",
        "clCreateCommandQueue-properties-not-found",
        "clCreateProgramWithSource",
        "clCreateProgramWithSource-args",
        "clCreateProgramWithIL",
        "clCreateProgramWithIL-args",
        "clCreateProgramWithBinary",
        "clCreateProgramWithBinary-args",
        "clBuildProgram",
        "clCompileProgram",
        "clLinkProgram",
        "clSetProgramSpecializationConstant",
        "clCreateKernel",
        "clCreateKernelsInProgram",
        "clCreateKernelsInProgram-kernel",
        "clCloneKernel",
        "clSetKernelArg",
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
        "clCommandBarrierWithWaitListKHR",
        "clCommandCopyBufferKHR",
        "clCommandCopyBufferRectKHR",
        "clCommandCopyBufferToImageKHR",
        "clCommandCopyImageKHR",
        "clCommandCopyImageToBufferKHR",
        "clCommandFillBufferKHR",
        "clCommandFillImageKHR",
        "clCommandNDRangeKernelKHR",
        "clUpdateMutableCommandsKHR",
        "clUpdateMutableCommandsKHR-config",
        "clUpdateMutableCommandsKHR-arg",
    };
    return kSetupEvents.count(name) > 0 || name.rfind("clkp-queue_", 0) == 0;
}

void add_uint_annotation(clkp::ParsedEvent &ev, const char *name, uint64_t val)
{
    clkp::ParsedAnnotation ann;
    ann.name = name;
    ann.type = clkp::ParsedAnnotation::ValueType::kUint;
    ann.uint_val = val;
    ev.annotations.push_back(std::move(ann));
}

void handle_slice_begin(
    clkp::ParsedEvent &&ev, std::vector<SliceContext> &stack, std::vector<clkp::ParsedEvent> &clkp_events)
{
    bool is_clkp = ev.has_category(CLKP_PERFETTO_CATEGORY);
    SliceContext ctx;
    ctx.is_clkp = is_clkp;
    ctx.call_id = ev.get_uint("call_id", 0);
    ctx.dispatch_id = ev.get_uint("dispatch_id", UINT64_MAX);
    ctx.cmdbuf_id = ev.get_uint("cmdbuf_id", 0);
    ctx.begin_timestamp = ev.timestamp;
    ctx.name = ev.name;
    if (is_clkp) {
        ctx.begin_event_index = clkp_events.size();
        clkp_events.push_back(std::move(ev));
    }
    stack.push_back(std::move(ctx));
}

void handle_slice_end(
    clkp::ParsedEvent &&ev, std::vector<SliceContext> &stack, std::vector<clkp::ParsedEvent> &clkp_events)
{
    if (stack.empty()) {
        if (ev.has_category(CLKP_PERFETTO_CATEGORY)) {
            clkp_events.push_back(std::move(ev));
        }
        return;
    }

    SliceContext top = stack.back();
    stack.pop_back();
    if (!top.is_clkp) {
        return;
    }

    if (top.call_id != 0) {
        add_uint_annotation(ev, "call_id", top.call_id);
    }
    if (top.dispatch_id != UINT64_MAX) {
        add_uint_annotation(ev, "dispatch_id", top.dispatch_id);
    }
    ev.name = top.name;
    clkp_events.push_back(std::move(ev));
}

void handle_instant_event(
    clkp::ParsedEvent &&ev, const std::vector<SliceContext> &stack, std::vector<clkp::ParsedEvent> &clkp_events)
{
    if (!ev.has_category(CLKP_PERFETTO_CATEGORY)) {
        return;
    }
    if (!ev.has_key("call_id")) {
        for (auto rit = stack.rbegin(); rit != stack.rend(); ++rit) {
            if (rit->is_clkp && rit->call_id != 0) {
                add_uint_annotation(ev, "call_id", rit->call_id);
                break;
            }
        }
    }
    clkp_events.push_back(std::move(ev));
}

void parse_trace_packets(const std::vector<uint8_t> &trace_bytes, std::vector<clkp::ParsedEvent> &clkp_events,
    TrackDescriptorMap &track_descriptors)
{
    clkp::TraceIncrementState inc_state;
    std::map<SequenceTrackKey, std::vector<SliceContext>> open_slices;

    perfetto::protos::pbzero::Trace_Decoder trace_dec(trace_bytes.data(), trace_bytes.size());
    size_t packet_idx = 0;
    for (auto pkt_it = trace_dec.packet(); pkt_it; ++pkt_it, ++packet_idx) {
        protozero::ConstBytes pkt_bytes = *pkt_it;
        perfetto::protos::pbzero::TracePacket_Decoder pkt_dec(pkt_bytes.data, pkt_bytes.size);

        inc_state.update_from_packet(pkt_dec);

        if (pkt_dec.has_track_descriptor()) {
            perfetto::protos::pbzero::TrackDescriptor_Decoder td(pkt_dec.track_descriptor());
            if (td.has_uuid()) {
                track_descriptors[td.uuid()] = { pkt_bytes.data, pkt_bytes.size };
            }
        }

        clkp::ParsedEvent ev;
        if (!inc_state.decode_track_event(pkt_dec, packet_idx, ev)) {
            continue;
        }

        auto &stack = open_slices[{ ev.sequence_id, ev.track_uuid }];
        if (ev.event_type == 1) {
            handle_slice_begin(std::move(ev), stack, clkp_events);
        } else if (ev.event_type == 2) {
            handle_slice_end(std::move(ev), stack, clkp_events);
        } else if (ev.event_type == 3) {
            handle_instant_event(std::move(ev), stack, clkp_events);
        }
    }
}

bool is_dispatch_api(const std::string &name)
{
    return name == "clEnqueueNDRangeKernel" || name == "clCommandNDRangeKernelKHR"
        || name == "clEnqueueCommandBufferKHR";
}

void record_dispatch_begin(const clkp::ParsedEvent &ev, std::map<uint64_t, DispatchSummary> &dispatches,
    std::map<SequenceTrackKey, std::pair<uint64_t, uint64_t>> &gpu_slice_starts)
{
    uint64_t did = ev.get_uint("dispatch_id", UINT64_MAX);
    if (did == UINT64_MAX) {
        return;
    }

    if (!is_dispatch_api(ev.name)) {
        gpu_slice_starts[{ ev.sequence_id, ev.track_uuid }] = { did, ev.timestamp };
        return;
    }

    auto &d = dispatches[did];
    d.dispatch_id = did;
    d.call_id = ev.get_uint("call_id", 0);
    d.api_name = ev.name;
    d.program = ev.get_string("program", "");
    d.kernel_name = ev.get_string("kernel_name", "");
    d.cmdbuf_id = ev.get_uint("cmdbuf_id", 0);
    d.gws = ev.get_string("gws", "");
    if (d.gws.empty() && ev.has_key("gidX")) {
        d.gws = std::to_string(ev.get_uint("gidX", 1)) + "," + std::to_string(ev.get_uint("gidY", 1)) + ","
            + std::to_string(ev.get_uint("gidZ", 1));
    }
    d.lws = ev.get_string("lws", "");
}

void record_dispatch_end(const clkp::ParsedEvent &ev, std::map<uint64_t, DispatchSummary> &dispatches,
    std::map<SequenceTrackKey, std::pair<uint64_t, uint64_t>> &gpu_slice_starts)
{
    auto it = gpu_slice_starts.find({ ev.sequence_id, ev.track_uuid });
    if (it == gpu_slice_starts.end()) {
        return;
    }
    uint64_t did = it->second.first;
    uint64_t start_ts = it->second.second;
    if (ev.timestamp >= start_ts && dispatches.count(did)) {
        dispatches[did].gpu_duration_ns = ev.timestamp - start_ts;
        dispatches[did].has_gpu_duration = true;
    }
    gpu_slice_starts.erase(it);
}

std::map<uint64_t, DispatchSummary> build_dispatch_summaries(const std::vector<clkp::ParsedEvent> &clkp_events)
{
    std::map<uint64_t, DispatchSummary> dispatches;
    std::map<SequenceTrackKey, std::pair<uint64_t, uint64_t>> gpu_slice_starts;

    for (const auto &ev : clkp_events) {
        if (ev.event_type == 1) {
            record_dispatch_begin(ev, dispatches, gpu_slice_starts);
        } else if (ev.event_type == 2) {
            record_dispatch_end(ev, dispatches, gpu_slice_starts);
        }
    }
    return dispatches;
}

void print_dispatch_table(const std::map<uint64_t, DispatchSummary> &dispatches)
{
    std::cout << std::left << std::setw(12) << "DISPATCH_ID" << std::setw(10) << "CALL_ID" << std::setw(28) << "API"
              << std::setw(12) << "PROGRAM" << std::setw(24) << "KERNEL/CMDBUF" << std::setw(16) << "GWS"
              << std::setw(16) << "LWS" << "GPU_TIME_US\n";
    for (const auto &[did, d] : dispatches) {
        std::string target = !d.kernel_name.empty() ? d.kernel_name : ("cmdbuf_" + std::to_string(d.cmdbuf_id));
        std::cout << std::left << std::setw(12) << d.dispatch_id << std::setw(10) << d.call_id << std::setw(28)
                  << d.api_name << std::setw(12) << d.program << std::setw(24) << target << std::setw(16) << d.gws
                  << std::setw(16) << (d.lws.empty() ? "NULL" : d.lws);
        if (d.has_gpu_duration) {
            std::cout << std::fixed << std::setprecision(2) << (static_cast<double>(d.gpu_duration_ns) / 1000.0);
        } else {
            std::cout << "N/A";
        }
        std::cout << "\n";
    }
}

bool resolve_call_window(const ExtractorOptions &opts, const std::map<uint64_t, DispatchSummary> &dispatches,
    uint64_t &window_start_call, uint64_t &window_end_call)
{
    window_start_call = opts.has_start_call ? opts.start_call : 0;
    window_end_call = opts.has_end_call ? opts.end_call : UINT64_MAX;

    if (!opts.has_start_dispatch && !opts.has_end_dispatch) {
        return true;
    }

    uint64_t min_call = UINT64_MAX;
    uint64_t max_call = 0;
    for (const auto &[did, d] : dispatches) {
        if (did >= opts.start_dispatch && did <= opts.end_dispatch) {
            min_call = std::min(min_call, d.call_id);
            max_call = std::max(max_call, d.call_id);
        }
    }
    if (min_call == UINT64_MAX) {
        std::cerr << "Error: no dispatches found in requested dispatch range.\n";
        return false;
    }
    if (!opts.has_start_call) {
        window_start_call = min_call;
    }
    if (!opts.has_end_call) {
        window_end_call = max_call;
    }
    return true;
}

struct DependencyClosure {
    std::unordered_set<uint64_t> contexts;
    std::unordered_set<uint64_t> queues;
    std::unordered_set<uint64_t> programs;
    std::unordered_set<uint64_t> kernels;
    std::unordered_set<uint64_t> mems;
    std::unordered_set<uint64_t> samplers;
    std::unordered_set<uint64_t> cmdbufs;
    std::unordered_set<uint64_t> kept_call_ids;

    static void insert_if_nonzero(std::unordered_set<uint64_t> &set, uint64_t id)
    {
        if (id != 0) {
            set.insert(id);
        }
    }

    static void insert_list_if_nonzero(std::unordered_set<uint64_t> &set, const std::string &csv)
    {
        for (uint64_t id : clkp::string_to_uint64_list(csv)) {
            insert_if_nonzero(set, id);
        }
    }

    void collect_refs_from_event(const clkp::ParsedEvent &ev)
    {
        insert_if_nonzero(contexts, ev.get_uint("context_id", 0));
        insert_if_nonzero(queues, ev.get_uint("queue_id", 0));
        insert_list_if_nonzero(queues, ev.get_string("queue_ids"));

        insert_if_nonzero(programs, ev.get_uint("program_id", 0));
        insert_list_if_nonzero(programs, ev.get_string("header_program_ids"));
        insert_list_if_nonzero(programs, ev.get_string("input_program_ids"));

        insert_if_nonzero(kernels, ev.get_uint("kernel_id", 0));
        insert_if_nonzero(kernels, ev.get_uint("src_kernel_id", 0));

        insert_if_nonzero(mems, ev.get_uint("mem_id", 0));
        insert_if_nonzero(mems, ev.get_uint("src_mem_id", 0));
        insert_if_nonzero(mems, ev.get_uint("dst_mem_id", 0));
        insert_if_nonzero(mems, ev.get_uint("parent_mem_id", 0));
        insert_list_if_nonzero(mems, ev.get_string("mem_ids"));

        insert_if_nonzero(samplers, ev.get_uint("sampler_id", 0));
        insert_if_nonzero(cmdbufs, ev.get_uint("cmdbuf_id", 0));
    }

    bool matches_needed_resource(const clkp::ParsedEvent &ev) const
    {
        if (ev.has_key("cmdbuf_id") && cmdbufs.count(ev.get_uint("cmdbuf_id")) > 0) {
            return true;
        }
        if (ev.has_key("kernel_id") && kernels.count(ev.get_uint("kernel_id")) > 0) {
            return true;
        }
        if (ev.has_key("program_id") && programs.count(ev.get_uint("program_id")) > 0) {
            return true;
        }
        bool is_arg_set = (ev.name == "clSetKernelArg" || ev.name == "clUpdateMutableCommandsKHR-arg");
        if (!is_arg_set && ev.has_key("mem_id") && mems.count(ev.get_uint("mem_id")) > 0) {
            return true;
        }
        if (!is_arg_set && ev.has_key("sampler_id") && samplers.count(ev.get_uint("sampler_id")) > 0) {
            return true;
        }
        bool is_queue_setup = (ev.name == "clCreateCommandQueue" || ev.name == "clCreateCommandQueueWithProperties"
            || ev.name.rfind("clkp-queue_", 0) == 0);
        if (is_queue_setup && ev.has_key("queue_id") && queues.count(ev.get_uint("queue_id")) > 0) {
            return true;
        }
        bool is_ctx_setup = (ev.name == "clCreateContext" || ev.name == "clCreateContextFromType");
        if (is_ctx_setup && ev.has_key("context_id") && contexts.count(ev.get_uint("context_id")) > 0) {
            return true;
        }
        return false;
    }
};

std::vector<bool> compute_retained_events(const std::vector<clkp::ParsedEvent> &clkp_events, bool has_window,
    uint64_t window_start_call, uint64_t window_end_call)
{
    std::vector<bool> keep_event(clkp_events.size(), !has_window);
    if (!has_window) {
        return keep_event;
    }

    DependencyClosure closure;
    for (size_t i = 0; i < clkp_events.size(); ++i) {
        const auto &ev = clkp_events[i];
        uint64_t cid = ev.get_uint("call_id", 0);
        if (cid != 0 && cid >= window_start_call && cid <= window_end_call) {
            keep_event[i] = true;
            closure.kept_call_ids.insert(cid);
            closure.collect_refs_from_event(ev);
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < clkp_events.size(); ++i) {
            if (keep_event[i]) {
                continue;
            }
            const auto &ev = clkp_events[i];
            uint64_t cid = ev.get_uint("call_id", 0);
            if (cid > window_end_call) {
                continue;
            }
            bool keep_by_call = (cid != 0 && closure.kept_call_ids.count(cid) > 0);
            bool keep_by_resource = is_creation_or_setup_event(ev.name) && closure.matches_needed_resource(ev);
            if (keep_by_call || keep_by_resource) {
                keep_event[i] = true;
                if (cid != 0) {
                    closure.kept_call_ids.insert(cid);
                }
                closure.collect_refs_from_event(ev);
                changed = true;
            }
        }
    }

    return keep_event;
}

void serialize_annotation(const clkp::ParsedAnnotation &ann, perfetto::protos::pbzero::DebugAnnotation *da)
{
    da->set_name(ann.name.c_str());
    switch (ann.type) {
    case clkp::ParsedAnnotation::ValueType::kString:
        da->set_string_value(ann.string_val.c_str());
        break;
    case clkp::ParsedAnnotation::ValueType::kUint:
        da->set_uint_value(ann.uint_val);
        break;
    case clkp::ParsedAnnotation::ValueType::kInt:
        da->set_int_value(ann.int_val);
        break;
    case clkp::ParsedAnnotation::ValueType::kPointer:
        da->set_pointer_value(ann.uint_val);
        break;
    case clkp::ParsedAnnotation::ValueType::kBool:
        da->set_bool_value(ann.bool_val);
        break;
    case clkp::ParsedAnnotation::ValueType::kDouble:
        da->set_double_value(ann.double_val);
        break;
    default:
        break;
    }
}

void serialize_track_event(const clkp::ParsedEvent &ev, perfetto::protos::pbzero::TracePacket *pkt)
{
    pkt->set_timestamp(ev.timestamp);
    pkt->set_trusted_packet_sequence_id(ev.sequence_id != 0 ? ev.sequence_id : 1);

    auto *te = pkt->set_track_event();
    te->set_type(static_cast<perfetto::protos::pbzero::TrackEvent::Type>(ev.event_type));
    if (ev.track_uuid != 0) {
        te->set_track_uuid(ev.track_uuid);
    }
    if (ev.event_type == 2) {
        return;
    }

    if (ev.categories.empty()) {
        te->add_categories(CLKP_PERFETTO_CATEGORY);
    } else {
        for (const auto &cat : ev.categories) {
            te->add_categories(cat.c_str());
        }
    }
    if (!ev.name.empty()) {
        te->set_name(ev.name.c_str());
    }
    for (const auto &ann : ev.annotations) {
        serialize_annotation(ann, te->add_debug_annotations());
    }
}

bool write_filtered_trace(const std::string &output_path, const std::vector<clkp::ParsedEvent> &clkp_events,
    const std::vector<bool> &keep_event, const TrackDescriptorMap &track_descriptors)
{
    protozero::HeapBuffered<perfetto::protos::pbzero::Trace> out_trace;
    std::unordered_set<uint64_t> emitted_tracks;

    for (size_t i = 0; i < clkp_events.size(); ++i) {
        if (!keep_event[i]) {
            continue;
        }
        const auto &ev = clkp_events[i];
        if (ev.track_uuid != 0 && emitted_tracks.insert(ev.track_uuid).second) {
            auto td_it = track_descriptors.find(ev.track_uuid);
            if (td_it != track_descriptors.end()) {
                auto *raw_pkt = out_trace->add_packet();
                raw_pkt->AppendRawProtoBytes(td_it->second.first, td_it->second.second);
            }
        }
        serialize_track_event(ev, out_trace->add_packet());
    }

    std::vector<uint8_t> serialized = out_trace.SerializeAsArray();
    std::ofstream out_file(output_path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out_file.is_open()) {
        std::cerr << "Error: failed to open output file '" << output_path << "'\n";
        return false;
    }
    out_file.write(reinterpret_cast<const char *>(serialized.data()), serialized.size());
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    ExtractorOptions opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<uint8_t> trace_bytes;
    if (!clkp::read_file_bytes(opts.input_path, trace_bytes)) {
        std::cerr << "Error: failed to read input file '" << opts.input_path << "'\n";
        return 1;
    }

    std::vector<clkp::ParsedEvent> clkp_events;
    TrackDescriptorMap track_descriptors;
    parse_trace_packets(trace_bytes, clkp_events, track_descriptors);

    auto dispatches = build_dispatch_summaries(clkp_events);
    if (opts.list_dispatches) {
        print_dispatch_table(dispatches);
        if (opts.output_path.empty()) {
            return 0;
        }
    }

    uint64_t window_start_call = 0;
    uint64_t window_end_call = UINT64_MAX;
    if (!resolve_call_window(opts, dispatches, window_start_call, window_end_call)) {
        return 1;
    }

    auto keep_event = compute_retained_events(clkp_events, opts.has_window(), window_start_call, window_end_call);
    if (!write_filtered_trace(opts.output_path, clkp_events, keep_event, track_descriptors)) {
        return 1;
    }

    return 0;
}
