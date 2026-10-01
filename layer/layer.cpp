// Copyright 2023 The OpenCL Kernel Profiler authors.
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

#include <CL/cl_ext.h>
#include <CL/cl_layer.h>
#include <assert.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#ifdef SPIRV_DISASSEMBLY
#include <spirv-tools/libspirv.hpp>
#endif

/*****************************************************************************/
/* PERFETTO GLOBAL VARIABLES *************************************************/
/*****************************************************************************/

PERFETTO_DEFINE_CATEGORIES(perfetto::Category(CLKP_PERFETTO_CATEGORY).SetDescription("OpenCL Kernel Profiler Events"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

#ifdef BACKEND_INPROCESS
static perfetto::TracingSession *gTracingSession;
#endif

/*****************************************************************************/
/* ICD DISPATCH GLOBAL VARIABLES *********************************************/
/*****************************************************************************/

static struct _cl_icd_dispatch dispatch;
static const struct _cl_icd_dispatch *tdispatch;

/*****************************************************************************/
/* MACROS ********************************************************************/
/*****************************************************************************/

#ifdef __ANDROID__
#define PRINT(message, ...)                                                                                            \
    do {                                                                                                               \
        __android_log_print(                                                                                           \
            android_LogPriority::ANDROID_LOG_ERROR, "CLKP", " %s: " message "\n", __func__, ##__VA_ARGS__);            \
    } while (0)
#else
#define PRINT(message, ...)                                                                                            \
    do {                                                                                                               \
        fprintf(stderr, "[CLKP] %s: " message "\n", __func__, ##__VA_ARGS__);                                          \
    } while (0)
#endif
#define CHECK(test, statement, message, ...)                                                                           \
    do {                                                                                                               \
        if (!(test)) {                                                                                                 \
            PRINT(message, ##__VA_ARGS__);                                                                             \
            statement;                                                                                                 \
        }                                                                                                              \
    } while (0)
#define CHECK_CL(err, statement, message, ...) CHECK((err) == CL_SUCCESS, statement, message, ##__VA_ARGS__)
#define CHECK_ALLOC(ptr, statement) CHECK(ptr != nullptr, statement, "allocation failed")

/*****************************************************************************/
/* STATE TRACKING & SYNTHETIC IDS ********************************************/
/*****************************************************************************/

static std::recursive_mutex g_lock;
static std::atomic<uint64_t> g_next_call_id { 1 };
static std::atomic<uint64_t> g_next_dispatch_id { 0 };

static uint64_t next_call_id() { return g_next_call_id.fetch_add(1, std::memory_order_relaxed); }
static uint64_t next_dispatch_id() { return g_next_dispatch_id.fetch_add(1, std::memory_order_relaxed); }

template <typename HandleT> class HandleIdTracker {
public:
    uint64_t allocate_id() { return next_id_++; }

    void bind(HandleT handle, uint64_t id)
    {
        if (handle != nullptr && id != 0) {
            map_[handle] = id;
        }
    }

    uint64_t get_or_create(HandleT handle)
    {
        if (handle == nullptr) {
            return 0;
        }
        auto it = map_.find(handle);
        if (it != map_.end()) {
            return it->second;
        }
        uint64_t id = next_id_++;
        map_[handle] = id;
        return id;
    }

    uint64_t create_fresh(HandleT handle)
    {
        if (handle == nullptr) {
            return 0;
        }
        uint64_t id = next_id_++;
        map_[handle] = id;
        return id;
    }

    uint64_t find(HandleT handle) const
    {
        if (handle == nullptr) {
            return 0;
        }
        auto it = map_.find(handle);
        return (it != map_.end()) ? it->second : 0;
    }

    bool contains(HandleT handle) const
    {
        if (handle == nullptr) {
            return false;
        }
        return map_.find(handle) != map_.end();
    }

private:
    std::unordered_map<HandleT, uint64_t> map_;
    uint64_t next_id_ = 1;
};

static HandleIdTracker<cl_context> g_context_ids;
static HandleIdTracker<cl_command_queue> g_queue_ids;
static HandleIdTracker<cl_program> g_program_ids;
static HandleIdTracker<cl_kernel> g_kernel_ids;
static HandleIdTracker<cl_mem> g_mem_ids;
static HandleIdTracker<cl_sampler> g_sampler_ids;
static HandleIdTracker<cl_event> g_event_ids;
static HandleIdTracker<void *> g_map_ids;
static HandleIdTracker<cl_command_buffer_khr> g_cmdbuf_ids;
static HandleIdTracker<cl_mutable_command_khr> g_mutable_cmd_ids;

static std::map<cl_program, std::string> program_to_string;
static std::map<cl_program, uint64_t> program_to_context_id;
static std::map<cl_kernel, std::string> kernel_to_kernel_name;
static std::map<cl_kernel, cl_program> kernel_to_program;
static std::map<cl_command_buffer_khr, std::vector<cl_command_queue>> cmdbuf_to_queues;
static std::map<cl_command_buffer_khr, uint64_t> cmdbuf_to_context_id;
static std::map<cl_command_queue, uint64_t> queue_to_context_id;

static std::string size_array_to_string(const size_t *arr, cl_uint count)
{
    if (arr == nullptr || count == 0) {
        return "";
    }
    std::vector<uint64_t> vals(count);
    for (cl_uint i = 0; i < count; ++i) {
        vals[i] = static_cast<uint64_t>(arr[i]);
    }
    return clkp::uint64_list_to_string(vals);
}

static std::string event_list_to_string(cl_uint num_events, const cl_event *event_list)
{
    if (event_list == nullptr || num_events == 0) {
        return "";
    }
    std::vector<uint64_t> ids;
    ids.reserve(num_events);
    for (cl_uint i = 0; i < num_events; ++i) {
        ids.push_back(g_event_ids.get_or_create(event_list[i]));
    }
    return clkp::uint64_list_to_string(ids);
}

static std::string sync_point_list_to_string(cl_uint num_sync_points, const cl_sync_point_khr *sync_points)
{
    if (sync_points == nullptr || num_sync_points == 0) {
        return "";
    }
    std::vector<uint64_t> ids(num_sync_points);
    for (cl_uint i = 0; i < num_sync_points; ++i) {
        ids[i] = static_cast<uint64_t>(sync_points[i]);
    }
    return clkp::uint64_list_to_string(ids);
}

template <typename PropT> static std::string properties_to_string(const PropT *props)
{
    if (props == nullptr) {
        return "";
    }
    std::vector<uint64_t> vals;
    for (size_t i = 0; props[i] != 0; i += 2) {
        vals.push_back(static_cast<uint64_t>(props[i]));
        vals.push_back(static_cast<uint64_t>(props[i + 1]));
    }
    return clkp::uint64_list_to_string(vals);
}

/*****************************************************************************/
/* CONTEXT LIFECYCLE *********************************************************/
/*****************************************************************************/

static cl_context clkp_clCreateContext(const cl_context_properties *properties, cl_uint num_devices,
    const cl_device_id *devices,
    void(CL_CALLBACK *pfn_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data),
    void *user_data, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.allocate_id();
    }
    TRACE_EVENT(
        CLKP_PERFETTO_CATEGORY, "clCreateContext", "call_id", cid, "context_id", ctx_id, "num_devices", num_devices);
    cl_context ctx = tdispatch->clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_context_ids.bind(ctx, ctx_id);
    }
    return ctx;
}

static cl_context clkp_clCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type,
    void(CL_CALLBACK *pfn_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data),
    void *user_data, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.allocate_id();
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateContextFromType", "call_id", cid, "context_id", ctx_id, "device_type",
        static_cast<uint64_t>(device_type));
    cl_context ctx = tdispatch->clCreateContextFromType(properties, device_type, pfn_notify, user_data, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_context_ids.bind(ctx, ctx_id);
    }
    return ctx;
}

static cl_int clkp_clRetainContext(cl_context context)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clRetainContext", "call_id", cid, "context_id", ctx_id);
    return tdispatch->clRetainContext(context);
}

static cl_int clkp_clReleaseContext(cl_context context)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.find(context);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clReleaseContext", "call_id", cid, "context_id", ctx_id);
    return tdispatch->clReleaseContext(context);
}

/*****************************************************************************/
/* CREATE KERNEL & PROGRAM ***************************************************/
/*****************************************************************************/

static void writeProgramOnDisk(
    std::filesystem::path filename, cl_uint count, const char **data, const size_t *lengths, bool binary)
{
    auto dirname = filename.parent_path();
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "writeProgramOnDisk", "dir",
        perfetto::DynamicString((const char *)dirname.u8string().c_str()), "program",
        perfetto::DynamicString((const char *)filename.filename().u8string().c_str()));
    CHECK(std::filesystem::exists(dirname), return;
          , "'%s' does not exist, could not write program on disk", dirname.c_str());
    std::ofstream file(filename, binary ? std::ios::binary | std::ios::trunc : std::ios::trunc);
    CHECK(file.is_open(), return;, "Could not open '%s'", filename.c_str());
    for (unsigned i = 0; i < count; i++) {
        size_t length = (lengths == nullptr || lengths[i] == 0) ? strlen(data[i]) : lengths[i];
        file.write(data[i], length);
    }
}

#ifdef SPIRV_DISASSEMBLY
static bool tryDisassembleSpirv(
    const void *il, size_t length, const std::string &program_name, std::string &disassembly)
{
    const uint32_t *spirv_data = (const uint32_t *)(il);
    size_t spirv_words = length / sizeof(uint32_t);

    // Check for SPIR-V magic number
    if (spirv_words == 0 || *spirv_data != 0x07230203) {
        return false;
    }

    spvtools::SpirvTools tools(SPV_ENV_OPENCL_2_2);
    CHECK(tools.Disassemble(spirv_data, spirv_words, &disassembly), return false;
          , "Failed to disassemble SPIR-V for program %s", program_name.c_str());
    return true;
}
#endif

static std::string get_program_str(uint64_t *out_num = nullptr)
{
    static uint32_t program_number = 0;
    uint32_t num = program_number++;
    if (out_num) {
        *out_num = num + 1;
    }
    return std::string("clkp_p") + std::to_string(num);
}

static char *get_kernel_dir() { return getenv("CLKP_KERNEL_DIR"); }

static bool is_program_binary_cache_disabled()
{
    const char *env = getenv("CLKP_DISABLE_PROGRAM_BINARY_CACHE");
    return env != nullptr && strcmp(env, "0") != 0 && strcmp(env, "") != 0;
}

static cl_program clkp_clCreateProgramWithSource(
    cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t prog_id = 0;
    std::string program_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        prog_id = g_program_ids.allocate_id();
        program_str = get_program_str();
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithSource", "call_id", cid, "context_id", ctx_id, "program_id",
        prog_id, "program", perfetto::DynamicString(program_str), "count", count);

    if (auto dir = get_kernel_dir()) {
        writeProgramOnDisk(std::filesystem::path(dir) / (program_str + ".cl"), count, strings, lengths, false);
    }

    cl_program program = tdispatch->clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_program_ids.bind(program, prog_id);
        program_to_string[program] = program_str;
        program_to_context_id[program] = ctx_id;
    }

    if (TRACE_EVENT_CATEGORY_ENABLED(CLKP_PERFETTO_CATEGORY)) {
        for (unsigned i = 0; i < count; i++) {
            if (strings == nullptr || strings[i] == nullptr) {
                continue;
            }
            size_t len = (lengths == nullptr || lengths[i] == 0) ? strlen(strings[i]) : lengths[i];
            std::string full_str(strings[i], len);
            size_t chunk_size = clkp::kMaxPayloadChunkBytes;
            size_t num_chunks = (full_str.size() + chunk_size - 1) / chunk_size;
            if (num_chunks == 0) {
                num_chunks = 1;
            }
            for (size_t c = 0; c < num_chunks; ++c) {
                std::string chunk = full_str.substr(c * chunk_size, chunk_size);
                TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithSource-args", "call_id", cid,
                    "program_id", prog_id, "program", perfetto::DynamicString(program_str), "string_index", i,
                    "chunk_index", c, "chunk_count", num_chunks, "string", perfetto::DynamicString(chunk));
            }
        }
    }
    return program;
}

static cl_program clkp_clCreateProgramWithBinary(cl_context context, cl_uint num_devices,
    const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status,
    cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t prog_id = 0;
    std::string program_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        prog_id = g_program_ids.allocate_id();
        program_str = get_program_str();
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithBinary", "call_id", cid, "context_id", ctx_id, "program_id",
        prog_id, "program", perfetto::DynamicString(program_str), "num_devices", num_devices);

    if (auto dir = get_kernel_dir()) {
        for (unsigned i = 0; i < num_devices; i++) {
            writeProgramOnDisk(std::filesystem::path(dir) / (program_str + "-" + std::to_string(i) + ".bin"), 1,
                (const char **)&binaries[i], &lengths[i], true);
        }
    }
    cl_program program = tdispatch->clCreateProgramWithBinary(
        context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_program_ids.bind(program, prog_id);
        program_to_string[program] = program_str;
        program_to_context_id[program] = ctx_id;
    }

    if (TRACE_EVENT_CATEGORY_ENABLED(CLKP_PERFETTO_CATEGORY) && binaries != nullptr && lengths != nullptr) {
        for (unsigned i = 0; i < num_devices; ++i) {
            if (binaries[i] == nullptr || lengths[i] == 0) {
                continue;
            }
            std::string hex = clkp::bytes_to_hex(binaries[i], lengths[i]);
            size_t chunk_size = clkp::kMaxPayloadChunkBytes;
            size_t num_chunks = (hex.size() + chunk_size - 1) / chunk_size;
            for (size_t c = 0; c < num_chunks; ++c) {
                std::string chunk = hex.substr(c * chunk_size, chunk_size);
                TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithBinary-args", "call_id", cid,
                    "program_id", prog_id, "program", perfetto::DynamicString(program_str), "device_index", i,
                    "chunk_index", c, "chunk_count", num_chunks, "binary_hex", perfetto::DynamicString(chunk));
            }
        }
    }
    return program;
}

static cl_program create_program_with_il_common(
    cl_context context, const void *il, size_t length, cl_int *errcode_ret, clCreateProgramWithILKHR_fn ext_fn)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t prog_id = 0;
    std::string program_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        prog_id = g_program_ids.allocate_id();
        program_str = get_program_str();
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithIL", "call_id", cid, "context_id", ctx_id, "program_id",
        prog_id, "program", perfetto::DynamicString(program_str.c_str()), "length", length);

    std::string disassembly;
    bool is_spirv = false;
#ifdef SPIRV_DISASSEMBLY
    is_spirv = tryDisassembleSpirv(il, length, program_str, disassembly);
#endif

    if (auto dir = get_kernel_dir()) {
        std::filesystem::path filename = std::filesystem::path(dir) / program_str;

        // Write the raw IL binary
        std::filesystem::path il_filename = filename.replace_extension(is_spirv ? "spv" : "il");
        const char *il_str = (const char *)il;
        writeProgramOnDisk(il_filename, 1, &il_str, &length, true);

        if (is_spirv) {
            // Write the disassembly
            std::filesystem::path asm_filename = filename.replace_extension("spvasm");
            const char *disassembly_str = disassembly.c_str();
            size_t disassembly_length = disassembly.length();
            writeProgramOnDisk(asm_filename, 1, &disassembly_str, &disassembly_length, false);
        }
    }

    cl_program program = ext_fn ? ext_fn(context, il, length, errcode_ret)
                                : tdispatch->clCreateProgramWithIL(context, il, length, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_program_ids.bind(program, prog_id);
        program_to_string[program] = program_str;
        program_to_context_id[program] = ctx_id;
    }

    if (TRACE_EVENT_CATEGORY_ENABLED(CLKP_PERFETTO_CATEGORY)) {
        std::string il_hex = clkp::bytes_to_hex(il, length);
        size_t chunk_size = clkp::kMaxPayloadChunkBytes;
        size_t num_chunks = (il_hex.size() + chunk_size - 1) / chunk_size;
        if (num_chunks == 0) {
            num_chunks = 1;
        }
        for (size_t c = 0; c < num_chunks; ++c) {
            std::string chunk = il_hex.empty() ? "" : il_hex.substr(c * chunk_size, chunk_size);
            if (c == 0 && is_spirv) {
                TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithIL-args", "call_id", cid, "program_id",
                    prog_id, "program", perfetto::DynamicString(program_str), "disassembly",
                    perfetto::DynamicString(disassembly), "chunk_index", c, "chunk_count", num_chunks, "il_hex",
                    perfetto::DynamicString(chunk));
            } else {
                TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithIL-args", "call_id", cid, "program_id",
                    prog_id, "program", perfetto::DynamicString(program_str), "chunk_index", c, "chunk_count",
                    num_chunks, "il_hex", perfetto::DynamicString(chunk));
            }
        }
    }

    return program;
}

static cl_program clkp_clCreateProgramWithIL(cl_context context, const void *il, size_t length, cl_int *errcode_ret)
{
    return create_program_with_il_common(context, il, length, errcode_ret, nullptr);
}

static cl_int clkp_clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret)
{
    if (param_name == CL_PROGRAM_BINARY_SIZES && is_program_binary_cache_disabled()) {
        cl_int err
            = tdispatch->clGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret);
        if (err == CL_SUCCESS && param_value != nullptr) {
            memset(param_value, 0, param_value_size);
        }
        return err;
    }
    return tdispatch->clGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret);
}

static cl_int clkp_clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data)
{
    uint64_t cid = 0;
    uint64_t prog_id = 0;
    std::string prog_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        prog_id = g_program_ids.get_or_create(program);
        prog_str = program_to_string.count(program) ? program_to_string[program] : "clkp_p?";
    }
    std::string opts = options ? options : "";
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clBuildProgram", "call_id", cid, "program_id", prog_id, "program",
        perfetto::DynamicString(prog_str), "options", perfetto::DynamicString(opts));
    return tdispatch->clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
}

static cl_int clkp_clCompileProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, cl_uint num_input_headers, const cl_program *input_headers, const char **header_include_names,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data)
{
    uint64_t cid = 0;
    uint64_t prog_id = 0;
    std::string prog_str;
    std::vector<uint64_t> header_ids;
    std::vector<std::string> header_names;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        prog_id = g_program_ids.get_or_create(program);
        prog_str = program_to_string.count(program) ? program_to_string[program] : "clkp_p?";
        for (cl_uint i = 0; i < num_input_headers; ++i) {
            if (input_headers) {
                header_ids.push_back(g_program_ids.get_or_create(input_headers[i]));
            }
            if (header_include_names && header_include_names[i]) {
                header_names.push_back(header_include_names[i]);
            }
        }
    }
    std::string opts = options ? options : "";
    std::string header_ids_str = clkp::uint64_list_to_string(header_ids);
    std::string header_names_str = clkp::string_list_to_joined(header_names, ';');
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCompileProgram", "call_id", cid, "program_id", prog_id, "program",
        perfetto::DynamicString(prog_str), "options", perfetto::DynamicString(opts), "num_input_headers",
        num_input_headers, "header_program_ids", perfetto::DynamicString(header_ids_str), "header_include_names",
        perfetto::DynamicString(header_names_str));
    return tdispatch->clCompileProgram(program, num_devices, device_list, options, num_input_headers, input_headers,
        header_include_names, pfn_notify, user_data);
}

static cl_program clkp_clLinkProgram(cl_context context, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, cl_uint num_input_programs, const cl_program *input_programs,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t prog_id = 0;
    std::string program_str;
    std::vector<uint64_t> input_ids;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        prog_id = g_program_ids.allocate_id();
        program_str = get_program_str();
        for (cl_uint i = 0; i < num_input_programs; ++i) {
            if (input_programs) {
                input_ids.push_back(g_program_ids.get_or_create(input_programs[i]));
            }
        }
    }
    std::string opts = options ? options : "";
    std::string input_ids_str = clkp::uint64_list_to_string(input_ids);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clLinkProgram", "call_id", cid, "context_id", ctx_id, "program_id", prog_id,
        "program", perfetto::DynamicString(program_str), "options", perfetto::DynamicString(opts), "num_input_programs",
        num_input_programs, "input_program_ids", perfetto::DynamicString(input_ids_str));
    cl_program program = tdispatch->clLinkProgram(context, num_devices, device_list, options, num_input_programs,
        input_programs, pfn_notify, user_data, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_program_ids.bind(program, prog_id);
        program_to_string[program] = program_str;
        program_to_context_id[program] = ctx_id;
    }
    return program;
}

static cl_int clkp_clSetProgramSpecializationConstant(
    cl_program program, cl_uint spec_id, size_t spec_size, const void *spec_value)
{
    uint64_t cid = 0;
    uint64_t prog_id = 0;
    std::string prog_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        prog_id = g_program_ids.get_or_create(program);
        prog_str = program_to_string.count(program) ? program_to_string[program] : "clkp_p?";
    }
    std::string val_hex = clkp::bytes_to_hex(spec_value, spec_size);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clSetProgramSpecializationConstant", "call_id", cid, "program_id", prog_id,
        "program", perfetto::DynamicString(prog_str), "spec_id", spec_id, "spec_size", spec_size, "value_hex",
        perfetto::DynamicString(val_hex));
    return tdispatch->clSetProgramSpecializationConstant(program, spec_id, spec_size, spec_value);
}

static cl_int clkp_clRetainProgram(cl_program program)
{
    uint64_t cid = 0;
    uint64_t prog_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        prog_id = g_program_ids.get_or_create(program);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clRetainProgram", "call_id", cid, "program_id", prog_id);
    return tdispatch->clRetainProgram(program);
}

static cl_int clkp_clReleaseProgram(cl_program program)
{
    uint64_t cid = 0;
    uint64_t prog_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        prog_id = g_program_ids.find(program);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clReleaseProgram", "call_id", cid, "program_id", prog_id);
    return tdispatch->clReleaseProgram(program);
}

static cl_kernel clkp_clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t kern_id = 0;
    uint64_t prog_id = 0;
    std::string prog_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        kern_id = g_kernel_ids.allocate_id();
        prog_id = g_program_ids.get_or_create(program);
        prog_str = program_to_string.count(program) ? program_to_string[program] : "clkp_p?";
    }
    std::string kname = kernel_name ? kernel_name : "";
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateKernel", "call_id", cid, "program_id", prog_id, "program",
        perfetto::DynamicString(prog_str), "kernel_id", kern_id, "kernel_name", perfetto::DynamicString(kname));
    cl_kernel kernel = tdispatch->clCreateKernel(program, kernel_name, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_kernel_ids.bind(kernel, kern_id);
        if (kernel != nullptr) {
            kernel_to_program[kernel] = program;
            kernel_to_kernel_name[kernel] = kname;
        }
    }
    return kernel;
}

static cl_int clkp_clCreateKernelsInProgram(
    cl_program program, cl_uint num_kernels, cl_kernel *kernels, cl_uint *num_kernels_ret)
{
    uint64_t cid = 0;
    uint64_t prog_id = 0;
    std::string prog_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        prog_id = g_program_ids.get_or_create(program);
        prog_str = program_to_string.count(program) ? program_to_string[program] : "clkp_p?";
    }
    cl_uint actual_kernels = 0;
    cl_int err = tdispatch->clCreateKernelsInProgram(program, num_kernels, kernels, &actual_kernels);
    if (num_kernels_ret != nullptr) {
        *num_kernels_ret = actual_kernels;
    }
    cl_uint created_count = (err == CL_SUCCESS && kernels != nullptr) ? std::min(num_kernels, actual_kernels) : 0;
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateKernelsInProgram", "call_id", cid, "program_id", prog_id, "program",
        perfetto::DynamicString(prog_str), "num_kernels_created", created_count);

    if (err == CL_SUCCESS && kernels != nullptr) {
        for (cl_uint i = 0; i < created_count; ++i) {
            cl_kernel k = kernels[i];
            char name_buf[256] = { 0 };
            size_t name_len = 0;
            if (tdispatch->clGetKernelInfo) {
                tdispatch->clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(name_buf) - 1, name_buf, &name_len);
            }
            std::string kname(name_buf);
            uint64_t kern_id = 0;
            {
                std::lock_guard<std::recursive_mutex> lock(g_lock);
                kern_id = g_kernel_ids.create_fresh(k);
                kernel_to_program[k] = program;
                kernel_to_kernel_name[k] = kname;
            }
            TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateKernelsInProgram-kernel", "call_id", cid, "program_id",
                prog_id, "program", perfetto::DynamicString(prog_str), "kernel_id", kern_id, "kernel_name",
                perfetto::DynamicString(kname));
        }
    }
    return err;
}

static cl_kernel clkp_clCloneKernel(cl_kernel source_kernel, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t src_id = 0;
    uint64_t dst_id = 0;
    cl_program prog = nullptr;
    uint64_t prog_id = 0;
    std::string prog_str;
    std::string kname;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        src_id = g_kernel_ids.get_or_create(source_kernel);
        dst_id = g_kernel_ids.allocate_id();
        prog = kernel_to_program.count(source_kernel) ? kernel_to_program[source_kernel] : nullptr;
        prog_id = g_program_ids.find(prog);
        prog_str = program_to_string.count(prog) ? program_to_string[prog] : "clkp_p?";
        kname = kernel_to_kernel_name.count(source_kernel) ? kernel_to_kernel_name[source_kernel] : "?";
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCloneKernel", "call_id", cid, "src_kernel_id", src_id, "kernel_id", dst_id,
        "program_id", prog_id, "program", perfetto::DynamicString(prog_str), "kernel_name",
        perfetto::DynamicString(kname));
    cl_kernel cloned = tdispatch->clCloneKernel(source_kernel, errcode_ret);
    if (cloned != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_kernel_ids.bind(cloned, dst_id);
        kernel_to_program[cloned] = prog;
        kernel_to_kernel_name[cloned] = kname;
    }
    return cloned;
}

static void classify_kernel_arg(size_t arg_size, const void *arg_value, std::string &out_kind, uint64_t &out_mem_id,
    uint64_t &out_sampler_id, std::string &out_val_hex)
{
    out_mem_id = 0;
    out_sampler_id = 0;
    out_val_hex.clear();

    if (arg_value == nullptr) {
        out_kind = "LOCAL";
        return;
    }
    if (arg_size == sizeof(cl_mem)) {
        cl_mem candidate_mem = *reinterpret_cast<const cl_mem *>(arg_value);
        if (candidate_mem != nullptr && g_mem_ids.contains(candidate_mem)) {
            out_kind = "MEM";
            out_mem_id = g_mem_ids.find(candidate_mem);
            return;
        }
    }
    if (arg_size == sizeof(cl_sampler)) {
        cl_sampler candidate_sampler = *reinterpret_cast<const cl_sampler *>(arg_value);
        if (candidate_sampler != nullptr && g_sampler_ids.contains(candidate_sampler)) {
            out_kind = "SAMPLER";
            out_sampler_id = g_sampler_ids.find(candidate_sampler);
            return;
        }
    }
    out_kind = "VALUE";
    out_val_hex = clkp::bytes_to_hex(arg_value, arg_size);
}

static cl_int clkp_clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)
{
    uint64_t cid = 0;
    uint64_t kern_id = 0;
    std::string arg_kind;
    uint64_t mem_id = 0;
    uint64_t sampler_id = 0;
    std::string val_hex;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        kern_id = g_kernel_ids.get_or_create(kernel);
        classify_kernel_arg(arg_size, arg_value, arg_kind, mem_id, sampler_id, val_hex);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clSetKernelArg", "call_id", cid, "kernel_id", kern_id, "arg_index", arg_index,
        "arg_size", arg_size, "arg_kind", perfetto::DynamicString(arg_kind), "mem_id", mem_id, "sampler_id", sampler_id,
        "value_hex", perfetto::DynamicString(val_hex));
    return tdispatch->clSetKernelArg(kernel, arg_index, arg_size, arg_value);
}

static cl_int clkp_clRetainKernel(cl_kernel kernel)
{
    uint64_t cid = 0;
    uint64_t kern_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        kern_id = g_kernel_ids.get_or_create(kernel);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clRetainKernel", "call_id", cid, "kernel_id", kern_id);
    return tdispatch->clRetainKernel(kernel);
}

static cl_int clkp_clReleaseKernel(cl_kernel kernel)
{
    uint64_t cid = 0;
    uint64_t kern_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        kern_id = g_kernel_ids.find(kernel);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clReleaseKernel", "call_id", cid, "kernel_id", kern_id);
    return tdispatch->clReleaseKernel(kernel);
}

/*****************************************************************************/
/* MEMORY & SAMPLER LIFECYCLE ************************************************/
/*****************************************************************************/

static cl_mem clkp_clCreateBuffer(
    cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t mem_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        mem_id = g_mem_ids.allocate_id();
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateBuffer", "call_id", cid, "context_id", ctx_id, "mem_id", mem_id,
        "flags", static_cast<uint64_t>(flags), "size", size);
    cl_mem mem = tdispatch->clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_mem_ids.bind(mem, mem_id);
    }
    return mem;
}

static cl_mem clkp_clCreateBufferWithProperties(cl_context context, const cl_mem_properties *properties,
    cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t mem_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        mem_id = g_mem_ids.allocate_id();
    }
    std::string props_str = properties_to_string(properties);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateBufferWithProperties", "call_id", cid, "context_id", ctx_id, "mem_id",
        mem_id, "flags", static_cast<uint64_t>(flags), "size", size, "properties", perfetto::DynamicString(props_str));
    cl_mem mem = tdispatch->clCreateBufferWithProperties(context, properties, flags, size, host_ptr, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_mem_ids.bind(mem, mem_id);
    }
    return mem;
}

static cl_mem clkp_clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type,
    const void *buffer_create_info, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t parent_id = 0;
    uint64_t mem_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        parent_id = g_mem_ids.get_or_create(buffer);
        mem_id = g_mem_ids.allocate_id();
    }
    size_t origin = 0;
    size_t size = 0;
    if (buffer_create_type == CL_BUFFER_CREATE_TYPE_REGION && buffer_create_info != nullptr) {
        const auto *reg = reinterpret_cast<const cl_buffer_region *>(buffer_create_info);
        origin = reg->origin;
        size = reg->size;
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateSubBuffer", "call_id", cid, "parent_mem_id", parent_id, "mem_id",
        mem_id, "flags", static_cast<uint64_t>(flags), "buffer_create_type", static_cast<uint64_t>(buffer_create_type),
        "origin", origin, "size", size);
    cl_mem sub = tdispatch->clCreateSubBuffer(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_mem_ids.bind(sub, mem_id);
    }
    return sub;
}

static cl_mem clkp_clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
    const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t mem_id = 0;
    uint64_t parent_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        mem_id = g_mem_ids.allocate_id();
        parent_id = (image_desc && image_desc->mem_object) ? g_mem_ids.get_or_create(image_desc->mem_object) : 0;
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateImage", "call_id", cid, "context_id", ctx_id, "mem_id", mem_id,
        "flags", static_cast<uint64_t>(flags), "channel_order", image_format ? image_format->image_channel_order : 0u,
        "channel_data_type", image_format ? image_format->image_channel_data_type : 0u, "image_type",
        image_desc ? image_desc->image_type : 0u, "width", image_desc ? image_desc->image_width : 0u, "height",
        image_desc ? image_desc->image_height : 0u, "depth", image_desc ? image_desc->image_depth : 0u, "array_size",
        image_desc ? image_desc->image_array_size : 0u, "row_pitch", image_desc ? image_desc->image_row_pitch : 0u,
        "slice_pitch", image_desc ? image_desc->image_slice_pitch : 0u, "num_mip_levels",
        image_desc ? image_desc->num_mip_levels : 0u, "num_samples", image_desc ? image_desc->num_samples : 0u,
        "parent_mem_id", parent_id);
    cl_mem img = tdispatch->clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_mem_ids.bind(img, mem_id);
    }
    return img;
}

static cl_mem clkp_clCreateImageWithProperties(cl_context context, const cl_mem_properties *properties,
    cl_mem_flags flags, const cl_image_format *image_format, const cl_image_desc *image_desc, void *host_ptr,
    cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t mem_id = 0;
    uint64_t parent_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        mem_id = g_mem_ids.allocate_id();
        parent_id = (image_desc && image_desc->mem_object) ? g_mem_ids.get_or_create(image_desc->mem_object) : 0;
    }
    std::string props_str = properties_to_string(properties);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateImageWithProperties", "call_id", cid, "context_id", ctx_id, "mem_id",
        mem_id, "flags", static_cast<uint64_t>(flags), "channel_order",
        image_format ? image_format->image_channel_order : 0u, "channel_data_type",
        image_format ? image_format->image_channel_data_type : 0u, "image_type",
        image_desc ? image_desc->image_type : 0u, "width", image_desc ? image_desc->image_width : 0u, "height",
        image_desc ? image_desc->image_height : 0u, "depth", image_desc ? image_desc->image_depth : 0u, "array_size",
        image_desc ? image_desc->image_array_size : 0u, "row_pitch", image_desc ? image_desc->image_row_pitch : 0u,
        "slice_pitch", image_desc ? image_desc->image_slice_pitch : 0u, "num_mip_levels",
        image_desc ? image_desc->num_mip_levels : 0u, "num_samples", image_desc ? image_desc->num_samples : 0u,
        "parent_mem_id", parent_id, "properties", perfetto::DynamicString(props_str));
    cl_mem img = tdispatch->clCreateImageWithProperties(
        context, properties, flags, image_format, image_desc, host_ptr, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_mem_ids.bind(img, mem_id);
    }
    return img;
}

static cl_mem clkp_clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
    size_t image_width, size_t image_height, size_t image_row_pitch, void *host_ptr, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t mem_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        mem_id = g_mem_ids.allocate_id();
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateImage2D", "call_id", cid, "context_id", ctx_id, "mem_id", mem_id,
        "flags", static_cast<uint64_t>(flags), "channel_order", image_format ? image_format->image_channel_order : 0u,
        "channel_data_type", image_format ? image_format->image_channel_data_type : 0u, "image_type",
        static_cast<uint64_t>(CL_MEM_OBJECT_IMAGE2D), "width", image_width, "height", image_height, "depth", 1u,
        "row_pitch", image_row_pitch);
    cl_mem img = tdispatch->clCreateImage2D(
        context, flags, image_format, image_width, image_height, image_row_pitch, host_ptr, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_mem_ids.bind(img, mem_id);
    }
    return img;
}

static cl_mem clkp_clCreateImage3D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
    size_t image_width, size_t image_height, size_t image_depth, size_t image_row_pitch, size_t image_slice_pitch,
    void *host_ptr, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t mem_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        mem_id = g_mem_ids.allocate_id();
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateImage3D", "call_id", cid, "context_id", ctx_id, "mem_id", mem_id,
        "flags", static_cast<uint64_t>(flags), "channel_order", image_format ? image_format->image_channel_order : 0u,
        "channel_data_type", image_format ? image_format->image_channel_data_type : 0u, "image_type",
        static_cast<uint64_t>(CL_MEM_OBJECT_IMAGE3D), "width", image_width, "height", image_height, "depth",
        image_depth, "row_pitch", image_row_pitch, "slice_pitch", image_slice_pitch);
    cl_mem img = tdispatch->clCreateImage3D(context, flags, image_format, image_width, image_height, image_depth,
        image_row_pitch, image_slice_pitch, host_ptr, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_mem_ids.bind(img, mem_id);
    }
    return img;
}

static cl_int clkp_clRetainMemObject(cl_mem memobj)
{
    uint64_t cid = 0;
    uint64_t mem_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        mem_id = g_mem_ids.get_or_create(memobj);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clRetainMemObject", "call_id", cid, "mem_id", mem_id);
    return tdispatch->clRetainMemObject(memobj);
}

static cl_int clkp_clReleaseMemObject(cl_mem memobj)
{
    uint64_t cid = 0;
    uint64_t mem_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        mem_id = g_mem_ids.find(memobj);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clReleaseMemObject", "call_id", cid, "mem_id", mem_id);
    return tdispatch->clReleaseMemObject(memobj);
}

static cl_sampler clkp_clCreateSampler(cl_context context, cl_bool normalized_coords,
    cl_addressing_mode addressing_mode, cl_filter_mode filter_mode, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t samp_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        samp_id = g_sampler_ids.allocate_id();
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateSampler", "call_id", cid, "context_id", ctx_id, "sampler_id", samp_id,
        "normalized_coords", normalized_coords, "addressing_mode", addressing_mode, "filter_mode", filter_mode);
    cl_sampler samp = tdispatch->clCreateSampler(context, normalized_coords, addressing_mode, filter_mode, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_sampler_ids.bind(samp, samp_id);
    }
    return samp;
}

static cl_sampler clkp_clCreateSamplerWithProperties(
    cl_context context, const cl_sampler_properties *sampler_properties, cl_int *errcode_ret)
{
    uint64_t cid = 0;
    uint64_t ctx_id = 0;
    uint64_t samp_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        ctx_id = g_context_ids.get_or_create(context);
        samp_id = g_sampler_ids.allocate_id();
    }
    cl_bool normalized_coords = CL_TRUE;
    cl_addressing_mode addressing_mode = CL_ADDRESS_CLAMP;
    cl_filter_mode filter_mode = CL_FILTER_NEAREST;
    if (sampler_properties != nullptr) {
        for (size_t i = 0; sampler_properties[i] != 0; i += 2) {
            if (sampler_properties[i] == CL_SAMPLER_NORMALIZED_COORDS) {
                normalized_coords = static_cast<cl_bool>(sampler_properties[i + 1]);
            } else if (sampler_properties[i] == CL_SAMPLER_ADDRESSING_MODE) {
                addressing_mode = static_cast<cl_addressing_mode>(sampler_properties[i + 1]);
            } else if (sampler_properties[i] == CL_SAMPLER_FILTER_MODE) {
                filter_mode = static_cast<cl_filter_mode>(sampler_properties[i + 1]);
            }
        }
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateSamplerWithProperties", "call_id", cid, "context_id", ctx_id,
        "sampler_id", samp_id, "normalized_coords", normalized_coords, "addressing_mode", addressing_mode,
        "filter_mode", filter_mode);
    cl_sampler samp = tdispatch->clCreateSamplerWithProperties(context, sampler_properties, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_sampler_ids.bind(samp, samp_id);
    }
    return samp;
}

static cl_int clkp_clRetainSampler(cl_sampler sampler)
{
    uint64_t cid = 0;
    uint64_t samp_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        samp_id = g_sampler_ids.get_or_create(sampler);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clRetainSampler", "call_id", cid, "sampler_id", samp_id);
    return tdispatch->clRetainSampler(sampler);
}

static cl_int clkp_clReleaseSampler(cl_sampler sampler)
{
    uint64_t cid = 0;
    uint64_t samp_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        samp_id = g_sampler_ids.find(sampler);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clReleaseSampler", "call_id", cid, "sampler_id", samp_id);
    return tdispatch->clReleaseSampler(sampler);
}

/*****************************************************************************/
/* ENQUEUE NDRANGE KERNEL & PROFILING CALLBACKS ******************************/
/*****************************************************************************/

struct ThreadInfo;

struct callback_data {
    cl_command_queue queue;
    ThreadInfo *thread_info = nullptr;
    cl_event event;
    size_t gidX, gidY, gidZ;
    int64_t time_offset;
    std::string kernel_name;
    std::string program_string;
    uint64_t call_id = 0;
    uint64_t dispatch_id = 0;
    uint64_t queue_id = 0;
    uint64_t kernel_id = 0;
    uint64_t cmdbuf_id = 0;
    bool is_cmdbuf = false;
};

struct ThreadInfo {
    std::mutex lock;
    std::condition_variable cv;
    std::queue<callback_data *> callbacks;
    bool stop;
    int pending_callbacks;
};

static std::map<cl_command_queue, ThreadInfo *> queue_to_thread_info;
static std::map<cl_command_queue, std::thread> queue_to_thread;
static std::map<cl_command_queue, int64_t> queue_to_time_offset;
static std::map<cl_command_queue, uint32_t> queue_to_refcount;

static void callback(cl_event event, cl_int event_command_exec_status, void *user_data)
{
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clkp-callback");

    struct callback_data *data = (struct callback_data *)user_data;
    assert(data != nullptr);
    assert(event_command_exec_status == CL_COMPLETE);

    ThreadInfo *thread_info = data->thread_info;
    if (thread_info == nullptr) {
        tdispatch->clReleaseEvent(data->event);
        delete data;
        return;
    }

    {
        std::lock_guard<std::mutex> lock_ti(thread_info->lock);
        thread_info->callbacks.push(data);
        thread_info->cv.notify_all();
    }
}

static callback_data *get_callback(ThreadInfo *thread_info)
{
    std::unique_lock<std::mutex> lock(thread_info->lock);
    while (thread_info->callbacks.empty()) {
        if (thread_info->stop && thread_info->pending_callbacks == 0) {
            return nullptr;
        }
        TRACE_EVENT_BEGIN(CLKP_PERFETTO_CATEGORY, "clkp_wait");
        thread_info->cv.wait(lock);
        TRACE_EVENT_END(CLKP_PERFETTO_CATEGORY);
    }
    callback_data *data = thread_info->callbacks.front();
    thread_info->callbacks.pop();
    return data;
}

static void trace_callback(callback_data *data)
{
    cl_command_queue queue = data->queue;
    cl_event event = data->event;
    size_t gidX = data->gidX, gidY = data->gidY, gidZ = data->gidZ;
    int64_t time_offset = data->time_offset;

    cl_ulong start, end;
    cl_int err;
    err = tdispatch->clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    CHECK_CL(err, tdispatch->clReleaseEvent(event);
             return, "clGetEventProfilingInfo(CL_PROFILING_COMMAND_START) failed (%i)", err);
    err = tdispatch->clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    CHECK_CL(err, tdispatch->clReleaseEvent(event);
             return, "clGetEventProfilingInfo(CL_PROFILING_COMMAND_END) failed (%i)", err);
    if (end < start) {
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, perfetto::StaticString("INVALID_TIMESTAMPS"),
            perfetto::Track((uintptr_t)queue), "start", start, "end", end);
        tdispatch->clReleaseEvent(event);
        return;
    }

    if (data->is_cmdbuf) {
        std::string name = "clkp_cmdbuf_" + std::to_string(data->cmdbuf_id);
        TRACE_EVENT_BEGIN(CLKP_PERFETTO_CATEGORY, perfetto::DynamicString(name), perfetto::Track((uintptr_t)queue),
            (uint64_t)(start + time_offset), "cmdbuf_id", data->cmdbuf_id, "dispatch_id", data->dispatch_id, "call_id",
            data->call_id, "queue_id", data->queue_id);
        TRACE_EVENT_END(CLKP_PERFETTO_CATEGORY, perfetto::Track((uintptr_t)queue), (uint64_t)(end + time_offset));
    } else {
        std::string name = data->program_string + "-" + data->kernel_name + "-" + std::to_string(gidX) + "."
            + std::to_string(gidY) + "." + std::to_string(gidZ);

        TRACE_EVENT_BEGIN(CLKP_PERFETTO_CATEGORY, perfetto::DynamicString(name), perfetto::Track((uintptr_t)queue),
            (uint64_t)(start + time_offset), "program", perfetto::DynamicString(data->program_string), "kernel_name",
            perfetto::DynamicString(data->kernel_name), "gidX", gidX, "gidY", gidY, "gidZ", gidZ, "dispatch_id",
            data->dispatch_id, "call_id", data->call_id, "queue_id", data->queue_id, "kernel_id", data->kernel_id);
        TRACE_EVENT_END(CLKP_PERFETTO_CATEGORY, perfetto::Track((uintptr_t)queue), (uint64_t)(end + time_offset));
    }

    tdispatch->clReleaseEvent(event);
}

static void queue_thread_function(ThreadInfo *thread_info)
{
#ifdef __APPLE__
    pthread_setname_np("clkp");
#else
    pthread_setname_np(pthread_self(), "clkp");
#endif
    while (true) {
        callback_data *data = get_callback(thread_info);
        if (data == nullptr) {
            return;
        }
        trace_callback(data);
        {
            std::lock_guard<std::mutex> lock(thread_info->lock);
            thread_info->pending_callbacks--;
            thread_info->cv.notify_all();
        }
        delete data;
    }
}

static cl_int attach_profiling_callback(cl_command_queue command_queue, cl_event *event_to_use, bool event_is_null,
    cl_event local_event, callback_data *data)
{
    ThreadInfo *thread_info = nullptr;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        auto it = queue_to_thread_info.find(command_queue);
        if (it != queue_to_thread_info.end()) {
            thread_info = it->second;
            std::lock_guard<std::mutex> lock_ti(thread_info->lock);
            if (thread_info->stop) {
                thread_info = nullptr;
            } else {
                thread_info->pending_callbacks++;
                data->thread_info = thread_info;
            }
        }
    }

    if (thread_info == nullptr) {
        delete data;
        if (event_is_null) {
            tdispatch->clReleaseEvent(local_event);
        }
        return CL_SUCCESS;
    }

    // Retain the event for the callback
    tdispatch->clRetainEvent(*event_to_use);

    cl_int err_cb = tdispatch->clSetEventCallback(*event_to_use, CL_COMPLETE, callback, data);
    if (err_cb != CL_SUCCESS) {
        PRINT("clSetEventCallback failed (%i)", err_cb);
        tdispatch->clReleaseEvent(*event_to_use); // Undo retain
        delete data;
        {
            std::lock_guard<std::mutex> lock_ti(thread_info->lock);
            thread_info->pending_callbacks--;
            thread_info->cv.notify_all();
        }
        if (event_is_null) {
            tdispatch->clReleaseEvent(local_event);
        }
        return err_cb;
    }

    if (event_is_null) {
        tdispatch->clReleaseEvent(local_event);
    }

    return CL_SUCCESS;
}

static cl_int clkp_clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    size_t gidX = (work_dim > 0 && global_work_size) ? global_work_size[0] : 1;
    size_t gidY = (work_dim > 1 && global_work_size) ? global_work_size[1] : 1;
    size_t gidZ = (work_dim > 2 && global_work_size) ? global_work_size[2] : 1;

    std::string gwo_str = size_array_to_string(global_work_offset, work_dim);
    std::string gws_str = size_array_to_string(global_work_size, work_dim);
    std::string lws_str = size_array_to_string(local_work_size, work_dim);

    bool event_is_null = event == nullptr;
    cl_event local_event = nullptr;
    cl_event *event_to_use = event_is_null ? &local_event : event;

    uint64_t cid = 0;
    uint64_t did = 0;
    uint64_t q_id = 0;
    uint64_t k_id = 0;
    std::string kernel_name = "?";
    uint64_t prog_id = 0;
    std::string program_string = "clkp_p?";
    std::string wait_str;
    uint64_t ret_event_id = 0;
    int64_t time_offset = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cid = next_call_id();
        did = next_dispatch_id();
        q_id = g_queue_ids.get_or_create(command_queue);
        k_id = g_kernel_ids.get_or_create(kernel);
        if (kernel_to_kernel_name.count(kernel)) {
            kernel_name = kernel_to_kernel_name[kernel];
        }
        cl_program prog = kernel_to_program.count(kernel) ? kernel_to_program[kernel] : nullptr;
        prog_id = g_program_ids.find(prog);
        if (prog && program_to_string.count(prog)) {
            program_string = program_to_string[prog];
        }
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
        ret_event_id = !event_is_null ? g_event_ids.allocate_id() : 0;
        if (queue_to_time_offset.count(command_queue)) {
            time_offset = queue_to_time_offset[command_queue];
        }
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueNDRangeKernel", "call_id", cid, "dispatch_id", did, "queue_id", q_id,
        "kernel_id", k_id, "program_id", prog_id, "program", perfetto::DynamicString(program_string), "kernel_name",
        perfetto::DynamicString(kernel_name), "work_dim", work_dim, "gidX", gidX, "gidY", gidY, "gidZ", gidZ, "gwo",
        perfetto::DynamicString(gwo_str), "gws", perfetto::DynamicString(gws_str), "lws",
        perfetto::DynamicString(lws_str), "wait_events", perfetto::DynamicString(wait_str), "ret_event_id",
        ret_event_id);

    cl_int err = tdispatch->clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset,
        global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event_to_use);

    if (err != CL_SUCCESS) {
        return err;
    }
    if (!event_is_null && *event != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_event_ids.bind(*event, ret_event_id);
    }

    struct callback_data *data = new (std::nothrow) callback_data();
    if (data == nullptr) {
        if (event_is_null) {
            tdispatch->clReleaseEvent(local_event);
        }
        return CL_OUT_OF_HOST_MEMORY;
    }

    data->queue = command_queue;
    data->event = *event_to_use;
    data->gidX = gidX;
    data->gidY = gidY;
    data->gidZ = gidZ;
    data->time_offset = time_offset;
    data->kernel_name = kernel_name;
    data->program_string = program_string;
    data->call_id = cid;
    data->dispatch_id = did;
    data->queue_id = q_id;
    data->kernel_id = k_id;
    data->is_cmdbuf = false;

    return attach_profiling_callback(command_queue, event_to_use, event_is_null, local_event, data);
}

static cl_int clkp_clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
    size_t gws[1] = { 1 };
    size_t lws[1] = { 1 };
    return clkp_clEnqueueNDRangeKernel(
        command_queue, kernel, 1, nullptr, gws, lws, num_events_in_wait_list, event_wait_list, event);
}

/*****************************************************************************/
/* DATA TRANSFER, MAP/UNMAP & SYNCHRONIZATION ENQUEUES ***********************/
/*****************************************************************************/

static cl_int clkp_clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
    size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(buffer);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    cl_int err = tdispatch->clEnqueueReadBuffer(
        command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueReadBuffer", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "blocking", blocking_read, "offset", offset, "size", size, "wait_events", perfetto::DynamicString(wait_str),
        "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
    size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(buffer);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    cl_int err = tdispatch->clEnqueueWriteBuffer(
        command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueWriteBuffer", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "blocking", blocking_write, "offset", offset, "size", size, "wait_events", perfetto::DynamicString(wait_str),
        "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueReadBufferRect(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
    const size_t *buffer_origin, const size_t *host_origin, const size_t *region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, void *ptr,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(buffer);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string bo_str = size_array_to_string(buffer_origin, 3);
    std::string ho_str = size_array_to_string(host_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    cl_int err = tdispatch->clEnqueueReadBufferRect(command_queue, buffer, blocking_read, buffer_origin, host_origin,
        region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list,
        event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueReadBufferRect", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "blocking", blocking_read, "buffer_origin", perfetto::DynamicString(bo_str), "host_origin",
        perfetto::DynamicString(ho_str), "region", perfetto::DynamicString(reg_str), "buffer_row_pitch",
        buffer_row_pitch, "buffer_slice_pitch", buffer_slice_pitch, "host_row_pitch", host_row_pitch,
        "host_slice_pitch", host_slice_pitch, "wait_events", perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueWriteBufferRect(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
    const size_t *buffer_origin, const size_t *host_origin, const size_t *region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, const void *ptr,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(buffer);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string bo_str = size_array_to_string(buffer_origin, 3);
    std::string ho_str = size_array_to_string(host_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    cl_int err = tdispatch->clEnqueueWriteBufferRect(command_queue, buffer, blocking_write, buffer_origin, host_origin,
        region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list,
        event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueWriteBufferRect", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "blocking", blocking_write, "buffer_origin", perfetto::DynamicString(bo_str), "host_origin",
        perfetto::DynamicString(ho_str), "region", perfetto::DynamicString(reg_str), "buffer_row_pitch",
        buffer_row_pitch, "buffer_slice_pitch", buffer_slice_pitch, "host_row_pitch", host_row_pitch,
        "host_slice_pitch", host_slice_pitch, "wait_events", perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
    size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        src_id = g_mem_ids.get_or_create(src_buffer);
        dst_id = g_mem_ids.get_or_create(dst_buffer);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    cl_int err = tdispatch->clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size,
        num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueCopyBuffer", "call_id", cid, "queue_id", q_id, "src_mem_id", src_id,
        "dst_mem_id", dst_id, "src_offset", src_offset, "dst_offset", dst_offset, "size", size, "wait_events",
        perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueCopyBufferRect(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region, size_t src_row_pitch,
    size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        src_id = g_mem_ids.get_or_create(src_buffer);
        dst_id = g_mem_ids.get_or_create(dst_buffer);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string so_str = size_array_to_string(src_origin, 3);
    std::string do_str = size_array_to_string(dst_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    cl_int err = tdispatch->clEnqueueCopyBufferRect(command_queue, src_buffer, dst_buffer, src_origin, dst_origin,
        region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, num_events_in_wait_list,
        event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueCopyBufferRect", "call_id", cid, "queue_id", q_id, "src_mem_id",
        src_id, "dst_mem_id", dst_id, "src_origin", perfetto::DynamicString(so_str), "dst_origin",
        perfetto::DynamicString(do_str), "region", perfetto::DynamicString(reg_str), "src_row_pitch", src_row_pitch,
        "src_slice_pitch", src_slice_pitch, "dst_row_pitch", dst_row_pitch, "dst_slice_pitch", dst_slice_pitch,
        "wait_events", perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void *pattern,
    size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(buffer);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string pat_hex = clkp::bytes_to_hex(pattern, pattern_size);
    cl_int err = tdispatch->clEnqueueFillBuffer(
        command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueFillBuffer", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "pattern_hex", perfetto::DynamicString(pat_hex), "pattern_size", pattern_size, "offset", offset, "size", size,
        "wait_events", perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueReadImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read,
    const size_t *origin, const size_t *region, size_t row_pitch, size_t slice_pitch, void *ptr,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(image);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string orig_str = size_array_to_string(origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    cl_int err = tdispatch->clEnqueueReadImage(command_queue, image, blocking_read, origin, region, row_pitch,
        slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueReadImage", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "blocking", blocking_read, "origin", perfetto::DynamicString(orig_str), "region",
        perfetto::DynamicString(reg_str), "row_pitch", row_pitch, "slice_pitch", slice_pitch, "wait_events",
        perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write,
    const size_t *origin, const size_t *region, size_t input_row_pitch, size_t input_slice_pitch, const void *ptr,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(image);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string orig_str = size_array_to_string(origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    cl_int err = tdispatch->clEnqueueWriteImage(command_queue, image, blocking_write, origin, region, input_row_pitch,
        input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueWriteImage", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "blocking", blocking_write, "origin", perfetto::DynamicString(orig_str), "region",
        perfetto::DynamicString(reg_str), "row_pitch", input_row_pitch, "slice_pitch", input_slice_pitch, "wait_events",
        perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueCopyImage(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        src_id = g_mem_ids.get_or_create(src_image);
        dst_id = g_mem_ids.get_or_create(dst_image);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string so_str = size_array_to_string(src_origin, 3);
    std::string do_str = size_array_to_string(dst_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    cl_int err = tdispatch->clEnqueueCopyImage(command_queue, src_image, dst_image, src_origin, dst_origin, region,
        num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueCopyImage", "call_id", cid, "queue_id", q_id, "src_mem_id", src_id,
        "dst_mem_id", dst_id, "src_origin", perfetto::DynamicString(so_str), "dst_origin",
        perfetto::DynamicString(do_str), "region", perfetto::DynamicString(reg_str), "wait_events",
        perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueFillImage(cl_command_queue command_queue, cl_mem image, const void *fill_color,
    const size_t *origin, const size_t *region, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(image);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string color_hex = clkp::bytes_to_hex(fill_color, 16);
    std::string orig_str = size_array_to_string(origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    cl_int err = tdispatch->clEnqueueFillImage(
        command_queue, image, fill_color, origin, region, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueFillImage", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "fill_color_hex", perfetto::DynamicString(color_hex), "origin", perfetto::DynamicString(orig_str), "region",
        perfetto::DynamicString(reg_str), "wait_events", perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer,
    const size_t *src_origin, const size_t *region, size_t dst_offset, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        src_id = g_mem_ids.get_or_create(src_image);
        dst_id = g_mem_ids.get_or_create(dst_buffer);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string so_str = size_array_to_string(src_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    cl_int err = tdispatch->clEnqueueCopyImageToBuffer(command_queue, src_image, dst_buffer, src_origin, region,
        dst_offset, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueCopyImageToBuffer", "call_id", cid, "queue_id", q_id, "src_mem_id",
        src_id, "dst_mem_id", dst_id, "src_origin", perfetto::DynamicString(so_str), "region",
        perfetto::DynamicString(reg_str), "dst_offset", dst_offset, "wait_events", perfetto::DynamicString(wait_str),
        "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image,
    size_t src_offset, const size_t *dst_origin, const size_t *region, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        src_id = g_mem_ids.get_or_create(src_buffer);
        dst_id = g_mem_ids.get_or_create(dst_image);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string do_str = size_array_to_string(dst_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    cl_int err = tdispatch->clEnqueueCopyBufferToImage(command_queue, src_buffer, dst_image, src_offset, dst_origin,
        region, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueCopyBufferToImage", "call_id", cid, "queue_id", q_id, "src_mem_id",
        src_id, "dst_mem_id", dst_id, "src_offset", src_offset, "dst_origin", perfetto::DynamicString(do_str), "region",
        perfetto::DynamicString(reg_str), "wait_events", perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static void *clkp_clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map,
    cl_map_flags map_flags, size_t offset, size_t size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event, cl_int *errcode_ret)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(buffer);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    void *ptr = tdispatch->clEnqueueMapBuffer(command_queue, buffer, blocking_map, map_flags, offset, size,
        num_events_in_wait_list, event_wait_list, event, errcode_ret);
    uint64_t map_id;
    uint64_t ret_ev = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        map_id = g_map_ids.get_or_create(ptr);
        if (event && *event) {
            ret_ev = g_event_ids.create_fresh(*event);
        }
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueMapBuffer", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "map_id", map_id, "blocking", blocking_map, "map_flags", static_cast<uint64_t>(map_flags), "offset", offset,
        "size", size, "wait_events", perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return ptr;
}

static void *clkp_clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map,
    cl_map_flags map_flags, const size_t *origin, const size_t *region, size_t *image_row_pitch,
    size_t *image_slice_pitch, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event,
    cl_int *errcode_ret)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(image);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string orig_str = size_array_to_string(origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    void *ptr = tdispatch->clEnqueueMapImage(command_queue, image, blocking_map, map_flags, origin, region,
        image_row_pitch, image_slice_pitch, num_events_in_wait_list, event_wait_list, event, errcode_ret);
    uint64_t map_id;
    uint64_t ret_ev = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        map_id = g_map_ids.get_or_create(ptr);
        if (event && *event) {
            ret_ev = g_event_ids.create_fresh(*event);
        }
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueMapImage", "call_id", cid, "queue_id", q_id, "mem_id", m_id, "map_id",
        map_id, "blocking", blocking_map, "map_flags", static_cast<uint64_t>(map_flags), "origin",
        perfetto::DynamicString(orig_str), "region", perfetto::DynamicString(reg_str), "wait_events",
        perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return ptr;
}

static cl_int clkp_clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    uint64_t m_id;
    uint64_t map_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        m_id = g_mem_ids.get_or_create(memobj);
        map_id = g_map_ids.find(mapped_ptr);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    cl_int err = tdispatch->clEnqueueUnmapMemObject(
        command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueUnmapMemObject", "call_id", cid, "queue_id", q_id, "mem_id", m_id,
        "map_id", map_id, "wait_events", perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueMigrateMemObjects(cl_command_queue command_queue, cl_uint num_mem_objects,
    const cl_mem *mem_objects, cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    std::vector<uint64_t> m_ids;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        for (cl_uint i = 0; i < num_mem_objects && mem_objects != nullptr; ++i) {
            m_ids.push_back(g_mem_ids.get_or_create(mem_objects[i]));
        }
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    std::string mems_str = clkp::uint64_list_to_string(m_ids);
    cl_int err = tdispatch->clEnqueueMigrateMemObjects(
        command_queue, num_mem_objects, mem_objects, flags, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueMigrateMemObjects", "call_id", cid, "queue_id", q_id, "mem_ids",
        perfetto::DynamicString(mems_str), "flags", static_cast<uint64_t>(flags), "wait_events",
        perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueMarkerWithWaitList(
    cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    cl_int err = tdispatch->clEnqueueMarkerWithWaitList(command_queue, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueMarkerWithWaitList", "call_id", cid, "queue_id", q_id, "wait_events",
        perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueBarrierWithWaitList(
    cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
    }
    cl_int err
        = tdispatch->clEnqueueBarrierWithWaitList(command_queue, num_events_in_wait_list, event_wait_list, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueBarrierWithWaitList", "call_id", cid, "queue_id", q_id, "wait_events",
        perfetto::DynamicString(wait_str), "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueMarker(cl_command_queue command_queue, cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
    }
    cl_int err = tdispatch->clEnqueueMarker(command_queue, event);
    uint64_t ret_ev = 0;
    if (event && err == CL_SUCCESS && *event) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ret_ev = g_event_ids.create_fresh(*event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueMarker", "call_id", cid, "queue_id", q_id, "ret_event_id", ret_ev);
    return err;
}

static cl_int clkp_clEnqueueBarrier(cl_command_queue command_queue)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
    }
    cl_int err = tdispatch->clEnqueueBarrier(command_queue);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueBarrier", "call_id", cid, "queue_id", q_id);
    return err;
}

static cl_int clkp_clEnqueueWaitForEvents(
    cl_command_queue command_queue, cl_uint num_events, const cl_event *event_list)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
        wait_str = event_list_to_string(num_events, event_list);
    }
    cl_int err = tdispatch->clEnqueueWaitForEvents(command_queue, num_events, event_list);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueWaitForEvents", "call_id", cid, "queue_id", q_id, "wait_events",
        perfetto::DynamicString(wait_str));
    return err;
}

static cl_int clkp_clFlush(cl_command_queue command_queue)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clFlush", "call_id", cid, "queue_id", q_id);
    return tdispatch->clFlush(command_queue);
}

static cl_int clkp_clFinish(cl_command_queue command_queue)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clFinish", "call_id", cid, "queue_id", q_id);
    return tdispatch->clFinish(command_queue);
}

static cl_int clkp_clWaitForEvents(cl_uint num_events, const cl_event *event_list)
{
    uint64_t cid = next_call_id();
    std::string wait_str;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        wait_str = event_list_to_string(num_events, event_list);
    }
    TRACE_EVENT(
        CLKP_PERFETTO_CATEGORY, "clWaitForEvents", "call_id", cid, "wait_events", perfetto::DynamicString(wait_str));
    return tdispatch->clWaitForEvents(num_events, event_list);
}

static cl_event clkp_clCreateUserEvent(cl_context context, cl_int *errcode_ret)
{
    uint64_t cid = next_call_id();
    uint64_t ctx_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ctx_id = g_context_ids.get_or_create(context);
    }
    cl_event ev = tdispatch->clCreateUserEvent(context, errcode_ret);
    uint64_t ev_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ev_id = g_event_ids.create_fresh(ev);
    }
    TRACE_EVENT(
        CLKP_PERFETTO_CATEGORY, "clCreateUserEvent", "call_id", cid, "context_id", ctx_id, "ret_event_id", ev_id);
    return ev;
}

static cl_int clkp_clSetUserEventStatus(cl_event event, cl_int execution_status)
{
    uint64_t cid = next_call_id();
    uint64_t ev_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ev_id = g_event_ids.get_or_create(event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clSetUserEventStatus", "call_id", cid, "event_id", ev_id, "execution_status",
        execution_status);
    return tdispatch->clSetUserEventStatus(event, execution_status);
}

static cl_int clkp_clRetainEvent(cl_event event)
{
    uint64_t cid = next_call_id();
    uint64_t ev_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ev_id = g_event_ids.get_or_create(event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clRetainEvent", "call_id", cid, "event_id", ev_id);
    return tdispatch->clRetainEvent(event);
}

static cl_int clkp_clReleaseEvent(cl_event event)
{
    uint64_t cid = next_call_id();
    uint64_t ev_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ev_id = g_event_ids.find(event);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clReleaseEvent", "call_id", cid, "event_id", ev_id);
    return tdispatch->clReleaseEvent(event);
}

/*****************************************************************************/
/* CREATE COMMAND QUEUE ******************************************************/
/*****************************************************************************/

static cl_int clkp_clRetainCommandQueue(cl_command_queue command_queue)
{
    uint64_t cid = next_call_id();
    uint64_t q_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.get_or_create(command_queue);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clRetainCommandQueue", "call_id", cid, "queue_id", q_id);
    cl_int ret = tdispatch->clRetainCommandQueue(command_queue);
    if (ret == CL_SUCCESS && command_queue != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        if (queue_to_refcount.count(command_queue)) {
            queue_to_refcount[command_queue]++;
        }
    }
    return ret;
}

static cl_int clkp_clReleaseCommandQueue(cl_command_queue command_queue)
{
    uint64_t cid = next_call_id();
    uint64_t q_id = 0;
    ThreadInfo *thread_info = nullptr;
    std::thread thread_to_join;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        q_id = g_queue_ids.find(command_queue);
        if (command_queue != nullptr) {
            bool destroy = true;
            auto ref_it = queue_to_refcount.find(command_queue);
            if (ref_it != queue_to_refcount.end()) {
                if (ref_it->second > 1) {
                    ref_it->second--;
                    destroy = false;
                } else {
                    queue_to_refcount.erase(ref_it);
                }
            }
            if (destroy) {
                if (queue_to_thread_info.count(command_queue)) {
                    thread_info = queue_to_thread_info[command_queue];
                    queue_to_thread_info.erase(command_queue);
                }
                if (queue_to_thread.count(command_queue)) {
                    thread_to_join = std::move(queue_to_thread[command_queue]);
                    queue_to_thread.erase(command_queue);
                }
                queue_to_time_offset.erase(command_queue);
            }
        }
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clReleaseCommandQueue", "call_id", cid, "queue_id", q_id);

    auto ret = tdispatch->clReleaseCommandQueue(command_queue);

    if (thread_info) {
        {
            std::unique_lock<std::mutex> lock_ti(thread_info->lock);
            thread_info->stop = true;
            thread_info->cv.notify_all();
            thread_info->cv.wait(lock_ti, [&] { return thread_info->pending_callbacks == 0; });
        }
        if (thread_to_join.joinable()) {
            thread_to_join.join();
        }
        delete thread_info;
    }

    return ret;
}

static cl_command_queue create_command_queue(
    cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret, uint64_t q_id)
{
    std::vector<cl_queue_properties> properties_array;
    bool cl_queue_properties_found = false;
    if (properties) {
        for (unsigned i = 0; properties[i] != 0; i += 2) {
            cl_queue_properties key = properties[i];
            cl_queue_properties val = properties[i + 1];
            if (key == CL_QUEUE_PROPERTIES) {
                TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueue-properties", "properties", val);

                val |= CL_QUEUE_PROFILING_ENABLE;
                cl_queue_properties_found = true;
            }
            properties_array.push_back(key);
            properties_array.push_back(val);
        }
    }
    if (!cl_queue_properties_found) {
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueue-properties-not-found");
        properties_array.push_back(CL_QUEUE_PROPERTIES);
        properties_array.push_back(CL_QUEUE_PROFILING_ENABLE);
    }
    properties_array.push_back(0);

    auto command_queue
        = tdispatch->clCreateCommandQueueWithProperties(context, device, properties_array.data(), errcode_ret);

    int64_t offset = 0;
    if (command_queue != nullptr) {
        if (tdispatch->clGetDeviceAndHostTimer) {
            cl_ulong device_timestamp, host_timestamp;
            cl_int timer_err = tdispatch->clGetDeviceAndHostTimer(device, &device_timestamp, &host_timestamp);
            if (timer_err == CL_SUCCESS) {
                uint64_t perfetto_timestamp = perfetto::TrackEvent::GetTraceTimeNs();
                offset = perfetto_timestamp - device_timestamp;
            } else {
                PRINT("clGetDeviceAndHostTimer failed (%i)", timer_err);
            }
        } else {
            PRINT("clGetDeviceAndHostTimer is not available");
        }
    }

    uint64_t ctx_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_queue_ids.bind(command_queue, q_id);
        ctx_id = g_context_ids.get_or_create(context);
        if (command_queue != nullptr) {
            queue_to_context_id[command_queue] = ctx_id;
            ThreadInfo *thread_info = new ThreadInfo();
            thread_info->stop = false;
            thread_info->pending_callbacks = 0;
            queue_to_thread_info[command_queue] = thread_info;
            queue_to_thread.emplace(command_queue, [thread_info] { queue_thread_function(thread_info); });
            queue_to_time_offset[command_queue] = offset;
            queue_to_refcount[command_queue] = 1;
        }
    }

    if (command_queue != nullptr) {
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY,
            perfetto::DynamicString("clkp-queue_" + std::to_string((uintptr_t)command_queue)),
            perfetto::Track((uintptr_t)command_queue), "queue_id", q_id, "context_id", ctx_id);
    }

    return command_queue;
}

static uint64_t extract_queue_properties_bitfield(const cl_queue_properties *properties)
{
    if (properties == nullptr) {
        return 0;
    }
    for (unsigned i = 0; properties[i] != 0; i += 2) {
        if (properties[i] == CL_QUEUE_PROPERTIES) {
            return static_cast<uint64_t>(properties[i + 1]);
        }
    }
    return 0;
}

static cl_command_queue clkp_clCreateCommandQueue(
    cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret)
{
    uint64_t cid = next_call_id();
    uint64_t ctx_id;
    uint64_t q_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ctx_id = g_context_ids.get_or_create(context);
        q_id = g_queue_ids.allocate_id();
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueue", "call_id", cid, "context_id", ctx_id, "queue_id", q_id,
        "properties", static_cast<uint64_t>(properties));
    cl_queue_properties props[4] = { CL_QUEUE_PROPERTIES, properties, 0, 0 };
    return create_command_queue(context, device, props, errcode_ret, q_id);
}

static cl_command_queue clkp_clCreateCommandQueueWithProperties(
    cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret)
{
    uint64_t cid = next_call_id();
    uint64_t ctx_id;
    uint64_t q_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        ctx_id = g_context_ids.get_or_create(context);
        q_id = g_queue_ids.allocate_id();
    }
    uint64_t orig_props = extract_queue_properties_bitfield(properties);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueueWithProperties", "call_id", cid, "context_id", ctx_id,
        "queue_id", q_id, "properties", orig_props);
    return create_command_queue(context, device, properties, errcode_ret, q_id);
}

/*****************************************************************************/
/* CL_KHR_COMMAND_BUFFER & EXTENSION FUNCTIONS *******************************/
/*****************************************************************************/

struct ExtensionDispatch {
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
    clGetCommandBufferInfoKHR_fn clGetCommandBufferInfoKHR = nullptr;
    clUpdateMutableCommandsKHR_fn clUpdateMutableCommandsKHR = nullptr;
    clGetMutableCommandInfoKHR_fn clGetMutableCommandInfoKHR = nullptr;
    clCreateProgramWithILKHR_fn clCreateProgramWithILKHR = nullptr;
    clCreateCommandQueueWithPropertiesKHR_fn clCreateCommandQueueWithPropertiesKHR = nullptr;
};

static ExtensionDispatch g_ext;

static cl_command_buffer_khr CL_API_CALL clkp_clCreateCommandBufferKHR(cl_uint num_queues,
    const cl_command_queue *queues, const cl_command_buffer_properties_khr *properties, cl_int *errcode_ret)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    std::vector<uint64_t> q_ids;
    std::vector<cl_command_queue> q_vec;
    uint64_t ctx_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.allocate_id();
        for (cl_uint i = 0; i < num_queues && queues != nullptr; ++i) {
            q_ids.push_back(g_queue_ids.get_or_create(queues[i]));
            q_vec.push_back(queues[i]);
            if (ctx_id == 0 && queue_to_context_id.count(queues[i])) {
                ctx_id = queue_to_context_id[queues[i]];
            }
        }
    }
    std::string q_ids_str = clkp::uint64_list_to_string(q_ids);
    std::string props_str = properties_to_string(properties);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateCommandBufferKHR", "call_id", cid, "cmdbuf_id", cb_id, "context_id",
        ctx_id, "queue_ids", perfetto::DynamicString(q_ids_str), "properties", perfetto::DynamicString(props_str));
    cl_command_buffer_khr cmdbuf = g_ext.clCreateCommandBufferKHR(num_queues, queues, properties, errcode_ret);
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_cmdbuf_ids.bind(cmdbuf, cb_id);
        if (cmdbuf != nullptr) {
            cmdbuf_to_queues[cmdbuf] = std::move(q_vec);
            cmdbuf_to_context_id[cmdbuf] = ctx_id;
        }
    }
    return cmdbuf;
}

static cl_int CL_API_CALL clkp_clFinalizeCommandBufferKHR(cl_command_buffer_khr command_buffer)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clFinalizeCommandBufferKHR", "call_id", cid, "cmdbuf_id", cb_id);
    return g_ext.clFinalizeCommandBufferKHR(command_buffer);
}

static cl_int CL_API_CALL clkp_clRetainCommandBufferKHR(cl_command_buffer_khr command_buffer)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clRetainCommandBufferKHR", "call_id", cid, "cmdbuf_id", cb_id);
    return g_ext.clRetainCommandBufferKHR(command_buffer);
}

static cl_int CL_API_CALL clkp_clReleaseCommandBufferKHR(cl_command_buffer_khr command_buffer)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.find(command_buffer);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clReleaseCommandBufferKHR", "call_id", cid, "cmdbuf_id", cb_id);
    return g_ext.clReleaseCommandBufferKHR(command_buffer);
}

static cl_int CL_API_CALL clkp_clEnqueueCommandBufferKHR(cl_uint num_queues, cl_command_queue *queues,
    cl_command_buffer_khr command_buffer, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event)
{
    uint64_t cid = next_call_id();
    uint64_t did = next_dispatch_id();
    uint64_t cb_id;
    std::vector<uint64_t> q_ids;
    cl_command_queue target_queue = nullptr;
    std::string wait_str;
    bool event_is_null = (event == nullptr);
    cl_event local_event = nullptr;
    cl_event *event_to_use = event_is_null ? &local_event : event;
    uint64_t ret_ev = 0;

    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        if (num_queues > 0 && queues != nullptr) {
            target_queue = queues[0];
            for (cl_uint i = 0; i < num_queues; ++i) {
                q_ids.push_back(g_queue_ids.get_or_create(queues[i]));
            }
        } else if (cmdbuf_to_queues.count(command_buffer) && !cmdbuf_to_queues[command_buffer].empty()) {
            target_queue = cmdbuf_to_queues[command_buffer][0];
        }
        wait_str = event_list_to_string(num_events_in_wait_list, event_wait_list);
        if (!event_is_null) {
            ret_ev = g_event_ids.allocate_id();
        }
    }
    std::string q_ids_str = clkp::uint64_list_to_string(q_ids);

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueCommandBufferKHR", "call_id", cid, "dispatch_id", did, "cmdbuf_id",
        cb_id, "queue_ids", perfetto::DynamicString(q_ids_str), "wait_events", perfetto::DynamicString(wait_str),
        "ret_event_id", ret_ev);

    cl_int err = g_ext.clEnqueueCommandBufferKHR(
        num_queues, queues, command_buffer, num_events_in_wait_list, event_wait_list, event_to_use);

    int64_t time_offset = 0;
    uint64_t target_q_id = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        if (err == CL_SUCCESS && !event_is_null && *event != nullptr) {
            g_event_ids.bind(*event, ret_ev);
        }
        if (target_queue != nullptr) {
            time_offset = queue_to_time_offset.count(target_queue) ? queue_to_time_offset[target_queue] : 0;
            target_q_id = g_queue_ids.find(target_queue);
        }
    }

    if (err != CL_SUCCESS || target_queue == nullptr || *event_to_use == nullptr) {
        if (err == CL_SUCCESS && event_is_null && local_event != nullptr) {
            tdispatch->clReleaseEvent(local_event);
        }
        return err;
    }

    struct callback_data *data = new (std::nothrow) callback_data();
    if (data == nullptr) {
        if (event_is_null) {
            tdispatch->clReleaseEvent(local_event);
        }
        return CL_OUT_OF_HOST_MEMORY;
    }

    data->queue = target_queue;
    data->event = *event_to_use;
    data->time_offset = time_offset;
    data->call_id = cid;
    data->dispatch_id = did;
    data->queue_id = target_q_id;
    data->cmdbuf_id = cb_id;
    data->is_cmdbuf = true;

    return attach_profiling_callback(target_queue, event_to_use, event_is_null, local_event, data);
}

static cl_int CL_API_CALL clkp_clCommandBarrierWithWaitListKHR(cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, const cl_command_properties_khr *properties, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    uint64_t q_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        q_id = g_queue_ids.find(command_queue);
    }
    std::string props_str = properties_to_string(properties);
    std::string wait_sp_str = sync_point_list_to_string(num_sync_points_in_wait_list, sync_point_wait_list);

    cl_int err = g_ext.clCommandBarrierWithWaitListKHR(command_buffer, command_queue, properties,
        num_sync_points_in_wait_list, sync_point_wait_list, sync_point, mutable_handle);

    uint64_t ret_sp = (err == CL_SUCCESS && sync_point != nullptr) ? static_cast<uint64_t>(*sync_point) : 0;
    uint64_t mut_id = 0;
    if (err == CL_SUCCESS && mutable_handle != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        mut_id = g_mutable_cmd_ids.create_fresh(*mutable_handle);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCommandBarrierWithWaitListKHR", "call_id", cid, "cmdbuf_id", cb_id,
        "queue_id", q_id, "properties", perfetto::DynamicString(props_str), "wait_sync_points",
        perfetto::DynamicString(wait_sp_str), "ret_sync_point", ret_sp, "mutable_cmd_id", mut_id);
    return err;
}

static cl_int CL_API_CALL clkp_clCommandCopyBufferKHR(cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, const cl_command_properties_khr *properties, cl_mem src_buffer, cl_mem dst_buffer,
    size_t src_offset, size_t dst_offset, size_t size, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        q_id = g_queue_ids.find(command_queue);
        src_id = g_mem_ids.get_or_create(src_buffer);
        dst_id = g_mem_ids.get_or_create(dst_buffer);
    }
    std::string props_str = properties_to_string(properties);
    std::string wait_sp_str = sync_point_list_to_string(num_sync_points_in_wait_list, sync_point_wait_list);

    cl_int err = g_ext.clCommandCopyBufferKHR(command_buffer, command_queue, properties, src_buffer, dst_buffer,
        src_offset, dst_offset, size, num_sync_points_in_wait_list, sync_point_wait_list, sync_point, mutable_handle);

    uint64_t ret_sp = (err == CL_SUCCESS && sync_point != nullptr) ? static_cast<uint64_t>(*sync_point) : 0;
    uint64_t mut_id = 0;
    if (err == CL_SUCCESS && mutable_handle != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        mut_id = g_mutable_cmd_ids.create_fresh(*mutable_handle);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCommandCopyBufferKHR", "call_id", cid, "cmdbuf_id", cb_id, "queue_id", q_id,
        "properties", perfetto::DynamicString(props_str), "src_mem_id", src_id, "dst_mem_id", dst_id, "src_offset",
        src_offset, "dst_offset", dst_offset, "size", size, "wait_sync_points", perfetto::DynamicString(wait_sp_str),
        "ret_sync_point", ret_sp, "mutable_cmd_id", mut_id);
    return err;
}

static cl_int CL_API_CALL clkp_clCommandCopyBufferRectKHR(cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, const cl_command_properties_khr *properties, cl_mem src_buffer, cl_mem dst_buffer,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region, size_t src_row_pitch,
    size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        q_id = g_queue_ids.find(command_queue);
        src_id = g_mem_ids.get_or_create(src_buffer);
        dst_id = g_mem_ids.get_or_create(dst_buffer);
    }
    std::string props_str = properties_to_string(properties);
    std::string so_str = size_array_to_string(src_origin, 3);
    std::string do_str = size_array_to_string(dst_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    std::string wait_sp_str = sync_point_list_to_string(num_sync_points_in_wait_list, sync_point_wait_list);

    cl_int err = g_ext.clCommandCopyBufferRectKHR(command_buffer, command_queue, properties, src_buffer, dst_buffer,
        src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch,
        num_sync_points_in_wait_list, sync_point_wait_list, sync_point, mutable_handle);

    uint64_t ret_sp = (err == CL_SUCCESS && sync_point != nullptr) ? static_cast<uint64_t>(*sync_point) : 0;
    uint64_t mut_id = 0;
    if (err == CL_SUCCESS && mutable_handle != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        mut_id = g_mutable_cmd_ids.create_fresh(*mutable_handle);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCommandCopyBufferRectKHR", "call_id", cid, "cmdbuf_id", cb_id, "queue_id",
        q_id, "properties", perfetto::DynamicString(props_str), "src_mem_id", src_id, "dst_mem_id", dst_id,
        "src_origin", perfetto::DynamicString(so_str), "dst_origin", perfetto::DynamicString(do_str), "region",
        perfetto::DynamicString(reg_str), "src_row_pitch", src_row_pitch, "src_slice_pitch", src_slice_pitch,
        "dst_row_pitch", dst_row_pitch, "dst_slice_pitch", dst_slice_pitch, "wait_sync_points",
        perfetto::DynamicString(wait_sp_str), "ret_sync_point", ret_sp, "mutable_cmd_id", mut_id);
    return err;
}

static cl_int CL_API_CALL clkp_clCommandCopyBufferToImageKHR(cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, const cl_command_properties_khr *properties, cl_mem src_buffer, cl_mem dst_image,
    size_t src_offset, const size_t *dst_origin, const size_t *region, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        q_id = g_queue_ids.find(command_queue);
        src_id = g_mem_ids.get_or_create(src_buffer);
        dst_id = g_mem_ids.get_or_create(dst_image);
    }
    std::string props_str = properties_to_string(properties);
    std::string do_str = size_array_to_string(dst_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    std::string wait_sp_str = sync_point_list_to_string(num_sync_points_in_wait_list, sync_point_wait_list);

    cl_int err = g_ext.clCommandCopyBufferToImageKHR(command_buffer, command_queue, properties, src_buffer, dst_image,
        src_offset, dst_origin, region, num_sync_points_in_wait_list, sync_point_wait_list, sync_point, mutable_handle);

    uint64_t ret_sp = (err == CL_SUCCESS && sync_point != nullptr) ? static_cast<uint64_t>(*sync_point) : 0;
    uint64_t mut_id = 0;
    if (err == CL_SUCCESS && mutable_handle != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        mut_id = g_mutable_cmd_ids.create_fresh(*mutable_handle);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCommandCopyBufferToImageKHR", "call_id", cid, "cmdbuf_id", cb_id, "queue_id",
        q_id, "properties", perfetto::DynamicString(props_str), "src_mem_id", src_id, "dst_mem_id", dst_id,
        "src_offset", src_offset, "dst_origin", perfetto::DynamicString(do_str), "region",
        perfetto::DynamicString(reg_str), "wait_sync_points", perfetto::DynamicString(wait_sp_str), "ret_sync_point",
        ret_sp, "mutable_cmd_id", mut_id);
    return err;
}

static cl_int CL_API_CALL clkp_clCommandCopyImageKHR(cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, const cl_command_properties_khr *properties, cl_mem src_image, cl_mem dst_image,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        q_id = g_queue_ids.find(command_queue);
        src_id = g_mem_ids.get_or_create(src_image);
        dst_id = g_mem_ids.get_or_create(dst_image);
    }
    std::string props_str = properties_to_string(properties);
    std::string so_str = size_array_to_string(src_origin, 3);
    std::string do_str = size_array_to_string(dst_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    std::string wait_sp_str = sync_point_list_to_string(num_sync_points_in_wait_list, sync_point_wait_list);

    cl_int err = g_ext.clCommandCopyImageKHR(command_buffer, command_queue, properties, src_image, dst_image,
        src_origin, dst_origin, region, num_sync_points_in_wait_list, sync_point_wait_list, sync_point, mutable_handle);

    uint64_t ret_sp = (err == CL_SUCCESS && sync_point != nullptr) ? static_cast<uint64_t>(*sync_point) : 0;
    uint64_t mut_id = 0;
    if (err == CL_SUCCESS && mutable_handle != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        mut_id = g_mutable_cmd_ids.create_fresh(*mutable_handle);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCommandCopyImageKHR", "call_id", cid, "cmdbuf_id", cb_id, "queue_id", q_id,
        "properties", perfetto::DynamicString(props_str), "src_mem_id", src_id, "dst_mem_id", dst_id, "src_origin",
        perfetto::DynamicString(so_str), "dst_origin", perfetto::DynamicString(do_str), "region",
        perfetto::DynamicString(reg_str), "wait_sync_points", perfetto::DynamicString(wait_sp_str), "ret_sync_point",
        ret_sp, "mutable_cmd_id", mut_id);
    return err;
}

static cl_int CL_API_CALL clkp_clCommandCopyImageToBufferKHR(cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, const cl_command_properties_khr *properties, cl_mem src_image, cl_mem dst_buffer,
    const size_t *src_origin, const size_t *region, size_t dst_offset, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    uint64_t q_id;
    uint64_t src_id;
    uint64_t dst_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        q_id = g_queue_ids.find(command_queue);
        src_id = g_mem_ids.get_or_create(src_image);
        dst_id = g_mem_ids.get_or_create(dst_buffer);
    }
    std::string props_str = properties_to_string(properties);
    std::string so_str = size_array_to_string(src_origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    std::string wait_sp_str = sync_point_list_to_string(num_sync_points_in_wait_list, sync_point_wait_list);

    cl_int err = g_ext.clCommandCopyImageToBufferKHR(command_buffer, command_queue, properties, src_image, dst_buffer,
        src_origin, region, dst_offset, num_sync_points_in_wait_list, sync_point_wait_list, sync_point, mutable_handle);

    uint64_t ret_sp = (err == CL_SUCCESS && sync_point != nullptr) ? static_cast<uint64_t>(*sync_point) : 0;
    uint64_t mut_id = 0;
    if (err == CL_SUCCESS && mutable_handle != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        mut_id = g_mutable_cmd_ids.create_fresh(*mutable_handle);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCommandCopyImageToBufferKHR", "call_id", cid, "cmdbuf_id", cb_id, "queue_id",
        q_id, "properties", perfetto::DynamicString(props_str), "src_mem_id", src_id, "dst_mem_id", dst_id,
        "src_origin", perfetto::DynamicString(so_str), "region", perfetto::DynamicString(reg_str), "dst_offset",
        dst_offset, "wait_sync_points", perfetto::DynamicString(wait_sp_str), "ret_sync_point", ret_sp,
        "mutable_cmd_id", mut_id);
    return err;
}

static cl_int CL_API_CALL clkp_clCommandFillBufferKHR(cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, const cl_command_properties_khr *properties, cl_mem buffer, const void *pattern,
    size_t pattern_size, size_t offset, size_t size, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    uint64_t q_id;
    uint64_t m_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        q_id = g_queue_ids.find(command_queue);
        m_id = g_mem_ids.get_or_create(buffer);
    }
    std::string props_str = properties_to_string(properties);
    std::string pat_hex = clkp::bytes_to_hex(pattern, pattern_size);
    std::string wait_sp_str = sync_point_list_to_string(num_sync_points_in_wait_list, sync_point_wait_list);

    cl_int err = g_ext.clCommandFillBufferKHR(command_buffer, command_queue, properties, buffer, pattern, pattern_size,
        offset, size, num_sync_points_in_wait_list, sync_point_wait_list, sync_point, mutable_handle);

    uint64_t ret_sp = (err == CL_SUCCESS && sync_point != nullptr) ? static_cast<uint64_t>(*sync_point) : 0;
    uint64_t mut_id = 0;
    if (err == CL_SUCCESS && mutable_handle != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        mut_id = g_mutable_cmd_ids.create_fresh(*mutable_handle);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCommandFillBufferKHR", "call_id", cid, "cmdbuf_id", cb_id, "queue_id", q_id,
        "properties", perfetto::DynamicString(props_str), "mem_id", m_id, "pattern_hex",
        perfetto::DynamicString(pat_hex), "pattern_size", pattern_size, "offset", offset, "size", size,
        "wait_sync_points", perfetto::DynamicString(wait_sp_str), "ret_sync_point", ret_sp, "mutable_cmd_id", mut_id);
    return err;
}

static cl_int CL_API_CALL clkp_clCommandFillImageKHR(cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, const cl_command_properties_khr *properties, cl_mem image, const void *fill_color,
    const size_t *origin, const size_t *region, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    uint64_t q_id;
    uint64_t m_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        q_id = g_queue_ids.find(command_queue);
        m_id = g_mem_ids.get_or_create(image);
    }
    std::string props_str = properties_to_string(properties);
    std::string color_hex = clkp::bytes_to_hex(fill_color, 16);
    std::string orig_str = size_array_to_string(origin, 3);
    std::string reg_str = size_array_to_string(region, 3);
    std::string wait_sp_str = sync_point_list_to_string(num_sync_points_in_wait_list, sync_point_wait_list);

    cl_int err = g_ext.clCommandFillImageKHR(command_buffer, command_queue, properties, image, fill_color, origin,
        region, num_sync_points_in_wait_list, sync_point_wait_list, sync_point, mutable_handle);

    uint64_t ret_sp = (err == CL_SUCCESS && sync_point != nullptr) ? static_cast<uint64_t>(*sync_point) : 0;
    uint64_t mut_id = 0;
    if (err == CL_SUCCESS && mutable_handle != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        mut_id = g_mutable_cmd_ids.create_fresh(*mutable_handle);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCommandFillImageKHR", "call_id", cid, "cmdbuf_id", cb_id, "queue_id", q_id,
        "properties", perfetto::DynamicString(props_str), "mem_id", m_id, "fill_color_hex",
        perfetto::DynamicString(color_hex), "origin", perfetto::DynamicString(orig_str), "region",
        perfetto::DynamicString(reg_str), "wait_sync_points", perfetto::DynamicString(wait_sp_str), "ret_sync_point",
        ret_sp, "mutable_cmd_id", mut_id);
    return err;
}

static cl_int CL_API_CALL clkp_clCommandNDRangeKernelKHR(cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, const cl_command_properties_khr *properties, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size,
    cl_uint num_sync_points_in_wait_list, const cl_sync_point_khr *sync_point_wait_list, cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle)
{
    uint64_t cid = next_call_id();
    uint64_t did = next_dispatch_id();
    uint64_t cb_id;
    uint64_t q_id;
    uint64_t k_id;
    std::string kernel_name;
    uint64_t prog_id;
    std::string program_string;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
        q_id = g_queue_ids.find(command_queue);
        k_id = g_kernel_ids.get_or_create(kernel);
        kernel_name = kernel_to_kernel_name.count(kernel) ? kernel_to_kernel_name[kernel] : "?";
        cl_program prog = kernel_to_program.count(kernel) ? kernel_to_program[kernel] : nullptr;
        prog_id = g_program_ids.find(prog);
        program_string = (prog && program_to_string.count(prog)) ? program_to_string[prog] : "clkp_p?";
    }

    size_t gidX = (work_dim > 0 && global_work_size) ? global_work_size[0] : 1;
    size_t gidY = (work_dim > 1 && global_work_size) ? global_work_size[1] : 1;
    size_t gidZ = (work_dim > 2 && global_work_size) ? global_work_size[2] : 1;

    std::string props_str = properties_to_string(properties);
    std::string gwo_str = size_array_to_string(global_work_offset, work_dim);
    std::string gws_str = size_array_to_string(global_work_size, work_dim);
    std::string lws_str = size_array_to_string(local_work_size, work_dim);
    std::string wait_sp_str = sync_point_list_to_string(num_sync_points_in_wait_list, sync_point_wait_list);

    cl_int err = g_ext.clCommandNDRangeKernelKHR(command_buffer, command_queue, properties, kernel, work_dim,
        global_work_offset, global_work_size, local_work_size, num_sync_points_in_wait_list, sync_point_wait_list,
        sync_point, mutable_handle);

    uint64_t ret_sp = (err == CL_SUCCESS && sync_point != nullptr) ? static_cast<uint64_t>(*sync_point) : 0;
    uint64_t mut_id = 0;
    if (err == CL_SUCCESS && mutable_handle != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        mut_id = g_mutable_cmd_ids.create_fresh(*mutable_handle);
    }

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCommandNDRangeKernelKHR", "call_id", cid, "dispatch_id", did, "cmdbuf_id",
        cb_id, "queue_id", q_id, "properties", perfetto::DynamicString(props_str), "kernel_id", k_id, "program_id",
        prog_id, "program", perfetto::DynamicString(program_string), "kernel_name",
        perfetto::DynamicString(kernel_name), "work_dim", work_dim, "gidX", gidX, "gidY", gidY, "gidZ", gidZ, "gwo",
        perfetto::DynamicString(gwo_str), "gws", perfetto::DynamicString(gws_str), "lws",
        perfetto::DynamicString(lws_str), "wait_sync_points", perfetto::DynamicString(wait_sp_str), "ret_sync_point",
        ret_sp, "mutable_cmd_id", mut_id);
    return err;
}

static cl_int CL_API_CALL clkp_clGetCommandBufferInfoKHR(cl_command_buffer_khr command_buffer,
    cl_command_buffer_info_khr param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret)
{
    return g_ext.clGetCommandBufferInfoKHR(
        command_buffer, param_name, param_value_size, param_value, param_value_size_ret);
}

static cl_int CL_API_CALL clkp_clUpdateMutableCommandsKHR(cl_command_buffer_khr command_buffer, cl_uint num_configs,
    const cl_command_buffer_update_type_khr *config_types, const void **configs)
{
    uint64_t cid = next_call_id();
    uint64_t cb_id;
    {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        cb_id = g_cmdbuf_ids.get_or_create(command_buffer);
    }
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clUpdateMutableCommandsKHR", "call_id", cid, "cmdbuf_id", cb_id, "num_configs",
        num_configs);

    if (config_types != nullptr && configs != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        for (cl_uint i = 0; i < num_configs; ++i) {
            if (config_types[i] == CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR && configs[i] != nullptr) {
                const auto *cfg = reinterpret_cast<const cl_mutable_dispatch_config_khr *>(configs[i]);
                uint64_t mut_id = g_mutable_cmd_ids.get_or_create(cfg->command);
                cl_uint dim = cfg->work_dim > 0 ? cfg->work_dim : 3;
                std::string gwo_str = size_array_to_string(cfg->global_work_offset, dim);
                std::string gws_str = size_array_to_string(cfg->global_work_size, dim);
                std::string lws_str = size_array_to_string(cfg->local_work_size, dim);
                TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clUpdateMutableCommandsKHR-config", "call_id", cid,
                    "cmdbuf_id", cb_id, "mutable_cmd_id", mut_id, "work_dim", cfg->work_dim, "gwo",
                    perfetto::DynamicString(gwo_str), "gws", perfetto::DynamicString(gws_str), "lws",
                    perfetto::DynamicString(lws_str), "num_args", cfg->num_args);

                if (cfg->arg_list != nullptr) {
                    for (cl_uint j = 0; j < cfg->num_args; ++j) {
                        const auto &arg = cfg->arg_list[j];
                        std::string arg_kind;
                        uint64_t mem_id = 0;
                        uint64_t sampler_id = 0;
                        std::string val_hex;
                        classify_kernel_arg(arg.arg_size, arg.arg_value, arg_kind, mem_id, sampler_id, val_hex);
                        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clUpdateMutableCommandsKHR-arg", "call_id", cid,
                            "cmdbuf_id", cb_id, "mutable_cmd_id", mut_id, "arg_index", arg.arg_index, "arg_size",
                            arg.arg_size, "arg_kind", perfetto::DynamicString(arg_kind), "mem_id", mem_id, "sampler_id",
                            sampler_id, "value_hex", perfetto::DynamicString(val_hex));
                    }
                }
            }
        }
    }

    return g_ext.clUpdateMutableCommandsKHR(command_buffer, num_configs, config_types, configs);
}

static cl_int CL_API_CALL clkp_clGetMutableCommandInfoKHR(cl_mutable_command_khr command,
    cl_mutable_command_info_khr param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret)
{
    return g_ext.clGetMutableCommandInfoKHR(command, param_name, param_value_size, param_value, param_value_size_ret);
}

static cl_program CL_API_CALL clkp_clCreateProgramWithILKHR(
    cl_context context, const void *il, size_t length, cl_int *errcode_ret)
{
    return create_program_with_il_common(context, il, length, errcode_ret, g_ext.clCreateProgramWithILKHR);
}

static cl_command_queue CL_API_CALL clkp_clCreateCommandQueueWithPropertiesKHR(
    cl_context context, cl_device_id device, const cl_queue_properties_khr *properties, cl_int *errcode_ret)
{
    return clkp_clCreateCommandQueueWithProperties(context, device, properties, errcode_ret);
}

static void *wrap_extension_function(const char *func_name, void *real_fn)
{
    if (func_name == nullptr || real_fn == nullptr) {
        return real_fn;
    }
    std::lock_guard<std::recursive_mutex> lock(g_lock);
#define HANDLE_EXT(name)                                                                                               \
    if (strcmp(func_name, #name) == 0) {                                                                               \
        g_ext.name = reinterpret_cast<name##_fn>(real_fn);                                                             \
        return reinterpret_cast<void *>(&clkp_##name);                                                                 \
    }
    HANDLE_EXT(clCreateCommandBufferKHR)
    HANDLE_EXT(clFinalizeCommandBufferKHR)
    HANDLE_EXT(clRetainCommandBufferKHR)
    HANDLE_EXT(clReleaseCommandBufferKHR)
    HANDLE_EXT(clEnqueueCommandBufferKHR)
    HANDLE_EXT(clCommandBarrierWithWaitListKHR)
    HANDLE_EXT(clCommandCopyBufferKHR)
    HANDLE_EXT(clCommandCopyBufferRectKHR)
    HANDLE_EXT(clCommandCopyBufferToImageKHR)
    HANDLE_EXT(clCommandCopyImageKHR)
    HANDLE_EXT(clCommandCopyImageToBufferKHR)
    HANDLE_EXT(clCommandFillBufferKHR)
    HANDLE_EXT(clCommandFillImageKHR)
    HANDLE_EXT(clCommandNDRangeKernelKHR)
    HANDLE_EXT(clGetCommandBufferInfoKHR)
    HANDLE_EXT(clUpdateMutableCommandsKHR)
    HANDLE_EXT(clGetMutableCommandInfoKHR)
    HANDLE_EXT(clCreateProgramWithILKHR)
    HANDLE_EXT(clCreateCommandQueueWithPropertiesKHR)
#undef HANDLE_EXT
    return real_fn;
}

static void *clkp_clGetExtensionFunctionAddress(const char *func_name)
{
    void *real_fn
        = tdispatch->clGetExtensionFunctionAddress ? tdispatch->clGetExtensionFunctionAddress(func_name) : nullptr;
    return wrap_extension_function(func_name, real_fn);
}

static void *clkp_clGetExtensionFunctionAddressForPlatform(cl_platform_id platform, const char *func_name)
{
    void *real_fn = nullptr;
    if (tdispatch->clGetExtensionFunctionAddressForPlatform) {
        real_fn = tdispatch->clGetExtensionFunctionAddressForPlatform(platform, func_name);
    }
    if (real_fn == nullptr && tdispatch->clGetExtensionFunctionAddress) {
        real_fn = tdispatch->clGetExtensionFunctionAddress(func_name);
    }
    return wrap_extension_function(func_name, real_fn);
}

/*****************************************************************************/
/* PERFETTO TRACE PARAMETERS *************************************************/
/*****************************************************************************/

#ifdef BACKEND_INPROCESS
static const char *get_trace_dest()
{
    if (auto trace_dest = getenv("CLKP_TRACE_DEST")) {
        return trace_dest;
    }
    return TRACE_DEST;
}

static const uint32_t get_trace_max_size()
{
    if (auto trace_max_size = getenv("CLKP_TRACE_MAX_SIZE")) {
        return atoi(trace_max_size);
    }
    return TRACE_MAX_SIZE;
}
#endif

/*****************************************************************************/
/* LAYER FUNCTIONS ***********************************************************/
/*****************************************************************************/

CL_API_ENTRY cl_int CL_API_CALL clGetLayerInfo(
    cl_layer_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret)
{
    switch (param_name) {
    case CL_LAYER_API_VERSION:
        if (param_value) {
            if (param_value_size < sizeof(cl_layer_api_version))
                return CL_INVALID_VALUE;
            *((cl_layer_api_version *)param_value) = CL_LAYER_API_VERSION_100;
        }
        if (param_value_size_ret)
            *param_value_size_ret = sizeof(cl_layer_api_version);
        break;
    default:
        return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clDeinitLayer()
{
#ifdef BACKEND_INPROCESS
    gTracingSession->StopBlocking();
    std::vector<char> trace_data(gTracingSession->ReadTraceBlocking());
    delete gTracingSession;

    std::ofstream output;
    output.open(get_trace_dest(), std::ios::out | std::ios::binary);
    output.write(&trace_data[0], trace_data.size());
    output.close();
#else
    perfetto::TrackEvent::Flush();
#endif
    return 0;
}

CL_API_ENTRY cl_int CL_API_CALL clInitLayerWithProperties(cl_uint num_entries, const cl_icd_dispatch *target_dispatch,
    cl_uint *num_entries_out, const cl_icd_dispatch **layer_dispatch_ret, const cl_layer_properties *properties)
{
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (!target_dispatch || !num_entries_out || !layer_dispatch_ret)
        return CL_INVALID_VALUE;
    if (num_entries < sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs))
        return CL_INVALID_VALUE;

    perfetto::TracingInitArgs args;
#ifdef BACKEND_INPROCESS
    args.backends |= perfetto::kInProcessBackend;
#else
    args.backends |= perfetto::kSystemBackend;
#endif
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

#ifdef BACKEND_INPROCESS
    perfetto::protos::gen::TrackEventConfig track_event_cfg;
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(get_trace_max_size());
    auto *ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    gTracingSession = perfetto::Tracing::NewTrace().release();
    gTracingSession->Setup(cfg);
    gTracingSession->StartBlocking();
#endif

    const uint32_t max_retry = 100;
    uint32_t retry = 0;
    while ((retry++ < max_retry) && !TRACE_EVENT_CATEGORY_ENABLED(CLKP_PERFETTO_CATEGORY)) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    if (!TRACE_EVENT_CATEGORY_ENABLED(CLKP_PERFETTO_CATEGORY)) {
        PRINT("perfetto category does not seem to be enabled");
    }

    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.clCreateContext = clkp_clCreateContext;
    dispatch.clCreateContextFromType = clkp_clCreateContextFromType;
    dispatch.clRetainContext = clkp_clRetainContext;
    dispatch.clReleaseContext = clkp_clReleaseContext;

    dispatch.clCreateCommandQueue = clkp_clCreateCommandQueue;
    dispatch.clCreateCommandQueueWithProperties = clkp_clCreateCommandQueueWithProperties;
    dispatch.clRetainCommandQueue = clkp_clRetainCommandQueue;
    dispatch.clReleaseCommandQueue = clkp_clReleaseCommandQueue;

    dispatch.clCreateBuffer = clkp_clCreateBuffer;
    dispatch.clCreateBufferWithProperties = clkp_clCreateBufferWithProperties;
    dispatch.clCreateSubBuffer = clkp_clCreateSubBuffer;
    dispatch.clCreateImage = clkp_clCreateImage;
    dispatch.clCreateImageWithProperties = clkp_clCreateImageWithProperties;
    dispatch.clCreateImage2D = clkp_clCreateImage2D;
    dispatch.clCreateImage3D = clkp_clCreateImage3D;
    dispatch.clRetainMemObject = clkp_clRetainMemObject;
    dispatch.clReleaseMemObject = clkp_clReleaseMemObject;

    dispatch.clCreateSampler = clkp_clCreateSampler;
    dispatch.clCreateSamplerWithProperties = clkp_clCreateSamplerWithProperties;
    dispatch.clRetainSampler = clkp_clRetainSampler;
    dispatch.clReleaseSampler = clkp_clReleaseSampler;

    dispatch.clCreateProgramWithSource = clkp_clCreateProgramWithSource;
    dispatch.clCreateProgramWithBinary = clkp_clCreateProgramWithBinary;
    dispatch.clCreateProgramWithIL = clkp_clCreateProgramWithIL;
    dispatch.clGetProgramInfo = clkp_clGetProgramInfo;
    dispatch.clBuildProgram = clkp_clBuildProgram;
    dispatch.clCompileProgram = clkp_clCompileProgram;
    dispatch.clLinkProgram = clkp_clLinkProgram;
    dispatch.clSetProgramSpecializationConstant = clkp_clSetProgramSpecializationConstant;
    dispatch.clRetainProgram = clkp_clRetainProgram;
    dispatch.clReleaseProgram = clkp_clReleaseProgram;

    dispatch.clCreateKernel = clkp_clCreateKernel;
    dispatch.clCreateKernelsInProgram = clkp_clCreateKernelsInProgram;
    dispatch.clCloneKernel = clkp_clCloneKernel;
    dispatch.clSetKernelArg = clkp_clSetKernelArg;
    dispatch.clRetainKernel = clkp_clRetainKernel;
    dispatch.clReleaseKernel = clkp_clReleaseKernel;

    dispatch.clEnqueueNDRangeKernel = clkp_clEnqueueNDRangeKernel;
    dispatch.clEnqueueTask = clkp_clEnqueueTask;
    dispatch.clEnqueueReadBuffer = clkp_clEnqueueReadBuffer;
    dispatch.clEnqueueWriteBuffer = clkp_clEnqueueWriteBuffer;
    dispatch.clEnqueueReadBufferRect = clkp_clEnqueueReadBufferRect;
    dispatch.clEnqueueWriteBufferRect = clkp_clEnqueueWriteBufferRect;
    dispatch.clEnqueueCopyBuffer = clkp_clEnqueueCopyBuffer;
    dispatch.clEnqueueCopyBufferRect = clkp_clEnqueueCopyBufferRect;
    dispatch.clEnqueueFillBuffer = clkp_clEnqueueFillBuffer;
    dispatch.clEnqueueReadImage = clkp_clEnqueueReadImage;
    dispatch.clEnqueueWriteImage = clkp_clEnqueueWriteImage;
    dispatch.clEnqueueCopyImage = clkp_clEnqueueCopyImage;
    dispatch.clEnqueueFillImage = clkp_clEnqueueFillImage;
    dispatch.clEnqueueCopyImageToBuffer = clkp_clEnqueueCopyImageToBuffer;
    dispatch.clEnqueueCopyBufferToImage = clkp_clEnqueueCopyBufferToImage;
    dispatch.clEnqueueMapBuffer = clkp_clEnqueueMapBuffer;
    dispatch.clEnqueueMapImage = clkp_clEnqueueMapImage;
    dispatch.clEnqueueUnmapMemObject = clkp_clEnqueueUnmapMemObject;
    dispatch.clEnqueueMigrateMemObjects = clkp_clEnqueueMigrateMemObjects;
    dispatch.clEnqueueMarkerWithWaitList = clkp_clEnqueueMarkerWithWaitList;
    dispatch.clEnqueueBarrierWithWaitList = clkp_clEnqueueBarrierWithWaitList;
    dispatch.clEnqueueMarker = clkp_clEnqueueMarker;
    dispatch.clEnqueueBarrier = clkp_clEnqueueBarrier;
    dispatch.clEnqueueWaitForEvents = clkp_clEnqueueWaitForEvents;

    dispatch.clFlush = clkp_clFlush;
    dispatch.clFinish = clkp_clFinish;
    dispatch.clWaitForEvents = clkp_clWaitForEvents;
    dispatch.clCreateUserEvent = clkp_clCreateUserEvent;
    dispatch.clSetUserEventStatus = clkp_clSetUserEventStatus;
    dispatch.clRetainEvent = clkp_clRetainEvent;
    dispatch.clReleaseEvent = clkp_clReleaseEvent;

    dispatch.clGetExtensionFunctionAddress = clkp_clGetExtensionFunctionAddress;
    dispatch.clGetExtensionFunctionAddressForPlatform = clkp_clGetExtensionFunctionAddressForPlatform;

    tdispatch = target_dispatch;
    *layer_dispatch_ret = &dispatch;
    *num_entries_out = sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs);

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clInitLayer(cl_uint num_entries, const struct _cl_icd_dispatch *target_dispatch,
    cl_uint *num_entries_out, const struct _cl_icd_dispatch **layer_dispatch_ret)
{
    return clInitLayerWithProperties(num_entries, target_dispatch, num_entries_out, layer_dispatch_ret, nullptr);
}
