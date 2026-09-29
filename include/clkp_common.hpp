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

#ifndef CLKP_COMMON_HPP
#define CLKP_COMMON_HPP

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifdef CLKP_PERFETTO_AMALGAMATED
#include "perfetto.h"
#else
#include "perfetto/tracing.h"
#include "protos/perfetto/trace/interned_data/interned_data.pbzero.h"
#include "protos/perfetto/trace/profiling/profile_common.pbzero.h"
#include "protos/perfetto/trace/trace.pbzero.h"
#include "protos/perfetto/trace/trace_packet.pbzero.h"
#include "protos/perfetto/trace/trace_packet_defaults.pbzero.h"
#include "protos/perfetto/trace/track_event/debug_annotation.pbzero.h"
#include "protos/perfetto/trace/track_event/track_descriptor.pbzero.h"
#include "protos/perfetto/trace/track_event/track_event.pbzero.h"
#endif

#define CLKP_PERFETTO_CATEGORY "clkp"

namespace clkp {

// Chunk size (in bytes) for splitting large source strings or binary blobs across TrackEvents.
static constexpr size_t kMaxPayloadChunkBytes = 4096;

inline std::string bytes_to_hex(const void *data, size_t len)
{
    if (data == nullptr || len == 0) {
        return "";
    }
    static const char kHexDigits[] = "0123456789abcdef";
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    std::string out;
    out.resize(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out[i * 2] = kHexDigits[(bytes[i] >> 4) & 0xF];
        out[i * 2 + 1] = kHexDigits[bytes[i] & 0xF];
    }
    return out;
}

inline int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return 0;
}

inline std::vector<uint8_t> hex_to_bytes(const std::string &hex)
{
    std::vector<uint8_t> out;
    size_t count = hex.size() / 2;
    out.resize(count);
    for (size_t i = 0; i < count; ++i) {
        int hi = hex_nibble(hex[i * 2]);
        int lo = hex_nibble(hex[i * 2 + 1]);
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return out;
}

inline std::string uint64_list_to_string(const std::vector<uint64_t> &values)
{
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    return oss.str();
}

inline std::vector<uint64_t> string_to_uint64_list(const std::string &str)
{
    std::vector<uint64_t> result;
    if (str.empty()) {
        return result;
    }
    size_t start = 0;
    while (start < str.size()) {
        size_t comma = str.find(',', start);
        std::string token = (comma == std::string::npos) ? str.substr(start) : str.substr(start, comma - start);
        if (!token.empty()) {
            result.push_back(std::strtoull(token.c_str(), nullptr, 10));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return result;
}

inline std::string string_list_to_joined(const std::vector<std::string> &values, char sep = ';')
{
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << sep;
        }
        oss << values[i];
    }
    return oss.str();
}

inline std::vector<std::string> joined_to_string_list(const std::string &str, char sep = ';')
{
    std::vector<std::string> result;
    if (str.empty()) {
        return result;
    }
    size_t start = 0;
    while (start <= str.size()) {
        size_t pos = str.find(sep, start);
        if (pos == std::string::npos) {
            result.push_back(str.substr(start));
            break;
        }
        result.push_back(str.substr(start, pos - start));
        start = pos + 1;
    }
    return result;
}

struct ParsedAnnotation {
    enum class ValueType {
        kNone,
        kBool,
        kUint,
        kInt,
        kDouble,
        kPointer,
        kString,
    };

    std::string name;
    ValueType type = ValueType::kNone;
    bool bool_val = false;
    uint64_t uint_val = 0;
    int64_t int_val = 0;
    double double_val = 0.0;
    std::string string_val;
};

struct ParsedEvent {
    uint64_t timestamp = 0;
    uint32_t sequence_id = 0;
    uint64_t track_uuid = 0;
    int32_t event_type = 0; // TrackEvent::Type (1=SLICE_BEGIN, 2=SLICE_END, 3=INSTANT)
    std::string name;
    std::vector<std::string> categories;
    std::vector<ParsedAnnotation> annotations;
    size_t packet_index = 0;

    bool has_category(const char *cat) const
    {
        for (const auto &c : categories) {
            if (c == cat) {
                return true;
            }
        }
        return false;
    }

    const ParsedAnnotation *find_annotation(const char *key) const
    {
        for (const auto &ann : annotations) {
            if (ann.name == key) {
                return &ann;
            }
        }
        return nullptr;
    }

    bool has_key(const char *key) const { return find_annotation(key) != nullptr; }

    uint64_t get_uint(const char *key, uint64_t default_val = 0) const
    {
        const auto *ann = find_annotation(key);
        if (!ann) {
            return default_val;
        }
        switch (ann->type) {
        case ParsedAnnotation::ValueType::kUint:
        case ParsedAnnotation::ValueType::kPointer:
            return ann->uint_val;
        case ParsedAnnotation::ValueType::kInt:
            return static_cast<uint64_t>(ann->int_val);
        case ParsedAnnotation::ValueType::kBool:
            return ann->bool_val ? 1 : 0;
        case ParsedAnnotation::ValueType::kString:
            return std::strtoull(ann->string_val.c_str(), nullptr, 0);
        default:
            return default_val;
        }
    }

    int64_t get_int(const char *key, int64_t default_val = 0) const
    {
        const auto *ann = find_annotation(key);
        if (!ann) {
            return default_val;
        }
        switch (ann->type) {
        case ParsedAnnotation::ValueType::kInt:
            return ann->int_val;
        case ParsedAnnotation::ValueType::kUint:
        case ParsedAnnotation::ValueType::kPointer:
            return static_cast<int64_t>(ann->uint_val);
        case ParsedAnnotation::ValueType::kBool:
            return ann->bool_val ? 1 : 0;
        case ParsedAnnotation::ValueType::kString:
            return std::strtoll(ann->string_val.c_str(), nullptr, 0);
        default:
            return default_val;
        }
    }

    std::string get_string(const char *key, const std::string &default_val = "") const
    {
        const auto *ann = find_annotation(key);
        if (!ann) {
            return default_val;
        }
        if (ann->type == ParsedAnnotation::ValueType::kString) {
            return ann->string_val;
        }
        if (ann->type == ParsedAnnotation::ValueType::kUint || ann->type == ParsedAnnotation::ValueType::kPointer) {
            return std::to_string(ann->uint_val);
        }
        if (ann->type == ParsedAnnotation::ValueType::kInt) {
            return std::to_string(ann->int_val);
        }
        return default_val;
    }
};

struct SequenceInternTable {
    std::unordered_map<uint64_t, std::string> categories;
    std::unordered_map<uint64_t, std::string> event_names;
    std::unordered_map<uint64_t, std::string> annotation_names;
    std::unordered_map<uint64_t, std::string> annotation_strings;
    uint64_t default_track_uuid = 0;
    uint32_t default_clock_id = 0;

    void clear()
    {
        categories.clear();
        event_names.clear();
        annotation_names.clear();
        annotation_strings.clear();
        default_track_uuid = 0;
        default_clock_id = 0;
    }
};

class TraceIncrementState {
public:
    SequenceInternTable &get_sequence(uint32_t seq_id) { return sequences_[seq_id]; }

    void update_from_packet(const perfetto::protos::pbzero::TracePacket_Decoder &pkt)
    {
        uint32_t seq_id = pkt.has_trusted_packet_sequence_id() ? pkt.trusted_packet_sequence_id() : 0;
        auto &seq = sequences_[seq_id];

        // SequenceFlags::SEQ_INCREMENTAL_STATE_CLEARED == 1
        if ((pkt.has_sequence_flags() && (pkt.sequence_flags() & 1u))
            || (pkt.has_incremental_state_cleared() && pkt.incremental_state_cleared())) {
            seq.clear();
        }

        if (pkt.has_trace_packet_defaults()) {
            perfetto::protos::pbzero::TracePacketDefaults_Decoder defs(pkt.trace_packet_defaults());
            if (defs.has_timestamp_clock_id()) {
                seq.default_clock_id = defs.timestamp_clock_id();
            }
            if (defs.has_track_event_defaults()) {
                perfetto::protos::pbzero::TrackEventDefaults_Decoder ted(defs.track_event_defaults());
                if (ted.has_track_uuid()) {
                    seq.default_track_uuid = ted.track_uuid();
                }
            }
        }

        if (pkt.has_interned_data()) {
            perfetto::protos::pbzero::InternedData_Decoder idata(pkt.interned_data());
            for (auto it = idata.event_categories(); it; ++it) {
                perfetto::protos::pbzero::EventCategory_Decoder cat(*it);
                if (cat.has_iid() && cat.has_name()) {
                    seq.categories[cat.iid()] = cat.name().ToStdString();
                }
            }
            for (auto it = idata.event_names(); it; ++it) {
                perfetto::protos::pbzero::EventName_Decoder en(*it);
                if (en.has_iid() && en.has_name()) {
                    seq.event_names[en.iid()] = en.name().ToStdString();
                }
            }
            for (auto it = idata.debug_annotation_names(); it; ++it) {
                perfetto::protos::pbzero::DebugAnnotationName_Decoder an(*it);
                if (an.has_iid() && an.has_name()) {
                    seq.annotation_names[an.iid()] = an.name().ToStdString();
                }
            }
            for (auto it = idata.debug_annotation_string_values(); it; ++it) {
                perfetto::protos::pbzero::InternedString_Decoder is(*it);
                if (is.has_iid() && is.has_str()) {
                    auto bytes = is.str();
                    seq.annotation_strings[is.iid()]
                        = std::string(reinterpret_cast<const char *>(bytes.data), bytes.size);
                }
            }
        }
    }

    bool decode_track_event(
        const perfetto::protos::pbzero::TracePacket_Decoder &pkt, size_t packet_index, ParsedEvent &out_event)
    {
        if (!pkt.has_track_event()) {
            return false;
        }
        uint32_t seq_id = pkt.has_trusted_packet_sequence_id() ? pkt.trusted_packet_sequence_id() : 0;
        const auto &seq = sequences_[seq_id];

        perfetto::protos::pbzero::TrackEvent_Decoder te(pkt.track_event());
        out_event = ParsedEvent();
        out_event.packet_index = packet_index;
        out_event.sequence_id = seq_id;
        out_event.timestamp = pkt.has_timestamp() ? pkt.timestamp() : 0;
        out_event.event_type = te.has_type() ? te.type() : 0;
        out_event.track_uuid = te.has_track_uuid() ? te.track_uuid() : seq.default_track_uuid;

        for (auto it = te.categories(); it; ++it) {
            out_event.categories.push_back((*it).ToStdString());
        }
        for (auto it = te.category_iids(); it; ++it) {
            uint64_t iid = *it;
            auto found = seq.categories.find(iid);
            if (found != seq.categories.end()) {
                out_event.categories.push_back(found->second);
            }
        }

        if (te.has_name()) {
            out_event.name = te.name().ToStdString();
        } else if (te.has_name_iid()) {
            auto found = seq.event_names.find(te.name_iid());
            if (found != seq.event_names.end()) {
                out_event.name = found->second;
            }
        }

        for (auto it = te.debug_annotations(); it; ++it) {
            perfetto::protos::pbzero::DebugAnnotation_Decoder da(*it);
            ParsedAnnotation ann;
            if (da.has_name()) {
                ann.name = da.name().ToStdString();
            } else if (da.has_name_iid()) {
                auto found = seq.annotation_names.find(da.name_iid());
                if (found != seq.annotation_names.end()) {
                    ann.name = found->second;
                }
            }

            if (da.has_string_value()) {
                ann.type = ParsedAnnotation::ValueType::kString;
                ann.string_val = da.string_value().ToStdString();
            } else if (da.has_string_value_iid()) {
                ann.type = ParsedAnnotation::ValueType::kString;
                auto found = seq.annotation_strings.find(da.string_value_iid());
                if (found != seq.annotation_strings.end()) {
                    ann.string_val = found->second;
                }
            } else if (da.has_uint_value()) {
                ann.type = ParsedAnnotation::ValueType::kUint;
                ann.uint_val = da.uint_value();
            } else if (da.has_int_value()) {
                ann.type = ParsedAnnotation::ValueType::kInt;
                ann.int_val = da.int_value();
            } else if (da.has_pointer_value()) {
                ann.type = ParsedAnnotation::ValueType::kPointer;
                ann.uint_val = da.pointer_value();
            } else if (da.has_bool_value()) {
                ann.type = ParsedAnnotation::ValueType::kBool;
                ann.bool_val = da.bool_value();
            } else if (da.has_double_value()) {
                ann.type = ParsedAnnotation::ValueType::kDouble;
                ann.double_val = da.double_value();
            }
            out_event.annotations.push_back(std::move(ann));
        }

        return true;
    }

private:
    std::unordered_map<uint32_t, SequenceInternTable> sequences_;
};

inline bool read_file_bytes(const std::string &path, std::vector<uint8_t> &out)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }
    std::streamsize size = file.tellg();
    if (size < 0) {
        return false;
    }
    file.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (size > 0 && !file.read(reinterpret_cast<char *>(out.data()), size)) {
        return false;
    }
    return true;
}

} // namespace clkp

#endif // CLKP_COMMON_HPP
