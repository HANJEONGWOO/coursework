"""Audit every native operator while optionally returning compact completions."""
from pathlib import Path
import sys

root = Path(sys.argv[1])


def replace(relative, old, new):
    path = root / relative
    source = path.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f"Pinned streaming source mismatch: {relative}: {old[:70]!r}")
    path.write_text(source.replace(old, new))


replace("astra-sim/workload/Pa2Stream.hh",
        "struct Event { int rank; uint64_t node_id, start_ns, end_ns; bool output; };",
        "struct Event { int rank; uint64_t node_id, start_ns, end_ns; bool output; int resource; uint64_t release_ns; };")
replace("astra-sim/workload/Statistics.cc",
        '            node->has_attr("pa2_output") && node->get_attr_msg("pa2_output").bool_val()});\n',
        '            node->has_attr("pa2_output") && node->get_attr_msg("pa2_output").bool_val(),\n'
        '            (static_cast<int>(node->type()) == 4 && !node->is_cpu_op()) ? 0 :\n'
        '                ((static_cast<int>(node->type()) == 5 || static_cast<int>(node->type()) == 7) ? 1 : -1),\n'
        '            node->has_attr("pa2_release_ns") ? node->get_attr_msg("pa2_release_ns").uint64_val() : 0});\n')
main = "astra-sim/network_frontend/analytical/congestion_unaware/main.cc"
replace(main, "#include <cerrno>\n", "#include <cerrno>\n#include <iomanip>\n#include <sstream>\n#include <algorithm>\n")
replace(main, '        auto respond = [&](const std::string& command) {\n',
        r'''        const bool pa2_compact = std::getenv("PA2_STREAM_COMPACT") && std::string(std::getenv("PA2_STREAM_COMPACT")) == "1";
        uint64_t pa2_submitted_nodes = pending();
        uint64_t pa2_completed_nodes = 0;
        uint64_t pa2_event_hash = UINT64_C(14695981039346656037);
        struct Pa2RankAudit { uint64_t last_end[2] = {0,0}; uint64_t busy[2] = {0,0}; };
        std::map<int,Pa2RankAudit> pa2_resource_audit;
        auto respond = [&](const std::string& command) {
''')
replace(main, '            json event_records = json::array(), outputs = json::array();\n',
        '            json event_records = json::array(), outputs = json::array();\n'
        '            std::map<int,std::vector<uint64_t>> pa2_completed_ids;\n')
replace(main,
        r'''                json record = {{"rank",event.rank},{"node_id",event.node_id},
                    {"start_ns",event.start_ns},{"end_ns",event.end_ns}};
                event_records.push_back(record);
                if (event.output) outputs.push_back(record);
''',
        r'''                if (event.start_ns < event.release_ns || event.end_ns < event.start_ns ||
                    event.end_ns > event_queue->get_current_time())
                    throw std::runtime_error("PA2 native event violates release or clock boundary");
                if (event.resource >= 0) {
                    auto& rank_audit = pa2_resource_audit[event.rank];
                    if (event.start_ns < rank_audit.last_end[event.resource])
                        throw std::runtime_error("PA2 native resource intervals overlap");
                    rank_audit.last_end[event.resource] = event.end_ns;
                    rank_audit.busy[event.resource] += event.end_ns-event.start_ns;
                }
                ++pa2_completed_nodes;
                // FNV-1a-64 is an order-sensitive reproducibility checksum,
                // not a cryptographic authentication mechanism. Each tuple
                // uses four unsigned 64-bit integers in little-endian order.
                for (uint64_t value : {uint64_t(event.rank),event.node_id,event.start_ns,event.end_ns}) {
                    for (int byte = 0; byte < 8; ++byte) {
                        pa2_event_hash ^= value & UINT64_C(255);
                        pa2_event_hash *= UINT64_C(1099511628211);
                        value >>= 8;
                    }
                }
                if (!pa2_compact || event.output) {
                    json record = {{"rank",event.rank},{"node_id",event.node_id},
                        {"start_ns",event.start_ns},{"end_ns",event.end_ns}};
                    event_records.push_back(record);
                    if (event.output) outputs.push_back(record);
                }
                if (pa2_compact) pa2_completed_ids[event.rank].push_back(event.node_id);
''')
replace(main, '            pa2_emit(reply.dump());\n',
        r'''            json ranges = json::object();
            for (auto& item : pa2_completed_ids) {
                auto& ids = item.second;
                std::sort(ids.begin(),ids.end());
                json intervals = json::array();
                uint64_t first = ids.front(), last = first;
                for (size_t i = 1; i < ids.size(); ++i) {
                    if (last != UINT64_MAX && ids[i] == last+1) { last=ids[i]; continue; }
                    intervals.push_back({first,last}); first=last=ids[i];
                }
                intervals.push_back({first,last});
                ranges[std::to_string(item.first)] = intervals;
            }
            json by_rank = json::object();
            for (const auto& item : pa2_resource_audit) {
                json row = json::object();
                if (item.second.busy[0]) row["compute_busy_ns"] = item.second.busy[0];
                if (item.second.busy[1]) row["communication_busy_ns"] = item.second.busy[1];
                by_rank[std::to_string(item.first)] = row;
            }
            std::ostringstream checksum;
            checksum << std::hex << std::setfill('0') << std::setw(16) << pa2_event_hash;
            reply["compact"] = pa2_compact;
            reply["streaming_compact_version"] = "pa2-compact-v1";
            reply["completed_ranges"] = ranges;
            reply["audit"] = {{"submitted_nodes",pa2_submitted_nodes},{"completed_nodes",pa2_completed_nodes},
                {"clock_ns",event_queue->get_current_time()},{"events_fnv1a64",checksum.str()},
                {"event_checksum_algorithm","fnv1a64-uint64-le-rank-node-start-end-noncryptographic"},
                {"resource_intervals_verified",true},{"release_times_verified",true},
                {"resource_overlap_violations",0},{"by_rank",by_rank}};
            pa2_emit(reply.dump());
''')
replace(main, '                const auto request = json::parse(line);\n',
        '                const auto request = json::parse(line);\n'
        '                const auto pa2_pending_before = pending();\n')
replace(main, '\n                respond(command);\n',
        '\n                if (command == "append" || command == "instantiate")\n'
        '                    pa2_submitted_nodes += pending() - pa2_pending_before;\n'
        '                respond(command);\n')
print("Applied pa2-compact-v1 (all native operations still executed and audited)")
