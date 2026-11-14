"""Add a persistent, causal PA2 command loop to the pinned ASTRA source.

Apply instrument_astra.py first. This patch is used only by a separate
runtime_async build; the archived synchronous binary remains untouched.
"""
from pathlib import Path
import sys

root = Path(sys.argv[1])


def replace(path, old, new):
    path = root / path
    source = path.read_text()
    if new in source:
        return
    if old.startswith("    // run simulation\n") and "// PA2 streaming mode keeps" in source:
        return  # A newer incremental patch already installed this driver.
    if old == "using namespace AstraSim;\n" and '#include "astra-sim/workload/Pa2Stream.hh"' in source:
        return
    if old == "  // legacy interface\n" and "void pa2_append_trace" in source:
        return
    if old == "  std::ifstream chakra_file;\n" and "pa2_nodes;" in source:
        return
    if old.startswith("    // PA2: emitted only") and "Pa2Stream::events()" in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f"Pinned source does not match patch: {path}: {old[:70]!r}")
    path.write_text(source.replace(old, new))


feeder = "extern/graph_frontend/chakra/src/feeder_v3/"
# Both classes expose the same incremental resolution operation.
path = root / feeder / "dependancy_solver.h"
source = path.read_text()
if "pa2_resolve_incremental" not in source:
    assert source.count("  void resolve_dependancy_free_nodes();") == 2
    source = source.replace("  void resolve_dependancy_free_nodes();",
                            "  void resolve_dependancy_free_nodes();\n  void pa2_resolve_incremental();")
    source = source.replace("  bool enable_data_deps;", "  std::unordered_set<NodeId> pa2_finished_nodes;\n  bool enable_data_deps;")
    path.write_text(source)
replace(feeder + "dependancy_solver.cpp",
        "  for (auto& parent : node.data_deps()) {\n",
        "  for (auto& parent : node.data_deps()) {\n    if (pa2_finished_nodes.count(parent)) continue;\n")
replace(feeder + "dependancy_solver.cpp",
        "  for (auto& parent : node.ctrl_deps()) {\n",
        "  for (auto& parent : node.ctrl_deps()) {\n    if (pa2_finished_nodes.count(parent)) continue;\n")
replace(feeder + "dependancy_solver.cpp",
        "  this->enabled_dependancy.finish_node(node);\n",
        "  this->enabled_dependancy.finish_node(node);\n  this->pa2_finished_nodes.insert(node);\n")
path = root / feeder / "dependancy_solver.cpp"
source = path.read_text()
if "void _DependancyLayer::pa2_resolve_incremental()" not in source:
    source += r'''
// PA2: adding pending nodes must never re-queue an already running node.
void _DependancyLayer::pa2_resolve_incremental() {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  this->dependancy_free_nodes.clear();
  for (const auto& item : this->child_map_parent)
    if (item.second.empty() && !this->ongoing_nodes.count(item.first))
      this->dependancy_free_nodes.insert(item.first);
  this->dirty = false;
}
void DependancyResolver::pa2_resolve_incremental() {
  this->data_dependancy.pa2_resolve_incremental();
  this->ctrl_dependancy.pa2_resolve_incremental();
  this->enabled_dependancy.pa2_resolve_incremental();
}
'''
    path.write_text(source)
replace(feeder + "et_feeder.h", "#include <fstream>\n",
        "#include <fstream>\n#include <cstdlib>\n#include <vector>\n")
replace(feeder + "et_feeder.h",
        "    this->build_index_dependancy_cache();\n    this->graph_sanity_check(); // make sure graph is sane\n",
        "    if (std::getenv(\"PA2_STREAMING\")) {\n"
        "      this->dependancy_resolver.pa2_resolve_incremental();\n"
        "      this->pa2_append_trace(file_path);\n"
        "    } else {\n"
        "      this->build_index_dependancy_cache();\n"
        "      this->graph_sanity_check(); // make sure graph is sane\n"
        "    }\n")
replace(feeder + "et_feeder.h", "  // legacy interface\n",
        "  void pa2_append_trace(const std::string& file_path);\n"
        "  void pa2_forget_completed(NodeId node_id) { pa2_nodes.erase(node_id); }\n"
        "  size_t pa2_pending_count() const { return pa2_nodes.size(); }\n\n"
        "  // legacy interface\n")
replace(feeder + "et_feeder.h", "  std::ifstream chakra_file;\n",
        "  std::ifstream chakra_file;\n"
        "  std::unordered_map<NodeId, std::shared_ptr<const ChakraNode>> pa2_nodes;\n"
        "  std::unordered_set<NodeId> pa2_seen_ids;\n")
replace(feeder + "et_feeder.cpp", "    NodeId node_id) {\n  auto key =",
        "    NodeId node_id) {\n"
        "  const auto injected = pa2_nodes.find(node_id);\n"
        "  if (injected != pa2_nodes.end()) return injected->second;\n"
        "  auto key =")
path = root / feeder / "et_feeder.cpp"
source = path.read_text()
if "void ETFeeder::pa2_append_trace" not in source:
    source += r'''
void ETFeeder::pa2_append_trace(const std::string& file_path) {
  std::ifstream input(file_path, std::ios::binary);
  if (!input) throw std::runtime_error("PA2 cannot open trace chunk: " + file_path);
  ChakraGlobalMetadata metadata;
  if (!ProtobufUtils::readMessage<ChakraGlobalMetadata>(input, metadata))
    throw std::runtime_error("PA2 chunk lacks GlobalMetadata: " + file_path);
  std::vector<ChakraNode> nodes;
  std::unordered_set<NodeId> new_ids;
  ChakraNode node;
  while (ProtobufUtils::readMessage<ChakraNode>(input, node)) {
    if (pa2_seen_ids.count(node.id()) || !new_ids.insert(node.id()).second)
      throw std::runtime_error("PA2 duplicate node ID " + std::to_string(node.id()));
    nodes.push_back(node);
  }
  // Reject missing/future dependencies before mutating the live resolver.
  for (const auto& item : nodes) {
    for (const auto parent : item.data_deps())
      if (!new_ids.count(parent) && !pa2_seen_ids.count(parent))
        throw std::runtime_error("PA2 unknown data dependency " + std::to_string(parent));
    for (const auto parent : item.ctrl_deps())
      if (!new_ids.count(parent) && !pa2_seen_ids.count(parent))
        throw std::runtime_error("PA2 unknown control dependency " + std::to_string(parent));
  }
  // Check the new subgraph for cycles; dependencies on old nodes cannot form
  // cycles because old nodes were never allowed to reference future IDs.
  std::unordered_map<NodeId, size_t> indegree;
  std::unordered_map<NodeId, std::vector<NodeId>> children;
  for (const auto& item : nodes) {
    std::unordered_set<NodeId> parents;
    for (const auto p : item.data_deps()) if (new_ids.count(p)) parents.insert(p);
    for (const auto p : item.ctrl_deps()) if (new_ids.count(p)) parents.insert(p);
    indegree[item.id()] = parents.size();
    for (const auto p : parents) children[p].push_back(item.id());
  }
  std::vector<NodeId> ready;
  for (const auto& item : indegree) if (!item.second) ready.push_back(item.first);
  size_t visited = 0;
  while (!ready.empty()) {
    auto id = ready.back(); ready.pop_back(); ++visited;
    for (const auto child : children[id]) if (--indegree[child] == 0) ready.push_back(child);
  }
  if (visited != nodes.size()) throw std::runtime_error("PA2 cyclic trace chunk");
  for (const auto& item : nodes) {
    pa2_seen_ids.insert(item.id());
    pa2_nodes[item.id()] = std::make_shared<ChakraNode>(item);
    dependancy_resolver.add_node(item);
  }
  dependancy_resolver.pa2_resolve_incremental();
}
'''
    path.write_text(source)

# A single in-process event buffer is shared by every rank and the driver.
(root / "astra-sim/workload/Pa2Stream.hh").write_text(r'''#pragma once
#include <cstdlib>
#include <cstdint>
#include <vector>
namespace Pa2Stream {
struct Event { int rank; uint64_t node_id, start_ns, end_ns; bool output; };
inline bool enabled() { static const bool value = std::getenv("PA2_STREAMING") != nullptr; return value; }
inline std::vector<Event>& events() { static std::vector<Event> value; return value; }
inline bool has_output() { for (const auto& e : events()) if (e.output) return true; return false; }
}
''')
replace("astra-sim/workload/Workload.hh", "#include <memory>\n",
        "#include <memory>\n#include <unordered_set>\n#include \"astra-sim/workload/Pa2Stream.hh\"\n")
replace("astra-sim/workload/Workload.hh", "    bool is_finished;\n",
        "    bool is_finished;\n    std::unordered_set<uint64_t> pa2_release_wakeups;\n")
replace("astra-sim/workload/Workload.cc",
        "        if (hw_resource->is_available(node)) {\n",
        "        if (Pa2Stream::enabled() && node->has_attr(\"pa2_release_ns\")) {\n"
        "            const auto release = node->get_attr_msg(\"pa2_release_ns\").uint64_val();\n"
        "            const auto now = Sys::boostedTick();\n"
        "            if (release > now) {\n"
        "                if (pa2_release_wakeups.insert(node_id).second)\n"
        "                    sys->register_event(this, EventType::General, nullptr, release - now);\n"
        "                continue;\n"
        "            }\n"
        "            pa2_release_wakeups.erase(node_id);\n"
        "        }\n"
        "        if (hw_resource->is_available(node)) {\n")
replace("astra-sim/workload/Workload.cc",
        "    if ((dep_resolver.get_dependancy_free_nodes().empty()) &&\n",
        "    if (!Pa2Stream::enabled() && (dep_resolver.get_dependancy_free_nodes().empty()) &&\n")
# Repair a development checkout that reran the older base instrumentation
# after installing streaming. Current instrument_astra.py prevents this.
stats_path = root / "astra-sim/workload/Statistics.cc"
stats_source = stats_path.read_text()
legacy_writer = ('    // PA2: emitted only at actual ASTRA completion callbacks; time is ns.\n'
                 '    static std::ofstream pa2_events("pa2-events.tsv");\n'
                 '    pa2_events << "PA2_EVENT " << workload->sys->id << " " << node_id\n'
                 '              << " " << this->get_operator_statistics(node_id).start_time\n'
                 '              << " " << end_time << std::endl;\n')
if stats_source.count(legacy_writer) == 2 and "Pa2Stream::events()" in stats_source:
    stats_path.write_text(stats_source.replace(legacy_writer, "", 1))
replace("astra-sim/workload/Statistics.cc",
        "    // PA2: emitted only at actual ASTRA completion callbacks; time is ns.\n",
        "    if (Pa2Stream::enabled()) {\n"
        "        Pa2Stream::events().push_back({workload->sys->id, node_id,\n"
        "            this->get_operator_statistics(node_id).start_time, end_time,\n"
        "            node->has_attr(\"pa2_output\") && node->get_attr_msg(\"pa2_output\").bool_val()});\n"
        "        return;\n"
        "    }\n"
        "    // PA2: emitted only at actual ASTRA completion callbacks; time is ns.\n")
replace("astra-sim/workload/Statistics.hh", "    void post_processing();\n",
        "    void pa2_forget_completed(NodeId id) {\n"
        "        const auto start = operator_statistics.at(id).start_time;\n"
        "        auto range = start_times.equal_range(start);\n"
        "        for (auto it = range.first; it != range.second; ++it)\n"
        "            if (it->second == id) { start_times.erase(it); break; }\n"
        "        operator_statistics.erase(id);\n"
        "    }\n"
        "    void post_processing();\n")

queue = "extern/network_backend/analytical/"
replace(queue + "include/astra-network-analytical/common/EventQueue.h",
        "    void proceed() noexcept;\n",
        "    void proceed() noexcept;\n"
        "    [[nodiscard]] EventTime pa2_next_time() const noexcept;\n"
        "    void pa2_advance_idle(EventTime target) noexcept;\n")
path = root / queue / "common/event-queue/EventQueue.cpp"
source = path.read_text()
if "EventTime EventQueue::pa2_next_time()" not in source:
    source += r'''
EventTime EventQueue::pa2_next_time() const noexcept {
    assert(!finished());
    return event_queue.front().get_event_time();
}
void EventQueue::pa2_advance_idle(EventTime target) noexcept {
    assert(target >= current_time);
    assert(finished() || target < event_queue.front().get_event_time());
    current_time = target;
}
'''
    path.write_text(source)

main = "astra-sim/network_frontend/analytical/congestion_unaware/main.cc"
replace(main, "using namespace AstraSim;\n",
        "#include <json/json.hpp>\n#include <iostream>\n#include <stdexcept>\n"
        "#include \"astra-sim/workload/Pa2Stream.hh\"\n\nusing namespace AstraSim;\n")
replace(main,
        "    // run simulation\n    while (!event_queue->finished()) {\n        event_queue->proceed();\n    }\n",
        r'''    // PA2 streaming mode keeps this same event queue, Workload and
    // HardwareResource alive across every causal submission.
    if (Pa2Stream::enabled()) {
        using json = nlohmann::json;
        auto pending = [&]() {
            size_t count = 0;
            for (auto* system : systems) count += system->workload->et_feeder->pa2_pending_count();
            return count;
        };
        auto respond = [&](const std::string& command) {
            json event_records = json::array(), outputs = json::array();
            for (const auto& event : Pa2Stream::events()) {
                json record = {{"rank",event.rank},{"node_id",event.node_id},
                    {"start_ns",event.start_ns},{"end_ns",event.end_ns}};
                event_records.push_back(record);
                if (event.output) outputs.push_back(record);
                // Callback processing at this timestamp is already complete.
                // Free large protobuf/stat objects; retain ID tombstones for
                // completed cross-chunk dependencies and duplicate rejection.
                auto* workload = systems.at(event.rank)->workload;
                workload->stats->pa2_forget_completed(event.node_id);
                workload->et_feeder->pa2_forget_completed(event.node_id);
            }
            Pa2Stream::events().clear();
            json reply = {{"command",command},{"version","pa2-streaming-v1"},
                {"time_ns",event_queue->get_current_time()}, {"events",event_records},
                {"outputs",outputs},{"idle",pending()==0},
                {"pending_nodes",pending()}, {"next_event_ns",nullptr}};
            if (!event_queue->finished()) reply["next_event_ns"] = event_queue->pa2_next_time();
            std::cout << "PA2_STREAM " << reply.dump() << std::endl;
        };
        respond("ready");
        std::string line;
        while (std::getline(std::cin, line)) {
            std::string command;
            try {
                const auto request = json::parse(line);
                command = request.at("command").get<std::string>();
                if (command == "append") {
                    const auto prefix = request.at("trace_prefix").get<std::string>();
                    // Install every rank before issuing any communication.
                    for (auto* system : systems)
                        system->workload->et_feeder->pa2_append_trace(
                            prefix + "." + std::to_string(system->id) + ".et");
                    for (auto* system : systems) system->workload->fire();
                } else if (command == "advance") {
                    const bool bounded = request.contains("until_ns") && !request.at("until_ns").is_null();
                    const uint64_t until = bounded ? request.at("until_ns").get<uint64_t>() : UINT64_MAX;
                    if (until < event_queue->get_current_time())
                        throw std::runtime_error("PA2 cannot advance backward");
                    while (!Pa2Stream::has_output()) {
                        if (event_queue->finished()) {
                            if (pending()) throw std::runtime_error("PA2 native deadlock: pending nodes without events");
                            if (bounded) event_queue->pa2_advance_idle(until);
                            break;
                        }
                        if (event_queue->pa2_next_time() > until) {
                            event_queue->pa2_advance_idle(until);
                            break;
                        }
                        event_queue->proceed(); // drains every callback at this timestamp
                        if (event_queue->get_current_time() == until) break;
                    }
                } else if (command == "finish") {
                    if (pending() || !event_queue->finished())
                        throw std::runtime_error("PA2 finish requires an idle native engine");
                    respond(command);
                    break;
                } else {
                    throw std::runtime_error("PA2 unknown command: " + command);
                }
                respond(command);
            } catch (const std::exception& error) {
                std::cout << "PA2_STREAM " << json({{"command",command},{"error",error.what()},
                    {"time_ns",event_queue->get_current_time()}}).dump() << std::endl;
                return 2;
            }
        }
        if (pending()) {
            std::cerr << "PA2 streaming input ended with unfinished work" << std::endl;
            return 3;
        }
    } else {
        while (!event_queue->finished()) event_queue->proceed();
    }
''')
# ASTRA's asynchronous console logger also writes stdout. Keep a dedicated
# original pipe FD for protocol frames and redirect every normal stdout writer
# to stderr before initializing logging; JSON cannot be interleaved by logs.
replace(main, "#include <stdexcept>\n",
        "#include <stdexcept>\n#include <unistd.h>\n#include <cerrno>\n")
replace(main, "int main(int argc, char* argv[]) {\n",
        r'''int main(int argc, char* argv[]) {
    int pa2_protocol_fd = -1;
    if (Pa2Stream::enabled()) {
        pa2_protocol_fd = dup(STDOUT_FILENO);
        if (pa2_protocol_fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
            throw std::runtime_error("PA2 cannot isolate protocol stdout");
    }
    auto pa2_emit = [&](const std::string& payload) {
        const auto frame = "PA2_STREAM " + payload + "\n";
        size_t offset = 0;
        while (offset < frame.size()) {
            const auto count = write(pa2_protocol_fd, frame.data()+offset, frame.size()-offset);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("PA2 protocol pipe closed");
            offset += static_cast<size_t>(count);
        }
    };
''')
replace(main, '            std::cout << "PA2_STREAM " << reply.dump() << std::endl;\n',
        '            pa2_emit(reply.dump());\n')
replace(main,
        '                std::cout << "PA2_STREAM " << json({{"command",command},{"error",error.what()},\n'
        '                    {"time_ns",event_queue->get_current_time()}}).dump() << std::endl;\n',
        '                pa2_emit(json({{"command",command},{"error",error.what()},\n'
        '                    {"time_ns",event_queue->get_current_time()}}).dump());\n')
print("Applied pa2-streaming-v1 (persistent native event queue/resources)")
