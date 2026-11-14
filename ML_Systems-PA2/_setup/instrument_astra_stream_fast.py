"""Install reusable native trace templates; preserve exact ASTRA scheduling."""
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


feeder = "extern/graph_frontend/chakra/src/feeder_v3/"
(root / feeder / "pa2_id_ranges.h").write_text(r'''#pragma once
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>

// Retired IDs are almost contiguous; retain exact membership without one
// heap allocation per completed operator. Active gaps remain represented.
class Pa2IdRanges {
  std::map<uint64_t,uint64_t> ranges;
 public:
  size_t count(uint64_t value) const {
    auto next = ranges.upper_bound(value);
    return next != ranges.begin() && std::prev(next)->second >= value ? 1 : 0;
  }
  void insert(uint64_t value) {
    auto next = ranges.upper_bound(value);
    uint64_t first = value, last = value;
    if (next != ranges.begin()) {
      auto prior = std::prev(next);
      if (prior->second >= value) return;
      if (prior->second != UINT64_MAX && prior->second + 1 == value) {
        first = prior->first;
        ranges.erase(prior);
      }
    }
    if (next != ranges.end() && value != UINT64_MAX && next->first == value+1) {
      last = next->second;
      ranges.erase(next);
    }
    ranges[first] = last;
  }
};
''')
replace(feeder + "dependancy_solver.h", '#include "common.h"\n',
        '#include "common.h"\n#include "pa2_id_ranges.h"\n')
replace(feeder + "dependancy_solver.h", "  std::unordered_set<NodeId> pa2_finished_nodes;\n",
        "  Pa2IdRanges pa2_finished_nodes;\n")
replace(feeder + "et_feeder.h", "  std::unordered_set<NodeId> pa2_seen_ids;\n",
        "  Pa2IdRanges pa2_seen_ids;\n")
replace(feeder + "et_feeder.h", "  void pa2_append_trace(const std::string& file_path);\n",
        "  void pa2_append_trace(const std::string& file_path);\n"
        "  static std::vector<ChakraNode> pa2_read_trace(const std::string& file_path);\n"
        "  void pa2_append_nodes(const std::vector<ChakraNode>& nodes);\n")
replace(feeder + "et_feeder.cpp",
        r'''void ETFeeder::pa2_append_trace(const std::string& file_path) {
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
''',
        r'''void ETFeeder::pa2_append_trace(const std::string& file_path) {
  pa2_append_nodes(pa2_read_trace(file_path));
}
std::vector<ChakraNode> ETFeeder::pa2_read_trace(const std::string& file_path) {
  std::ifstream input(file_path, std::ios::binary);
  if (!input) throw std::runtime_error("PA2 cannot open trace chunk: " + file_path);
  ChakraGlobalMetadata metadata;
  if (!ProtobufUtils::readMessage<ChakraGlobalMetadata>(input, metadata))
    throw std::runtime_error("PA2 chunk lacks GlobalMetadata: " + file_path);
  std::vector<ChakraNode> nodes;
  ChakraNode node;
  while (ProtobufUtils::readMessage<ChakraNode>(input, node)) nodes.push_back(node);
  return nodes;
}
void ETFeeder::pa2_append_nodes(const std::vector<ChakraNode>& nodes) {
  std::unordered_set<NodeId> new_ids;
  for (const auto& node : nodes)
    if (pa2_seen_ids.count(node.id()) || !new_ids.insert(node.id()).second)
      throw std::runtime_error("PA2 duplicate node ID " + std::to_string(node.id()));
''')
replace("astra-sim/workload/Workload.cc",
        "        collective_comm_wrapper_map.erase(coll_comm_id);\n",
        "        collective_comm_wrapper_map.erase(coll_comm_id);\n"
        "        if (Pa2Stream::enabled()) collective_comm_node_id_map.erase(coll_comm_id);\n")

main = "astra-sim/network_frontend/analytical/congestion_unaware/main.cc"
replace(main, "        using json = nlohmann::json;\n",
        "        using json = nlohmann::json;\n"
        "        using Pa2Nodes = std::vector<Chakra::FeederV3::ChakraNode>;\n"
        "        std::unordered_map<std::string, std::vector<Pa2Nodes>> pa2_templates;\n")
replace(main, '{"command",command},{"version","pa2-streaming-v1"},\n',
        '{"command",command},{"version","pa2-streaming-v1"},\n'
        '                {"streaming_templates_version","pa2-templates-v1"},\n')
replace(main, '                if (command == "append") {\n',
        r'''                if (command == "define") {
                    const auto key = request.at("template_id").get<std::string>();
                    if (key.empty() || pa2_templates.count(key))
                        throw std::runtime_error("PA2 template ID empty or already defined");
                    const auto prefix = request.at("trace_prefix").get<std::string>();
                    std::vector<Pa2Nodes> by_rank(systems.size());
                    for (auto* system : systems) {
                        auto nodes = Chakra::ETFeeder::pa2_read_trace(prefix+"."+std::to_string(system->id)+".et");
                        std::unordered_set<uint64_t> ids;
                        for (const auto& node : nodes)
                            if (node.id() < 1 || node.id() > nodes.size() || !ids.insert(node.id()).second)
                                throw std::runtime_error("PA2 template IDs must be exactly 1..N on each rank");
                        for (const auto& node : nodes) {
                            for (const auto dep : node.data_deps())
                                if (!ids.count(dep)) throw std::runtime_error("PA2 template has external data dependency");
                            for (const auto dep : node.ctrl_deps())
                                if (!ids.count(dep)) throw std::runtime_error("PA2 template has external control dependency");
                        }
                        by_rank.at(system->id) = std::move(nodes);
                    }
                    pa2_templates.emplace(key, std::move(by_rank));
                } else if (command == "instantiate") {
                    const auto key = request.at("template_id").get<std::string>();
                    const auto found = pa2_templates.find(key);
                    if (found == pa2_templates.end()) throw std::runtime_error("PA2 unknown template");
                    const auto release = request.at("release_ns").get<uint64_t>();
                    if (release < event_queue->get_current_time()) throw std::runtime_error("PA2 template release is in the past");
                    const auto tag_offset = request.value("tag_offset", uint64_t(0));
                    const auto extra = request.value("extra_deps", json::object());
                    std::vector<Pa2Nodes> copies = found->second;
                    for (auto* system : systems) {
                        auto& nodes = copies.at(system->id);
                        if (nodes.empty()) continue;
                        const auto rank = std::to_string(system->id);
                        const auto offset = request.at("id_offsets").at(rank).get<uint64_t>();
                        if (offset > UINT64_MAX - nodes.size()) throw std::runtime_error("PA2 node ID overflow");
                        const auto extra_rank = extra.value(rank,json::object());
                        for (const auto& pair : extra_rank.items()) {
                            const auto target = std::stoull(pair.key());
                            if (target < 1 || target > nodes.size()) throw std::runtime_error("PA2 extra dependency target outside template");
                        }
                        for (auto& node : nodes) {
                            const auto local_id = node.id();
                            node.set_id(local_id + offset);
                            for (auto& dep : *node.mutable_data_deps()) dep += offset;
                            for (auto& dep : *node.mutable_ctrl_deps()) dep += offset;
                            const auto local = std::to_string(local_id);
                            if (extra_rank.contains(local))
                                for (const auto& dep : extra_rank.at(local)) node.add_data_deps(dep.get<uint64_t>());
                            bool has_release = false;
                            for (auto& attr : *node.mutable_attr()) {
                                if (attr.name() == "pa2_release_ns") { attr.set_uint64_val(release); has_release = true; }
                                if (attr.name() == "comm_tag") {
                                    const auto tag = attr.int64_val();
                                    if (tag < 0 || tag_offset > UINT32_MAX || uint64_t(tag) > UINT32_MAX-tag_offset)
                                        throw std::runtime_error("PA2 message tag overflow");
                                    attr.set_int64_val(uint64_t(tag)+tag_offset);
                                }
                            }
                            if (!has_release) {
                                auto* attr = node.add_attr(); attr->set_name("pa2_release_ns"); attr->set_uint64_val(release);
                            }
                        }
                    }
                    for (auto* system : systems) system->workload->et_feeder->pa2_append_nodes(copies.at(system->id));
                    for (auto* system : systems) system->workload->fire();
                } else if (command == "append") {
''')
print("Applied pa2-templates-v1 and compact exact ID tombstones")
