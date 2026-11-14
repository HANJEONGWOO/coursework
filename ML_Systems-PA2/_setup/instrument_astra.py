"""Install the small, versioned PA2 timing instrumentation in pinned ASTRA."""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
assert (root / "astra-sim").is_dir(), root

def replace_once(path, old, new):
    text = path.read_text()
    if new in text:
        return
    # The persistent-engine patch inserts its protocol branch between the
    # assignment and this legacy event writer. Do not duplicate the writer
    # when students rerun setup on an already instrumented checkout.
    if (old == "    this->get_operator_statistics(node_id).end_time = end_time;\n"
            and 'static std::ofstream pa2_events("pa2-events.tsv");' in text):
        return
    if text.count(old) != 1:
        raise RuntimeError(f"Pinned ASTRA source changed: {path}")
    path.write_text(text.replace(old, new))

# Preserve the educational compute profile's nanosecond precision. Chakra's
# standard duration_micros field remains populated for other trace viewers.
replace_once(root / "astra-sim/workload/Workload.cc",
    "    uint64_t runtime = 1ul;\n    if (node->runtime() != 0ul) {\n        // chakra runtimes are in microseconds and we should convert it into\n        // nanoseconds\n        runtime = node->runtime() * 1000;\n    }\n",
    "    uint64_t runtime = 1ul;\n    if (node->runtime() != 0ul) {\n        // chakra runtimes are in microseconds and we should convert it into\n        // nanoseconds\n        runtime = node->runtime() * 1000;\n    }\n"
    "    // PA2: preserve exact ns while retaining standard Chakra microseconds.\n"
    "    if (node->has_attr(\"pa2_duration_ns\")) {\n"
    "        runtime = node->get_attr_msg(\"pa2_duration_ns\").uint64_val();\n"
    "    }\n")

stats = root / "astra-sim/workload/Statistics.cc"
replace_once(stats, '#include <vector>\n', '#include <vector>\n#include <fstream>  // PA2 event records\n')
old_instrumentation = (
    "    // PA2: emitted only at actual ASTRA completion callbacks; time is ns.\n"
    "    std::cout << \"PA2_EVENT \" << workload->sys->id << \" \" << node_id\n"
    "              << \" \" << this->get_operator_statistics(node_id).start_time\n"
    "              << \" \" << end_time << std::endl;\n")
if old_instrumentation in stats.read_text():
    stats.write_text(stats.read_text().replace(old_instrumentation, ""))
replace_once(stats,
    "    this->get_operator_statistics(node_id).end_time = end_time;\n",
    "    this->get_operator_statistics(node_id).end_time = end_time;\n"
    "    // PA2: emitted only at actual ASTRA completion callbacks; time is ns.\n"
    "    static std::ofstream pa2_events(\"pa2-events.tsv\");\n"
    "    pa2_events << \"PA2_EVENT \" << workload->sys->id << \" \" << node_id\n"
    "              << \" \" << this->get_operator_statistics(node_id).start_time\n"
    "              << \" \" << end_time << std::endl;\n")
replace_once(
    root / "extern/network_backend/analytical/congestion_unaware/multi-dim-topology/MultiDimTopology.cpp",
    "    for (auto dim = 0; dim < dims_count; dim++) {\n        // check the dim that has different address\n",
    "    // PA2 hierarchical links: a cross-node transfer uses the slow\n"
    "    // dimension even when its local GPU coordinate also differs.\n"
    "    for (auto dim = dims_count - 1; dim >= 0; dim--) {\n"
    "        // check the highest dimension that has a different address\n")
print("Applied pa2-node-events-v1 and pa2-hierarchical-link-v1")
