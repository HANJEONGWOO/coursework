#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${PA2_ASTRA_RUNTIME:-"$SCRIPT_DIR/../runtime"}
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)
JOBS=${PA2_BUILD_JOBS:-4}
trap 'printf "{\"state\":\"failed\",\"line\":%s}\n" "$LINENO" > "$ROOT/build-status.json"' ERR
printf '{"state":"building"}\n' > "$ROOT/build-status.json"
for tool in git g++ cmake python3; do command -v "$tool" >/dev/null || { echo "Missing $tool; ask your Linux administrator to install it." >&2; exit 1; }; done
LOCK="$SCRIPT_DIR/../runtime.lock.json"
if [[ ! -f "$LOCK" ]]; then LOCK="$SCRIPT_DIR/runtime.lock.json"; fi
ASTRA_COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["astra_commit"])' "$LOCK")
mkdir -p "$ROOT/src" "$ROOT/prefix"
if [[ ! -d "$ROOT/src/astra/.git" ]]; then git clone --no-checkout https://github.com/astra-sim/astra-sim.git "$ROOT/src/astra"; fi
git -C "$ROOT/src/astra" checkout --detach "$ASTRA_COMMIT"
git -C "$ROOT/src/astra" submodule update --init --recursive extern/graph_frontend/chakra extern/network_backend/analytical extern/remote_memory_backend/analytical extern/helper/fmt extern/helper/spdlog
if [[ ! -d "$ROOT/src/protobuf/.git" ]]; then git clone --depth 1 --branch v21.12 https://github.com/protocolbuffers/protobuf.git "$ROOT/src/protobuf"; fi
PROTOBUF_COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["protobuf_commit"])' "$LOCK")
if [[ "$(git -C "$ROOT/src/protobuf" rev-parse HEAD)" != "$PROTOBUF_COMMIT" ]]; then
  echo 'Protobuf tag does not match the pinned commit; refusing to build.' >&2
  exit 1
fi
if [[ ! -f "$ROOT/prefix/bin/protoc" ]]; then
  cmake -S "$ROOT/src/protobuf/cmake" -B "$ROOT/protobuf-build" -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="$ROOT/prefix"
  cmake --build "$ROOT/protobuf-build" -j "$JOBS"
  cmake --install "$ROOT/protobuf-build"
fi
export PATH="$ROOT/prefix/bin:$PATH"
export CMAKE_PREFIX_PATH="$ROOT/prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
python3 "$SCRIPT_DIR/instrument_astra.py" "$ROOT/src/astra"
python3 "$SCRIPT_DIR/instrument_astra_stream.py" "$ROOT/src/astra"
python3 "$SCRIPT_DIR/instrument_astra_stream_fast.py" "$ROOT/src/astra"
python3 "$SCRIPT_DIR/instrument_astra_stream_compact.py" "$ROOT/src/astra"
CHAKRA="$ROOT/src/astra/extern/graph_frontend/chakra/schema/protobuf"
if [[ ! -f "$CHAKRA/et_def.pb.cc" ]]; then
  protoc --proto_path="$CHAKRA" --cpp_out="$CHAKRA" --python_out="$CHAKRA" "$CHAKRA/et_def.proto"
fi
cmake -S "$ROOT/src/astra/build/astra_analytical" -B "$ROOT/astra-build" -DBUILDTARGET=congestion_unaware -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$ROOT/astra-build" -j "$JOBS"
python3 -m venv --without-pip "$ROOT/venv"
python3 - "$ROOT" "$LOCK" "$SCRIPT_DIR" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1]); lock=json.load(open(sys.argv[2])); scripts=pathlib.Path(sys.argv[3])
def revision(path): return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
lock['astra_commit']=revision(root/'src/astra')
lock['chakra_commit']=revision(root/'src/astra/extern/graph_frontend/chakra')
lock['backend_commit']=revision(root/'src/astra/extern/network_backend/analytical')
lock['protobuf_commit']=revision(root/'src/protobuf')
lock['binary']='astra-build/bin/AstraSim_Analytical_Congestion_Unaware'
lock['streaming_version']='pa2-streaming-v1'
lock['streaming_patch_sha256']=hashlib.sha256((scripts/'instrument_astra_stream.py').read_bytes()).hexdigest()
lock['streaming_templates_version']='pa2-templates-v1'
lock['template_patch_sha256']=hashlib.sha256((scripts/'instrument_astra_stream_fast.py').read_bytes()).hexdigest()
lock['streaming_compact_version']='pa2-compact-v1'
lock['compact_patch_sha256']=hashlib.sha256((scripts/'instrument_astra_stream_compact.py').read_bytes()).hexdigest()
lock['binary_sha256']=hashlib.sha256((root/lock['binary']).read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps(lock,indent=2)+'\n')
(root/'build-status.json').write_text(json.dumps({'state':'built','binary':lock['binary']})+'\n')
print(json.dumps(lock,indent=2))
PY
