#!/usr/bin/env bash
# Pull GGUF weights from Docker Hub's "ai/" model namespace (OCI artifacts; Docker itself is not needed).
# Equivalent to `docker model pull ai/<name>:<tag>`. Hugging Face GGUFs work just as well.
set -euo pipefail
mkdir -p models

pull() {
  local repo="$1" tag="$2" out="models/$3"
  if [ -s "$out" ]; then echo "have $out"; return; fi
  local tok acc m digest ref
  tok=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ai/$repo:pull" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
  acc="application/vnd.oci.image.manifest.v1+json,application/vnd.oci.image.index.v1+json"
  m=$(curl -s -H "Authorization: Bearer $tok" -H "Accept: $acc" "https://registry-1.docker.io/v2/ai/$repo/manifests/$tag")
  if echo "$m" | grep -q '"manifests"'; then
    ref=$(echo "$m" | python3 -c 'import sys,json;print(json.load(sys.stdin)["manifests"][0]["digest"])')
    m=$(curl -s -H "Authorization: Bearer $tok" -H "Accept: $acc" "https://registry-1.docker.io/v2/ai/$repo/manifests/$ref")
  fi
  digest=$(echo "$m" | python3 -c 'import sys,json;print([l["digest"] for l in json.load(sys.stdin)["layers"] if "gguf" in l["mediaType"]][0])')
  echo "pulling ai/$repo:$tag -> $out"
  curl -L --retry 5 -H "Authorization: Bearer $tok" "https://registry-1.docker.io/v2/ai/$repo/blobs/$digest" -o "$out"
}

pull qwen2.5  3B-Q4_K_M qwen2.5-3b-q4km.gguf
pull qwen2.5  7B-Q4_K_M qwen2.5-7b-q4km.gguf
pull gemma3   4B-Q4_K_M gemma3-4b-q4km.gguf
