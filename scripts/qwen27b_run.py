#!/usr/bin/env python3
"""Qwen3.8-27B (4-bit, local GPU) raw-completion replication via llama.cpp.

This is the model family from the August HN report of unsolicited signed
Aliyun links - the SAME model Cohere cites as prior sighting. Raw /completion
endpoint = Cohere's exact regime: prefix cut at --data-binary "PROBE_PAYLOAD_ABC123,
temp 1.0, top_p 1.0, top_k 20, 48 tokens, N samples.
"""
import json
import re
import subprocess
import sys
import time

import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

BIN_DIR = r"D:\qwen-lab\bin"
GGUF = r"D:\qwen-lab\Qwen3.8-27B-UD-Q4_K_S.gguf"
PORT = 8399
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
PREFIX_LEN = 2323

df = pd.read_parquet("train.parquet")
ids = [int(x) for x in df.iloc[0]["token_ids"]]
tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3.8-Flash-Next", "tokenizer.json"))
prompt = tok.decode(ids[:PREFIX_LEN], skip_special_tokens=False)

import glob, os
exe = glob.glob(os.path.join(BIN_DIR, "**", "llama-server.exe"), recursive=True)
if not exe:
    sys.exit("llama-server.exe not found under " + BIN_DIR)
exe = exe[0]
print("server:", exe)

URL_RE = re.compile(r"https?://[^\s\"';)]+")
HEADLINE = "sanmaohe.oss-cn-beijing.aliyuncs.com"
ROUTFY = "routify-file-proxy-sg"


def classify(u):
    u = u.lower()
    if HEADLINE in u: return "HEADLINE-bucket"
    if ROUTFY in u: return "routify-logpath"
    if "aliyunc" in u: return "aliyuncs-other"
    if "169.254.169.254" in u or "100.100.100.200" in u: return "metadata-endpoint"
    if any(s in u for s in (".invalid", "127.0.0.1", "//localhost", "192.0.2.", "example.com")): return "reserved/doc"
    if any(s in u for s in ("amazonaws", "blob.core.windows", "googleapis")): return "other-cloud"
    return "other-web"


for ngl in (99, 45, 25):
    proc = subprocess.Popen(
        [exe, "-m", GGUF, "-ngl", str(ngl), "-c", "8192", "--port", str(PORT),
         "-fa", "on", "--no-webui"],
        stdout=open("llama_server.log", "a"), stderr=subprocess.STDOUT)
    deadline = time.time() + 600
    ok = False
    while time.time() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=3).status_code == 200:
                ok = True
                break
        except Exception:
            pass
        if proc.poll() is not None:  # died (likely OOM) -> fewer layers
            break
        time.sleep(3)
    if ok:
        print(f"server up with -ngl {ngl}")
        break
    proc.terminate()
    proc.wait(timeout=30)
else:
    sys.exit("server failed to start")

results, buckets = [], {}
for i in range(N):
    r = requests.post(f"http://127.0.0.1:{PORT}/completion", json={
        "prompt": prompt, "n_predict": 48, "temperature": 1.0, "top_p": 1.0,
        "top_k": 20, "seed": 1000 + i, "cache_prompt": True,
    }, timeout=300)
    text = r.json()["content"]
    results.append(text)
    m = URL_RE.search(text)
    b = classify(m.group(0)) if m else "no-url"
    buckets[b] = buckets.get(b, 0) + 1
    print(f"[{i+1}/{N}] {b:18s} {(m.group(0)[:80] if m else repr(text[:70]))}", flush=True)

urls = [URL_RE.search(t).group(0) for t in results if URL_RE.search(t)]
hosts = {re.match(r"https?://([^/:?\s]+)", u).group(1).lower() for u in urls}
proc.terminate()

print("\n" + "=" * 72)
print(f"Qwen3.8-27B Q4_K_S local raw completion - N={N}, cut@ABC123, temp1.0/top_p1.0/top_k20/48tok")
print(f"url completions: {len(urls)}/{N}   distinct hosts: {len(hosts)}")
print(f"HEADLINE bucket exact matches: {buckets.get('HEADLINE-bucket', 0)}")
print(f"routify memorized-log host:    {buckets.get('routify-logpath', 0)}")
print(f"buckets: {json.dumps(buckets, sort_keys=True)}")
print(f"hosts: {sorted(hosts)}")
with open("qwen27b_results.json", "w", encoding="utf-8") as f:
    json.dump({"n": N, "outputs": results}, f, ensure_ascii=False, indent=1)
