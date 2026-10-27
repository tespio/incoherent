#!/usr/bin/env python3
"""T0: verify Cohere's released token IDs decode byte-for-byte to the published trace.
Also prints the load-bearing repro cut point. Requires: pip install pandas tokenizers"""
import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

df = pd.read_parquet("data/train.parquet")
ids = [int(x) for x in df.iloc[0]["token_ids"]]
trace = df.iloc[0]["trace"]
print(f"released artifact: 1 row, {len(ids)} token ids, trace {len(trace)} chars")

tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3.8-Flash-Next", "tokenizer.json"))
decoded = tok.decode(ids, skip_special_tokens=False)
print("decode == published trace:", decoded == trace)
print("lengths:", len(decoded), "==", len(trace))

prefix = tok.decode(ids[:2323], skip_special_tokens=False)
print("repro prefix [0:2323] ends with:", repr(prefix[-90:]))
print("sanmaohe occurrences:", trace.count("sanmaohe"),
      "| PROBE_PAYLOAD occurrences:", trace.count("PROBE_PAYLOAD"))
assert decoded == trace, "MISMATCH - artifact inconsistent"
print("VERDICT: artifact internally consistent")
