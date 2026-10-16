"""Wrap results.json as data/results.js so index.html works when opened from disk (file:// blocks fetch)."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "data", "results.json")
if not os.path.exists(src):
    src = os.path.join(os.environ.get("LAYA_REPO") or os.path.join(here, "..", "laya-return-intent-finetuning"),
                       "results", "round2", "results.json")
d = json.load(open(src))
os.makedirs(os.path.join(here, "data"), exist_ok=True)
open(os.path.join(here, "data", "results.js"), "w").write("window.RESULTS = " + json.dumps(d) + ";\n")
print("arms:", list(d["arms"]), "source:", src)
