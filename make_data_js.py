"""Wrap results as data/results.js so index.html works when opened without a server.

Uses data/results.local.json (your own runs) if it exists, otherwise the shipped data/results.json.
"""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, "data")
src = os.path.join(data, "results.local.json")
if not os.path.exists(src):
    src = os.path.join(data, "results.json")
d = json.load(open(src))
open(os.path.join(data, "results.js"), "w").write("window.RESULTS = " + json.dumps(d) + ";\n")
print("arms:", list(d["arms"]), "| source:", os.path.basename(src))
