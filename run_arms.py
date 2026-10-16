"""Run extra comparison arms (Jev, LLM) on the Laya return-intent test set and merge into results.json.

Reuses metrics() from the Laya repo's 03_evaluate.py so every arm is scored identically.
Does not modify the Laya repo: Laya's round-2 results are copied in, new arms are added.

Setup:
  LAYA_REPO            path to the laya-return-intent-finetuning repo (default: ../laya-return-intent-finetuning)
  JEV_API_KEY          in the environment, or in a gitignored .env file next to this script
  llm_cmd.local.json   gitignored; the LLM arm's command as a JSON list. It must read the message on stdin and
                       print one label; {SYSTEM} is replaced by the classification instructions, {MODEL} by --llm-model.
                       Example: ["my-llm-cli", "--system", "{SYSTEM}"]

Usage:
  python run_arms.py --probe-jev            # one Jev call, prints the raw response
  python run_arms.py --arms jev,llm         # full run, 149 messages per arm
  python run_arms.py --arms llm --limit 5   # smoke test
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LAYA_REPO") or os.path.normpath(os.path.join(HERE, "..", "laya-return-intent-finetuning"))
OUT = os.path.join(HERE, "data", "results.json")
RAW = os.path.join(HERE, "data", "predictions_extra.json")

_env = next((p for p in (os.path.join(HERE, ".env"), os.path.join(HERE, ".env.txt")) if os.path.exists(p)), None)
if _env:  # optional: a line JEV_API_KEY=... created by the user, never committed
    for _l in open(_env, encoding="utf-8-sig"):
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.strip().split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

spec = importlib.util.spec_from_file_location("evalmod", os.path.join(REPO, "scripts", "03_evaluate.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
LABELS = ev.LABELS


def jev_call(url, key, model, text, questions):
    body = json.dumps({"model": model, "state": text, "questions": questions}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def run_jev(rows, questions, url, model):
    key = os.environ.get("JEV_API_KEY")
    if not key:
        sys.exit("JEV_API_KEY is not set in this environment")
    jev_call(url, key, model, rows[0]["text"], questions)  # warmup
    preds, confs, probs, lat, tokens, versions = [], [], [], [], 0, set()
    for r in rows:
        t0 = time.perf_counter()
        resp = jev_call(url, key, model, r["text"], questions)
        lat.append((time.perf_counter() - t0) * 1000)
        a = resp["answers"]["intent"]
        preds.append(a["choice"])
        p = a.get("probabilities", {})
        probs.append([float(p.get(k, 0.0)) for k in LABELS])
        confs.append(float(a.get("confidence", max(p.values()) if p else 0.0)))
        tokens += resp.get("usage", {}).get("input_tokens", 0)
        versions.add(resp.get("model"))
    return (preds, confs, probs, lat), {"input_tokens_total": tokens, "model_versions": sorted(v for v in versions if v)}


def run_llm(rows, questions, model):
    cfg = os.path.join(HERE, "llm_cmd.local.json")
    if not os.path.exists(cfg):
        sys.exit("llm_cmd.local.json not found: see the setup notes at the top of this file")
    template = json.load(open(cfg, encoding="utf-8"))
    crit = questions["intent"]["criteria"]
    system = ("Classify a customer's return-request message. Reply with exactly one label and nothing else.\n"
              + "\n".join(f"- {k}: {v}" for k, v in crit.items()))
    cmd = [a.replace("{SYSTEM}", system).replace("{MODEL}", model) for a in template]
    cwd = tempfile.mkdtemp(prefix="llm_arm_")  # empty dir so no project context is loaded
    preds, lat, raw = [], [], []
    for r in rows:
        t0 = time.perf_counter()
        out = subprocess.run(cmd, input=r["text"], capture_output=True, text=True, cwd=cwd, timeout=180, encoding="utf-8")
        lat.append((time.perf_counter() - t0) * 1000)
        txt = out.stdout.strip().lower()
        raw.append(txt[:80])
        preds.append(next((l for l in LABELS if l in txt), "clarify"))
    unparsed = sum(1 for t in raw if not any(l in t for l in LABELS))
    return (preds, None, None, lat), {"unparsed_replies": unparsed, "sample_replies": raw[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--probe-jev", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jev-url", default=os.environ.get("JEV_URL", "https://api.typesafe.ai/v1/systemone"))
    ap.add_argument("--jev-model", default=os.environ.get("JEV_MODEL", "jev-latest"))
    ap.add_argument("--llm-model", default="small", help="substituted for {MODEL} in the LLM command")
    args = ap.parse_args()

    os.chdir(REPO)  # test set paths in the repo are relative
    rows = ev.load_test()
    questions = json.load(open("data/questions.json", encoding="utf-8"))
    if args.limit:
        rows = rows[:args.limit]

    if args.probe_jev:
        key = os.environ.get("JEV_API_KEY") or sys.exit("JEV_API_KEY is not set in this environment")
        print(json.dumps(jev_call(args.jev_url, key, args.jev_model, rows[0]["text"], questions), indent=2))
        return

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    base = json.load(open(os.path.join(REPO, "results", "round2", "results.json")))
    results = json.load(open(OUT)) if os.path.exists(OUT) else base
    extra = json.load(open(RAW)) if os.path.exists(RAW) else {}
    results.setdefault("meta", {})
    for arm in [a for a in args.arms.split(",") if a]:
        print("running", arm, flush=True)
        if arm == "jev":
            res, info = run_jev(rows, questions, args.jev_url, args.jev_model)
        elif arm == "llm":
            res, info = run_llm(rows, questions, args.llm_model)
        else:
            continue
        results["arms"][arm] = ev.metrics(rows, *res)
        results["meta"][arm] = info
        extra[arm] = {"preds": res[0], "confs": res[1], "probs": res[2]}
        a = results["arms"][arm]
        print(f"  n={a['n']} acc={a['accuracy']:.3f} f1={a['macro_f1']:.3f} inj={a['injection_catch_rate']} "
              f"unsafe={a['unsafe_error_rate']:.3f} p50={a['latency_ms']['p50']:.0f}ms {info}", flush=True)
        if not args.limit:  # smoke tests never overwrite real results
            json.dump(results, open(OUT, "w"), indent=2)
            json.dump(extra, open(RAW, "w"))


if __name__ == "__main__":
    main()
