"""Run extra comparison arms (Jev, LLM) on the return-intent test set and score them like the shipped arms.

Setup (see README):
  JEV_API_KEY      in the environment or in a gitignored .env file next to this script
  LLM arm, pick one:
    - any OpenAI-compatible chat API: set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (e.g. https://api.openai.com/v1)
    - any command line tool: llm_cmd.local.json (gitignored), a JSON list. It must read the message on stdin and
      print one label; {SYSTEM} is replaced by the instructions, {MODEL} by --llm-model.

Your runs are written to data/results.local.json (gitignored); the shipped data/results.json is never overwritten.

Usage:
  python run_arms.py --probe-jev            # one Jev call, prints the raw response
  python run_arms.py --arms jev,llm         # full run, 149 messages per arm
  python run_arms.py --arms llm --limit 5   # smoke test, saves nothing
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from scoring import LABELS, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SHIPPED = os.path.join(DATA, "results.json")
OUT = os.path.join(DATA, "results.local.json")
RAW = os.path.join(DATA, "predictions_extra.local.json")

_env = next((p for p in (os.path.join(HERE, ".env"), os.path.join(HERE, ".env.txt")) if os.path.exists(p)), None)
if _env:  # optional: lines like JEV_API_KEY=... created by you, never committed
    for _l in open(_env, encoding="utf-8-sig"):
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.strip().split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))


def post_json(url, headers, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def jev_call(url, key, model, text, questions):
    return post_json(url, {"Authorization": f"Bearer {key}"}, {"model": model, "state": text, "questions": questions})


def run_jev(rows, questions, url, model):
    key = os.environ.get("JEV_API_KEY")
    if not key:
        sys.exit("JEV_API_KEY is not set (environment or .env file)")
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


def llm_system(questions):
    crit = questions["intent"]["criteria"]
    return ("Classify a customer's return-request message. Reply with exactly one label and nothing else.\n"
            + "\n".join(f"- {k}: {v}" for k, v in crit.items()))


def run_llm(rows, questions, model):
    system = llm_system(questions)
    base_url = os.environ.get("LLM_BASE_URL")
    cfg = os.path.join(HERE, "llm_cmd.local.json")
    if base_url:
        key, name = os.environ.get("LLM_API_KEY", ""), os.environ.get("LLM_MODEL") or sys.exit("set LLM_MODEL")

        def ask(text):
            resp = post_json(base_url.rstrip("/") + "/chat/completions", {"Authorization": f"Bearer {key}"},
                             {"model": name, "temperature": 0, "max_tokens": 8,
                              "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]})
            return resp["choices"][0]["message"]["content"]
        mode = "http"
    elif os.path.exists(cfg):
        template = json.load(open(cfg, encoding="utf-8"))
        cmd = [a.replace("{SYSTEM}", system).replace("{MODEL}", model) for a in template]
        cwd = tempfile.mkdtemp(prefix="llm_arm_")  # empty dir so no project context is loaded

        def ask(text):
            return subprocess.run(cmd, input=text, capture_output=True, text=True, cwd=cwd,
                                  timeout=180, encoding="utf-8").stdout
        mode = "command"
    else:
        sys.exit("No LLM configured: set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL, or create llm_cmd.local.json (see README)")
    ask(rows[0]["text"])  # warmup
    preds, lat, raw = [], [], []
    for r in rows:
        t0 = time.perf_counter()
        txt = ask(r["text"]).strip().lower()
        lat.append((time.perf_counter() - t0) * 1000)
        raw.append(txt[:80])
        preds.append(next((l for l in LABELS if l in txt), "clarify"))
    unparsed = sum(1 for t in raw if not any(l in t for l in LABELS))
    return (preds, None, None, lat), {"mode": mode, "unparsed_replies": unparsed, "sample_replies": raw[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--probe-jev", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jev-url", default=os.environ.get("JEV_URL", "https://api.typesafe.ai/v1/systemone"))
    ap.add_argument("--jev-model", default=os.environ.get("JEV_MODEL", "jev-latest"))
    ap.add_argument("--llm-model", default="small", help="substituted for {MODEL} in a command-based LLM")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"), encoding="utf-8") if l.strip()]
    questions = json.load(open(os.path.join(DATA, "questions.json"), encoding="utf-8"))
    if args.limit:
        rows = rows[:args.limit]

    if args.probe_jev:
        key = os.environ.get("JEV_API_KEY") or sys.exit("JEV_API_KEY is not set (environment or .env file)")
        print(json.dumps(jev_call(args.jev_url, key, args.jev_model, rows[0]["text"], questions), indent=2))
        return

    results = json.load(open(OUT if os.path.exists(OUT) else SHIPPED))
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
        results["arms"][arm] = metrics(rows, *res)
        results["meta"][arm] = info
        extra[arm] = {"preds": res[0], "confs": res[1], "probs": res[2]}
        a = results["arms"][arm]
        print(f"  n={a['n']} acc={a['accuracy']:.3f} f1={a['macro_f1']:.3f} inj={a['injection_catch_rate']} "
              f"unsafe={a['unsafe_error_rate']:.3f} p50={a['latency_ms']['p50']:.0f}ms {info}", flush=True)
        if not args.limit:  # smoke tests never save
            json.dump(results, open(OUT, "w"), indent=2)
            json.dump(extra, open(RAW, "w"))
    if args.arms and not args.limit:
        print("saved to data/results.local.json; run: python make_data_js.py")


if __name__ == "__main__":
    main()
