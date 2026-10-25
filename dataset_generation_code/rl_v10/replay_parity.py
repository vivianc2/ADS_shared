#!/usr/bin/env python3
"""Replay parity for the RL env (priority 0.3, 2026-10-23).

Stored benchmark episodes (rpg_benchmark/out/gate_v10.2/*, out/opus55_v10.2_rest486) hold every reply the model
gave (turn["raw"], the text run_world received). Replaying those replies open-loop:

  A. run_world(frozen bench, ReplayLLM, deterministic resolver, history H)       -- the benchmark itself
  B. BenchStepEnv(history H) driven by step(reply)                                -- the RL env
  check 1: A == B exactly: every prompt run_world sent, every turn record (minus timing), the grade.  H in {full,
           stateless}. Proves the env adds nothing and drops nothing.
  check 2: A(stateless) vs the STORED episode (which ran stateless with the Opus 4.8 resolver): per-turn action
           types, what the agent was shown, and the grade. Differences can only come from the resolver (LLM vs
           deterministic) or from the frozen copy differing from the code that produced the run. Reported as rates.

  cd ~/ADS_folder/ADS_shared/dataset_generation_code/rl_v10
  RPG_PROTO=rpg_v9 RPG_SYNERGY_SOFT=20 ~/miniconda3/envs/ADS/bin/python replay_parity.py --workers 48 --out OUT_DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor  # non-daemonic workers: the code sandbox spawns children

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("RLV10_HARDEN_SANDBOX", "0")        # A must be the release sandbox; B opts in below
import stepenv                                            # noqa: E402  (puts the frozen bench on sys.path)
from stepenv import BenchStepEnv                          # noqa: E402
from agent_loop import run_world, load_world_file         # noqa: E402
from simulator import Simulator                           # noqa: E402

REL = os.environ.get("RLV10_REL", os.path.expanduser("~/ADS_folder/ADS_shared/rpg_benchmark"))
_VOLATILE = {"latency_s", "usage", "served_model", "text_blocks", "model_reasoning", "finish_reason"}


class ReplayLLM:
    def __init__(self, replies):
        self.replies, self.i, self.prompts = list(replies), 0, []
        self.last_finish_reason = "stop"
        self.last_reasoning = None
        self.last_usage = None
        self.exhausted = 0

    def generate_messages(self, messages, max_new_tokens=None):
        self.prompts.append([dict(m) for m in messages])
        if self.i < len(self.replies):
            r = self.replies[self.i]
            self.i += 1
            return r
        self.exhausted += 1
        return ""

    def generate(self, s, u, max_new_tokens=None):
        return self.generate_messages([{"role": "system", "content": s}, {"role": "user", "content": u}])


def _clean(obj):
    """JSON round trip (tuples -> lists, numpy -> float) minus volatile fields, so A and B compare as stored."""
    res = json.loads(json.dumps(obj, default=str))
    for t in res.get("turns", []):
        for k in _VOLATILE:
            t.pop(k, None)
    return res


def _strip_cache(prompts):
    return [[{k: v for k, v in m.items() if k != "cache"} for m in p] for p in prompts]


def run_a(world_file, replies, history, tmp):
    world, pre = load_world_file(world_file)
    sim = Simulator(world, resolver_llm=None, data_dir=os.path.join(tmp, f"a_{history}"), precomputed=pre)
    llm = ReplayLLM(replies)
    res = run_world(sim, llm, 32, 15, False, history=history)
    return res, _strip_cache(llm.prompts), llm.exhausted


B_HARDENED = os.environ.get("RLV10_PARITY_B_HARDENED", "0") == "1"


def run_b(world_file, replies, history, tmp):
    if B_HARDENED:                         # B = the env with the hardened sandbox (what training uses)
        import sandbox_hardened
        sandbox_hardened.install()
        try:
            return _run_b(world_file, replies, history, tmp)
        finally:
            sandbox_hardened.uninstall()
    return _run_b(world_file, replies, history, tmp)


def _run_b(world_file, replies, history, tmp):
    env = BenchStepEnv(world_file, os.path.join(tmp, f"b_{history}"), history=history)
    first = env.reset()
    prompts = [first]
    i, out = 0, None
    while True:
        r = replies[i] if i < len(replies) else ""
        i += 1
        out = env.step(r)
        if out.done:
            break
        prompts.append(env._msgs)
    return env.result, _strip_cache([[{"role": m["role"], "content": m["content"]} for m in p] for p in prompts]), out


_TIME_RE = re.compile(r"(Time:\s+)\d\d:\d\d:\d\d")
_DATE_RE = re.compile(r"(Date:\s+)\w{3}, \d\d \w{3} \d{4}")


def _mask(s: str, dd: str) -> str:
    """Mask what legitimately differs between two replays of the same episode: the episode's data folder and the
    wall-clock date/time that statsmodels prints in model summaries (agent code)."""
    s = s.replace(dd, "<DD>")
    return _DATE_RE.sub(r"\1<DATE>", _TIME_RE.sub(r"\1<TIME>", s))


def _norm_shown(s):
    return s if isinstance(s, str) else json.dumps(s, default=str)


def job(args):
    path, model, subset = args
    rec = {"file": path, "model": model, "subset": subset}
    try:
        stored = json.load(open(path))
        wid = stored["world_id"]
        wf = os.path.join(REL, "data/test", f"world_{wid}.json.gz")
        replies = [t.get("raw", "") or "" for t in stored["turns"] if t.get("turn") != "post"]
        rec.update(world_id=wid, n_replies=len(replies))
        with tempfile.TemporaryDirectory(prefix="parity_") as tmp:
            for h in ("full", "stateless"):
                ra, pa, exh = run_a(wf, replies, h, tmp)
                rb, pb, outb = run_b(wf, replies, h, tmp)
                ca, cb = _clean(ra), _clean(rb)
                # data_dir differs by construction (a_/b_ folders) and appears only inside code paths; compare
                # with it masked
                sa = _mask(json.dumps(ca, sort_keys=True), os.path.join(tmp, f"a_{h}"))
                sb = _mask(json.dumps(cb, sort_keys=True), os.path.join(tmp, f"b_{h}"))
                rec[f"{h}_result_equal"] = sa == sb
                rec[f"{h}_prompts_equal"] = (_mask(json.dumps(pa), os.path.join(tmp, f"a_{h}"))
                                             == _mask(json.dumps(pb), os.path.join(tmp, f"b_{h}")))
                if sa != sb:
                    i = next((k for k in range(min(len(sa), len(sb))) if sa[k] != sb[k]), 0)
                    rec[f"{h}_first_diff"] = [sa[max(0, i - 150):i + 150], sb[max(0, i - 150):i + 150]]
                rec[f"{h}_n_prompts"] = len(pa)
                rec[f"{h}_exhausted"] = exh
                rec[f"{h}_reward_b"] = outb.reward
                if h == "stateless":
                    g, gs = ra["grade"], stored["grade"]
                    rec["partA_replay"] = int(g["part_a_utility_ok"]); rec["partA_stored"] = int(gs["part_a_utility_ok"])
                    rec["acc_replay"] = int(g["accepted"]); rec["acc_stored"] = int(gs["accepted"])
                    rec["benefit_replay"] = g.get("benefit_recovered"); rec["benefit_stored"] = gs.get("benefit_recovered")
                    rec["battery_replay"] = g["battery_fraction"]; rec["battery_stored"] = gs["battery_fraction"]
                    rec["rec_iv_equal"] = (json.dumps(g["recommended_intervention"], sort_keys=True, default=str)
                                           == json.dumps(gs["recommended_intervention"], sort_keys=True, default=str))
                    ta = [t for t in ra["turns"] if t.get("turn") != "post"]
                    ts = [t for t in stored["turns"] if t.get("turn") != "post"]
                    rec["n_turns_replay"], rec["n_turns_stored"] = len(ta), len(ts)
                    at = sum(1 for x, y in zip(ta, ts) if x.get("action_type") == y.get("action_type")
                             and x.get("error") == y.get("error"))
                    sh = sum(1 for x, y in zip(ta, ts) if _norm_shown(x.get("shown_to_agent", x.get("shown_next")))
                             == _norm_shown(y.get("shown_to_agent", y.get("shown_next"))))
                    rec["turns_action_equal"], rec["turns_shown_equal"] = at, sh
                    rec["queries_replay"], rec["queries_stored"] = ra["queries_used"], stored["queries_used"]
                    # the same comparison for B (the RL env in its training configuration) vs the stored episode
                    gb = rb["grade"]
                    tb = [t for t in rb["turns"] if t.get("turn") != "post"]
                    rec["B_partA_equal"] = int(gb["part_a_utility_ok"]) == int(gs["part_a_utility_ok"])
                    rec["B_acc_equal"] = int(gb["accepted"]) == int(gs["accepted"])
                    rec["B_rec_iv_equal"] = (json.dumps(gb["recommended_intervention"], sort_keys=True, default=str)
                                             == json.dumps(gs["recommended_intervention"], sort_keys=True, default=str))
                    rec["B_benefit_absdiff"] = (abs(float(gb["benefit_recovered"]) - float(gs["benefit_recovered"]))
                                                if gb.get("benefit_recovered") is not None
                                                and gs.get("benefit_recovered") is not None else None)
                    rec["B_n_turns_equal"] = len(tb) == len(ts)
                    rec["B_turns_action_equal"] = sum(1 for x, y in zip(tb, ts) if x.get("action_type") == y.get("action_type")
                                                      and x.get("error") == y.get("error"))
                    shown_diff = [k for k, (x, y) in enumerate(zip(tb, ts))
                                  if _norm_shown(x.get("shown_to_agent", x.get("shown_next")))
                                  != _norm_shown(y.get("shown_to_agent", y.get("shown_next")))]
                    rec["B_turns_shown_equal"] = len(tb[:len(ts)]) - len(shown_diff)
                    rec["B_shown_diff_turns"] = shown_diff
                    rec["B_shown_diff_kinds"] = [tb[k].get("action_type") or tb[k].get("attempted_action_type")
                                                 for k in shown_diff]
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    files = [(f, f.split("/out/gate_v10.2/")[1].split("/")[0], "core54")
             for f in sorted(glob.glob(os.path.join(REL, "out/gate_v10.2/*/shard_*/result_*.json")))]
    files += [(f, "opus55", "rest486") for f in sorted(glob.glob(os.path.join(REL, "out/opus55_v10.2_rest486/shard_*/result_*.json")))]
    if args.models:
        keep = set(args.models.split(","))
        files = [x for x in files if x[1] in keep]
    if args.limit:
        files = files[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(files)} stored episodes", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(job, files, chunksize=1)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
    with open(os.path.join(args.out, "replay_parity.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    ok = [r for r in rows if "error" not in r]
    n = len(ok)

    def cnt(k):
        return sum(1 for r in ok if r.get(k))
    lines = [f"episodes {len(rows)}, errors {len(rows) - n}; B sandbox: {'hardened' if B_HARDENED else 'release'}"]
    for h in ("full", "stateless"):
        lines.append(f"[{h}] A==B result {cnt(f'{h}_result_equal')}/{n}, prompts {cnt(f'{h}_prompts_equal')}/{n}, "
                     f"replies exhausted {sum(1 for r in ok if r.get(f'{h}_exhausted'))}")
    lines.append(f"[stored vs replay, stateless] partA equal {sum(r['partA_replay'] == r['partA_stored'] for r in ok)}/{n}, "
                 f"accepted equal {sum(r['acc_replay'] == r['acc_stored'] for r in ok)}/{n}, "
                 f"recommended intervention equal {cnt('rec_iv_equal')}/{n}, "
                 f"turn count equal {sum(r['n_turns_replay'] == r['n_turns_stored'] for r in ok)}/{n}")
    tt = sum(r["n_turns_stored"] for r in ok)
    lines.append(f"[stored vs replay] turns with same action+error {sum(r['turns_action_equal'] for r in ok)}/{tt}, "
                 f"same text shown {sum(r['turns_shown_equal'] for r in ok)}/{tt}")
    lines.append(f"partA stored {sum(r['partA_stored'] for r in ok)} replay {sum(r['partA_replay'] for r in ok)}; "
                 f"accepted stored {sum(r['acc_stored'] for r in ok)} replay {sum(r['acc_replay'] for r in ok)}")
    lines.append(f"[stored vs B=env, stateless] partA equal {cnt('B_partA_equal')}/{n}, accepted equal "
                 f"{cnt('B_acc_equal')}/{n}, recommended intervention equal {cnt('B_rec_iv_equal')}/{n}, "
                 f"turn count equal {cnt('B_n_turns_equal')}/{n}, max |benefit diff| "
                 f"{max([r['B_benefit_absdiff'] for r in ok if r.get('B_benefit_absdiff') is not None] or [0]):.4f}")
    lines.append(f"[stored vs B] turns with same action+error {sum(r['B_turns_action_equal'] for r in ok)}/{tt}, "
                 f"same text shown {sum(r['B_turns_shown_equal'] for r in ok)}/{tt}; shown-differs turn kinds "
                 f"{dict(__import__('collections').Counter(k for r in ok for k in r['B_shown_diff_kinds']))}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(args.out, "replay_parity_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
