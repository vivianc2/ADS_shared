#!/usr/bin/env python3
"""ACED-Qual / ACED-Core accuracy per agent for finished runs, next to the paper's models.

  python summarize.py out/opus55 [out/sonnet55 ...]       # new runs (reads <run>/eval/eval_*.json)
  python summarize.py --paper-only                        # just the paper table (paper_reference.json)
  python summarize.py --build-reference <ADS/framework_code/evaluations>   # rebuild paper_reference.json

Grouping follows the paper's figure notebook (plot_for_neurips.ipynb, round4 figures):
  ACED-Qual = yes/no structural questions;  ACED-Core = causal-list structural questions + all decision questions.
  Agents: guess (guess-only), agent = reasoning, coder = active+code, coder_new = modular.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "paper_reference.json")
YESNO = {"causal_effect", "direct_marginal", "chain_marginal", "fork_marginal", "v_structure_marginal",
         "other_marginal", "chain_conditional", "fork_conditional", "v_structure_conditional", "other_conditional"}
LIST = {"all_causes_of", "all_effects_of"}
AGENTS = ["guess", "agent", "coder", "coder_new"]
# Structural questions count only on the 30-world subset (as the paper's figures do: some guess-only files cover all
# 60 structural worlds). core_struct's 10 worlds are inside it.
SUBSET = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(HERE, "data", "guess_struct", "world_*.json"))}


def rows(path, decision):
    """(group, world, correct) for every scored question in one eval file."""
    out = []
    for e in json.load(open(path)).get("evaluated", []):
        qt = e.get("question_type")
        group = "core" if decision or qt in LIST else "qual" if qt in YESNO else None
        world = e.get("world_name") or os.path.basename(str(e.get("world_file", "")))[:-5]
        if group and (decision or world in SUBSET):
            out.append((group, world, int(bool((e.get("eval") or {}).get("correct")))))
    return out


def agent_of(name):
    if "zero_shot" in name or "_guess_" in name:
        return "guess"
    for a in ("coder_new", "coder", "agent"):
        if f"_{a}_" in name:
            return a
    return None


def cells(files):
    """{agent: {group: [correct, total, worlds]}} over a list of eval files."""
    acc = {}
    for f in files:
        name = os.path.basename(f)
        a = agent_of(name)
        if a is None:
            continue
        for g, w, c in rows(f, decision=("decision" in name or "adv_v3" in name)):
            cell = acc.setdefault(a, {}).setdefault(g, [0, 0, set()])
            cell[0] += c
            cell[1] += 1
            cell[2].add(w)
    return {a: {g: [c, n, len(ws)] for g, (c, n, ws) in gs.items()} for a, gs in acc.items()}


def build_reference(eval_dir):
    """Paper cells: round4 evals for the 3 active agents, unrestricted guess-only on the same sets the figures used."""
    ref = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, "round", "eval_*_round4.json"))):
        model = os.path.basename(f)[5:].split("_")[0]
        ref.setdefault(model, []).append(f)
    for f in sorted(glob.glob(os.path.join(eval_dir, "for_paper", "eval_*zero_shot*.json"))):
        stem = os.path.basename(f)[5:-5].replace("_v2", "")
        model = stem.split("_")[-1] if stem.startswith("zero_shot") else stem.split("_")[0]
        ref.setdefault(model, []).append(f)
    out = {m: cells(fs) for m, fs in ref.items()}
    json.dump({"note": "ACED-Bench paper cells (round4 = 4-turn budget; guess = guess-only). [correct, total, worlds]. "
                       "Model keys as in the paper's file names: opus = Claude Opus 4.6, sonnet46 = Claude Sonnet 4.6, "
                       "gem3pro = Gemini 3.1 Pro preview, dsv4pro = DeepSeek V4 Pro, qwen36 = Qwen3.6-27B, "
                       "oss120 = GPT-OSS-120B, llama4mav = Llama 4 Maverick, gpt4o = GPT-4o, llama = Llama 3.3 70B.",
               "cells": out}, open(REF, "w"), indent=1)
    print(f"wrote {REF} ({len(out)} models)")


def show(table):
    head = f"{'model':14s}" + "".join(f"{a + ' Qual':>16s}{a + ' Core':>16s}" for a in AGENTS)
    print(head)
    print("-" * len(head))
    for model, c in table.items():
        line = f"{model:14s}"
        for a in AGENTS:
            for g in ("qual", "core"):
                x = c.get(a, {}).get(g)
                line += f"{'-':>16s}" if not x else f"{x[0] / x[1]:>9.2f} ({x[1]:3d})"
        print(line)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--build-reference"]:
        build_reference(args[1])
        sys.exit()
    table = {}
    for d in args:
        if d != "--paper-only":
            table[os.path.basename(os.path.normpath(d)) + " *"] = cells(glob.glob(os.path.join(d, "eval", "eval_*.json")))
    if os.path.exists(REF):
        table.update(json.load(open(REF))["cells"])
    show(table)
    print("\n* = new run. Numbers: accuracy (questions). Paper cells from paper_reference.json.")
