# check_faithfulness.py — Audit a world dataset for faithfulness violations.
#
# Background
# ----------
# The older generator `world_gen_causal.py` labels every question using
# graph-level d-separation only, without verifying that the random CPDs
# actually realize the claimed (in)dependence numerically.  A pair of
# d-connected variables can end up with tiny mutual information once CPDs
# dilute signal along a path — the golden answer says "dependent" but the
# model, looking at samples, sees something indistinguishable from
# independent.
#
# This script audits a directory of world JSON files produced by any of the
# generators.  For each question it rebuilds the Bayesian Network from the
# serialized CPDs, compares the graph-based golden answer against the
# numerical ground truth (TV-distance test), and reports which questions
# are structurally unfaithful.
#
# Usage
# -----
#   python check_faithfulness.py ./all_out_bn/out_bn_3_4 \
#       --eps 0.02 -o ./audits/eval_faithfulness_3_4.json
#
# Semantics
# ---------
# The faithfulness check matches the one in world_gen_small.py:
#   • INDEP claim (d-separated):  faithful  ⇔  TV(joint, product) ≤ eps
#                                              for every evidence state.
#   • DEP claim   (d-connected):  faithful  ⇔  TV > eps for some evidence.
#
# Causal-effect and list questions are about ancestor/descendant relations,
# which are graph-level facts; we proxy faithfulness there by requiring
# marginal dependence between the cause/effect pair to be numerically
# detectable (TV > eps), since a d-connected directed path implies marginal
# dependence under faithfulness.  "No-cause" / empty-list answers are
# structurally exact and always faithful.

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from itertools import product as iproduct
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from world_gen import find_ancestors, is_d_separated
from world_gen_causal import (
    _ALL_CAUSES_OF_TEMPLATES,
    _ALL_EFFECTS_OF_TEMPLATES,
    _CAUSAL_EFFECT_TEMPLATES,
    _COND1_DEP_TEMPLATES,
    _COND1_INDEP_TEMPLATES,
    _COND2_DEP_TEMPLATES,
    _COND2_INDEP_TEMPLATES,
    _MARGINAL_DEP_TEMPLATES,
    _MARGINAL_INDEP_TEMPLATES,
)


# ===================================================================
# Template → regex compilation
# ===================================================================
#
# Each template is compiled into a regex with named capture groups so we
# can recover variable roles (which name is X, which is Z, etc.) from the
# rendered question text.  The quoted-variable convention ('{x}', '{y}',
# ...) in the templates means variable names never contain apostrophes,
# so `[^']+` is a safe capture class.

def _template_to_regex(template: str) -> re.Pattern:
    out = []
    i = 0
    while i < len(template):
        c = template[i]
        if c == "{":
            j = template.index("}", i)
            name = template[i + 1 : j]
            out.append(f"(?P<{name}>[^']+)")
            i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


# kind tags used by per-question checkers below
_PATTERN_CATALOG: List[Tuple[re.Pattern, str]] = []

for _t in _CAUSAL_EFFECT_TEMPLATES:
    _PATTERN_CATALOG.append((_template_to_regex(_t), "causal_effect"))
for _t in _ALL_CAUSES_OF_TEMPLATES:
    _PATTERN_CATALOG.append((_template_to_regex(_t), "all_causes_of"))
for _t in _ALL_EFFECTS_OF_TEMPLATES:
    _PATTERN_CATALOG.append((_template_to_regex(_t), "all_effects_of"))
for _t in _MARGINAL_INDEP_TEMPLATES:
    _PATTERN_CATALOG.append((_template_to_regex(_t), "marginal_indep"))
for _t in _MARGINAL_DEP_TEMPLATES:
    _PATTERN_CATALOG.append((_template_to_regex(_t), "marginal_dep"))
for _t in _COND1_INDEP_TEMPLATES:
    _PATTERN_CATALOG.append((_template_to_regex(_t), "cond1_indep"))
for _t in _COND1_DEP_TEMPLATES:
    _PATTERN_CATALOG.append((_template_to_regex(_t), "cond1_dep"))
for _t in _COND2_INDEP_TEMPLATES:
    _PATTERN_CATALOG.append((_template_to_regex(_t), "cond2_indep"))
for _t in _COND2_DEP_TEMPLATES:
    _PATTERN_CATALOG.append((_template_to_regex(_t), "cond2_dep"))


def parse_question(q_text: str) -> Optional[Dict[str, Any]]:
    """Match q_text against the template catalog and return {'kind': ..., <named groups>}."""
    for regex, kind in _PATTERN_CATALOG:
        m = regex.match(q_text)
        if m:
            return {"kind": kind, **m.groupdict()}
    return None


# ===================================================================
# BN reconstruction from world JSON
# ===================================================================

def rebuild_model(world: Dict[str, Any]) -> Tuple[DiscreteBayesianNetwork, nx.DiGraph]:
    """Materialize a pgmpy BN + named DiGraph from a world JSON.

    Assumes the serialized CPD shape from `world_gen.serialize_cpds`:
    `values` is (child_card, product_of_parent_cards), parents in the
    exact order used at build time.
    """
    state_names = {v["name"]: list(v["values"]) for v in world["variables"]}
    names = [v["name"] for v in world["variables"]]
    edges = [tuple(e) for e in world["edges"]]

    g = nx.DiGraph()
    g.add_nodes_from(names)
    g.add_edges_from(edges)

    model = DiscreteBayesianNetwork()
    model.add_nodes_from(names)
    model.add_edges_from(edges)

    for spec in world["cpds"]:
        child = spec["child"]
        parents = list(spec["parents"])
        child_card = int(spec["cardinality"])
        values = np.asarray(spec["values"], dtype=float)

        if parents:
            evidence_card = [len(state_names[p]) for p in parents]
            sn = {child: state_names[child], **{p: state_names[p] for p in parents}}
            cpd = TabularCPD(
                variable=child,
                variable_card=child_card,
                values=values,
                evidence=parents,
                evidence_card=evidence_card,
                state_names=sn,
            )
        else:
            cpd = TabularCPD(
                variable=child,
                variable_card=child_card,
                values=values,
                state_names={child: state_names[child]},
            )
        model.add_cpds(cpd)

    model.check_model()
    return model, g


# ===================================================================
# TV helpers
# ===================================================================

def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p.ravel() - q.ravel()).sum())


def _marginal_tv(
    ve: VariableElimination,
    a: str,
    b: str,
    evidence: Optional[Dict[str, str]] = None,
) -> float:
    """TV between P(a, b | evidence) and P(a | evidence) · P(b | evidence)."""
    joint = ve.query(variables=[a, b], evidence=evidence, show_progress=False)
    pa = ve.query(variables=[a], evidence=evidence, show_progress=False)
    pb = ve.query(variables=[b], evidence=evidence, show_progress=False)
    joint_arr = joint.values
    if joint.variables[0] != a:
        joint_arr = joint_arr.T
    product = np.outer(pa.values, pb.values)
    return _tv(joint_arr, product)


# ===================================================================
# Per-question checkers
# ===================================================================

def _check_marginal(
    ve: VariableElimination,
    g: nx.DiGraph,
    parsed: Dict[str, Any],
    stored_answer: str,
    eps: float,
) -> Dict[str, Any]:
    a, b = parsed["x"], parsed["y"]
    d_sep = is_d_separated(g, a, b, set())
    tv = _marginal_tv(ve, a, b)
    numeric_indep = tv <= eps

    framing_indep = parsed["kind"] == "marginal_indep"
    graph_answer = "Yes" if (framing_indep == d_sep) else "No"
    numeric_answer = "Yes" if (framing_indep == numeric_indep) else "No"

    return {
        "faithful": graph_answer == numeric_answer,
        "graph_answer": graph_answer,
        "numeric_answer": numeric_answer,
        "stored_answer": stored_answer,
        "tv": tv,
        "d_separated": d_sep,
        "pair": [a, b],
    }


def _check_conditional(
    ve: VariableElimination,
    model: DiscreteBayesianNetwork,
    g: nx.DiGraph,
    parsed: Dict[str, Any],
    stored_answer: str,
    eps: float,
) -> Dict[str, Any]:
    kind = parsed["kind"]
    a, b = parsed["x"], parsed["y"]
    z_vars = [parsed["z"]] if kind in ("cond1_indep", "cond1_dep") else [parsed["z1"], parsed["z2"]]

    d_sep = is_d_separated(g, a, b, set(z_vars))

    # DEP claim needs at least one evidence state with TV > eps.
    # INDEP claim needs every evidence state with TV <= eps.
    z_state_lists = [model.get_cpds(z).state_names[z] for z in z_vars]
    tvs: List[float] = []
    for combo in iproduct(*z_state_lists):
        evidence = {z: s for z, s in zip(z_vars, combo)}
        tvs.append(_marginal_tv(ve, a, b, evidence))

    any_above = any(tv > eps for tv in tvs)
    numeric_indep = not any_above

    framing_indep = kind in ("cond1_indep", "cond2_indep")
    graph_answer = "Yes" if (framing_indep == d_sep) else "No"
    numeric_answer = "Yes" if (framing_indep == numeric_indep) else "No"

    return {
        "faithful": graph_answer == numeric_answer,
        "graph_answer": graph_answer,
        "numeric_answer": numeric_answer,
        "stored_answer": stored_answer,
        "max_tv": max(tvs) if tvs else 0.0,
        "min_tv": min(tvs) if tvs else 0.0,
        "n_evidence_states": len(tvs),
        "d_separated": d_sep,
        "pair": [a, b],
        "Z": z_vars,
    }


def _check_causal_effect(
    ve: VariableElimination,
    g: nx.DiGraph,
    parsed: Dict[str, Any],
    stored_answer: str,
    eps: float,
) -> Dict[str, Any]:
    """Proxy check: under faithfulness, a directed ancestor path implies
    marginal dependence.  Non-ancestry is a structural fact — exact for
    any CPD parameterization, so "No" answers are trivially faithful.
    """
    x, y = parsed["x"], parsed["y"]
    # Template has one form ("Is '{y}' caused by '{x}'?") where y appears
    # before x, so we check both orderings to determine the graph verdict.
    is_ancestor = (x in find_ancestors(g, y)) or (y in find_ancestors(g, x))
    graph_answer = "Yes" if is_ancestor else "No"

    tv = _marginal_tv(ve, x, y)

    if is_ancestor:
        faithful = tv > eps
        return {
            "faithful": faithful,
            "graph_answer": graph_answer,
            "stored_answer": stored_answer,
            "tv": tv,
            "is_ancestor": True,
            "pair": [x, y],
            "note": "marginal TV proxy for interventional effect",
        }
    # is_ancestor == False → structurally exact
    return {
        "faithful": True,
        "graph_answer": graph_answer,
        "stored_answer": stored_answer,
        "tv": tv,
        "is_ancestor": False,
        "pair": [x, y],
        "note": "non-ancestry is structural; always faithful",
    }


def _check_list(
    ve: VariableElimination,
    parsed: Dict[str, Any],
    stored_answer: Any,
    eps: float,
) -> Dict[str, Any]:
    """Verify that every listed ancestor/descendant has detectable marginal
    dependence with the target.  Empty lists are structurally exact.
    """
    node = parsed["node"]
    listed = list(stored_answer) if isinstance(stored_answer, list) else []

    if not listed:
        return {
            "faithful": True,
            "node": node,
            "stored_answer": [],
            "note": "empty list is structurally exact",
        }

    weak = []
    for v in listed:
        tv = _marginal_tv(ve, node, v)
        if tv <= eps:
            weak.append({"var": v, "tv": tv})

    return {
        "faithful": len(weak) == 0,
        "node": node,
        "stored_answer": listed,
        "weak_pairs": weak,
        "note": "marginal TV proxy for interventional effect",
    }


# ===================================================================
# World-level + dataset-level audit
# ===================================================================

@dataclass
class Summary:
    total_worlds: int = 0
    total_questions: int = 0
    parse_failures: int = 0
    reconstruction_failures: int = 0
    unfaithful: int = 0
    by_kind: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_world: Dict[str, int] = field(default_factory=dict)

    def bump_kind(self, kind: str, unfaithful: bool) -> None:
        entry = self.by_kind.setdefault(kind, {"total": 0, "unfaithful": 0})
        entry["total"] += 1
        if unfaithful:
            entry["unfaithful"] += 1


def audit_world(path: str, eps: float) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        world = json.load(f)

    try:
        model, g = rebuild_model(world)
    except Exception as exc:
        return {
            "path": path,
            "reconstruction_error": f"{type(exc).__name__}: {exc}",
            "results": [],
        }

    ve = VariableElimination(model)

    rep: Dict[str, Any] = {
        "path": path,
        "topic": world["meta"].get("topic"),
        "n_nodes": world["meta"].get("n_nodes"),
        "seed": world["meta"].get("seed"),
        "topology": world["meta"].get("topology"),
        "results": [],
    }

    for q in world["questions"]:
        parsed = parse_question(q["question"])
        record: Dict[str, Any] = {
            "id": q.get("id"),
            "question_type": q.get("question_type"),
            "question_group": q.get("question_group"),
            "question": q["question"],
            "answer": q["answer"],
        }
        if parsed is None:
            record["faithful"] = None
            record["error"] = "template match failed"
            rep["results"].append(record)
            continue

        kind = parsed["kind"]
        try:
            if kind in ("marginal_indep", "marginal_dep"):
                res = _check_marginal(ve, g, parsed, q["answer"], eps)
            elif kind in ("cond1_indep", "cond1_dep", "cond2_indep", "cond2_dep"):
                res = _check_conditional(ve, model, g, parsed, q["answer"], eps)
            elif kind == "causal_effect":
                res = _check_causal_effect(ve, g, parsed, q["answer"], eps)
            elif kind in ("all_causes_of", "all_effects_of"):
                res = _check_list(ve, parsed, q["answer"], eps)
            else:
                res = {"faithful": None, "error": f"unknown kind: {kind}"}
        except Exception as exc:
            res = {"faithful": None, "error": f"{type(exc).__name__}: {exc}"}

        res["parsed_kind"] = kind
        record.update(res)
        rep["results"].append(record)

    return rep


def audit_dataset(
    dataset_dir: str,
    eps: float,
    report_path: Optional[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    paths = sorted(glob.glob(os.path.join(dataset_dir, "world_*.json")))
    if not paths:
        print(f"error: no world_*.json found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    summary = Summary()
    world_reports: List[Dict[str, Any]] = []
    unfaithful_items: List[Dict[str, Any]] = []

    for i, path in enumerate(paths, 1):
        if verbose:
            print(f"[{i:3d}/{len(paths)}] {os.path.basename(path)}")
        rep = audit_world(path, eps)
        world_reports.append(rep)
        summary.total_worlds += 1

        if "reconstruction_error" in rep:
            summary.reconstruction_failures += 1
            if verbose:
                print(f"    !! reconstruction failed: {rep['reconstruction_error']}")
            continue

        n_unf_in_world = 0
        for r in rep["results"]:
            summary.total_questions += 1
            qt = r.get("question_type", "?")
            if r.get("faithful") is None:
                summary.parse_failures += 1
                summary.bump_kind(qt, unfaithful=False)  # don't count in unfaithful
                continue

            unf = r["faithful"] is False
            summary.bump_kind(qt, unfaithful=unf)
            if unf:
                summary.unfaithful += 1
                n_unf_in_world += 1
                unfaithful_items.append({"world": os.path.basename(path), **r})

        if n_unf_in_world:
            summary.by_world[os.path.basename(path)] = n_unf_in_world

    _print_summary(summary, eps)

    if report_path:
        payload = {
            "dataset_dir": os.path.abspath(dataset_dir),
            "eps": eps,
            "summary": {
                "total_worlds": summary.total_worlds,
                "total_questions": summary.total_questions,
                "unfaithful_questions": summary.unfaithful,
                "parse_failures": summary.parse_failures,
                "reconstruction_failures": summary.reconstruction_failures,
                "by_kind": summary.by_kind,
                "by_world": summary.by_world,
            },
            "unfaithful_questions": unfaithful_items,
            "world_reports": world_reports,
        }
        os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        print(f"\nReport written to {report_path}")

    return {"summary": summary, "unfaithful": unfaithful_items}


def _print_summary(summary: Summary, eps: float) -> None:
    print()
    print("=" * 68)
    print("FAITHFULNESS AUDIT SUMMARY")
    print("=" * 68)
    print(f"  eps threshold                {eps}")
    print(f"  worlds audited               {summary.total_worlds}")
    print(f"  questions audited            {summary.total_questions}")
    print(f"  unfaithful questions         {summary.unfaithful}")
    print(f"  parse failures               {summary.parse_failures}")
    print(f"  reconstruction failures      {summary.reconstruction_failures}")
    if summary.total_questions:
        pct = 100.0 * summary.unfaithful / summary.total_questions
        print(f"  overall unfaithful rate      {pct:.2f}%")

    if summary.by_kind:
        print("\n  By question type:")
        print(f"    {'type':30s}  {'unfaithful':>10s} / {'total':>6s}   {'%':>6s}")
        for qt in sorted(summary.by_kind):
            stats = summary.by_kind[qt]
            pct = 100.0 * stats["unfaithful"] / stats["total"] if stats["total"] else 0.0
            print(f"    {qt:30s}  {stats['unfaithful']:>10d} / {stats['total']:>6d}   {pct:>5.1f}%")

    if summary.by_world:
        print("\n  Worst worlds (by unfaithful count):")
        worst = sorted(summary.by_world.items(), key=lambda kv: -kv[1])[:10]
        for w, n in worst:
            print(f"    {n:3d}  {w}")


def _json_default(obj: Any) -> Any:
    """Gracefully serialize numpy scalars / arrays / sets for JSON output."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit a world dataset for faithfulness "
                    "(graph-based golden answer vs numerical BN)."
    )
    ap.add_argument(
        "dataset_dir",
        type=str,
        help="Directory containing world_*.json files "
             "(e.g. dataset_generation_code/all_out_bn/out_bn_3_4)",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=0.02,
        help="TV threshold for (in)dependence classification (default: 0.02).",
    )
    ap.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to write the full JSON audit report "
             "(summary + unfaithful questions + per-world results).",
    )
    ap.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-world progress lines.",
    )
    args = ap.parse_args()

    audit_dataset(
        dataset_dir=args.dataset_dir,
        eps=args.eps,
        report_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
