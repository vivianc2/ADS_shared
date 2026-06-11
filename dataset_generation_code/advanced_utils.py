"""Shared helpers for advanced causal question generation.

Exact ground-truth computations for the advanced question types:
- Interventional expected value E[Y|do(X=v)] via CausalInference.
- Backdoor adjustment sets via CausalInference.
- Mediator path classification via networkx.

All helpers are pure functions over a rebuilt pgmpy BN.  Reuses
`check_faithfulness.rebuild_model` to convert a world-JSON dict into
`(DiscreteBayesianNetwork, nx.DiGraph)`.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

import networkx as nx
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from check_faithfulness import rebuild_model


# ---------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------

def rebuild_from_world(world: Dict[str, Any]) -> Tuple[DiscreteBayesianNetwork, nx.DiGraph]:
    """Thin wrapper around check_faithfulness.rebuild_model."""
    return rebuild_model(world)


# ---------------------------------------------------------------------
# Scoring conventions
# ---------------------------------------------------------------------

def scoring_for_target(var_spec: Dict[str, Any]) -> Callable[[str], float]:
    """Return scoring(state) -> float.

    Convention: higher score = worse. If preferred_low is True, scoring is
    index-in-state-list (lower index = better, so reductions mean argmin).
    If preferred_low is False, scoring is -index (higher is better; argmin
    of this scoring still means the intervention we prefer).
    """
    states = list(var_spec["values"])
    preferred_low = var_spec.get("preferred_low")
    if preferred_low not in (True, False):
        raise ValueError(
            f"scoring_for_target: variable {var_spec.get('name')!r} has no "
            f"preferred_low set"
        )
    sign = 1.0 if preferred_low else -1.0
    index = {s: i for i, s in enumerate(states)}

    def score(state: str) -> float:
        return sign * float(index[state])

    return score


def target_is_scoreable(var_spec: Dict[str, Any]) -> bool:
    return var_spec.get("preferred_low") in (True, False)


# ---------------------------------------------------------------------
# Expected value under do()
# ---------------------------------------------------------------------

def mutilate_model(
    model: DiscreteBayesianNetwork, do_map: Dict[str, str],
) -> DiscreteBayesianNetwork:
    """Return a deep-copied model with do_map applied by graph mutilation.

    For each (X, v) in do_map:
      - remove every incoming edge into X
      - replace X's CPD with a point mass at v

    Robust against pgmpy 1.0's CausalInference.query edge-case where the
    target is upstream of the do-variable.
    """
    mutilated = deepcopy(model)
    for var, state in do_map.items():
        parents = list(mutilated.get_parents(var))
        for p in parents:
            mutilated.remove_edge(p, var)
        old_cpd = mutilated.get_cpds(var)
        states = list(old_cpd.state_names[var])
        if state not in states:
            raise ValueError(f"do({var}={state!r}) -- state not in {states}")
        idx = states.index(state)
        values = [[0.0] for _ in range(len(states))]
        values[idx] = [1.0]
        new_cpd = TabularCPD(
            variable=var, variable_card=len(states),
            values=values, state_names={var: states},
        )
        mutilated.remove_cpds(old_cpd)
        mutilated.add_cpds(new_cpd)
    return mutilated


def expected_value_under_do(
    ci_or_model,
    target: str,
    do_map: Dict[str, str],
    scoring: Callable[[str], float],
    evidence: Optional[Dict[str, str]] = None,
) -> float:
    """Exact E[scoring(Y) | do(X=v), evidence] via graph mutilation + VE.

    First argument may be either a CausalInference (we pull .model off it) or
    a DiscreteBayesianNetwork directly -- this keeps call sites backwards
    compatible.
    """
    model = ci_or_model.model if isinstance(ci_or_model, CausalInference) else ci_or_model
    mutilated = mutilate_model(model, do_map) if do_map else model
    ve = VariableElimination(mutilated)
    factor = ve.query(variables=[target], evidence=evidence, show_progress=False)
    probs = factor.values
    states = factor.state_names[target]
    return float(sum(float(p) * scoring(s) for p, s in zip(probs, states)))


def expected_value_observational(
    ve: VariableElimination,
    target: str,
    scoring: Callable[[str], float],
) -> float:
    factor = ve.query(variables=[target], show_progress=False)
    probs = factor.values
    states = factor.state_names[target]
    return float(sum(float(p) * scoring(s) for p, s in zip(probs, states)))


# ---------------------------------------------------------------------
# Intervenable action enumeration
# ---------------------------------------------------------------------

def get_non_intervenable_names(world: Dict[str, Any]) -> Set[str]:
    field = world.get("non_intervenable_variables", [])
    if isinstance(field, dict):
        return set(field.keys())
    names: Set[str] = set()
    for item in field:
        if isinstance(item, dict) and "name" in item:
            names.add(item["name"])
        elif isinstance(item, str):
            names.add(item)
    return names


def enumerate_intervenable_actions(
    world: Dict[str, Any],
    exclude: Optional[Set[str]] = None,
) -> Iterator[Tuple[str, str]]:
    """Yield (var, value) for every intervenable var and every state value."""
    exclude = exclude or set()
    non_int = get_non_intervenable_names(world)
    for v in world["variables"]:
        name = v["name"]
        if name in non_int or name in exclude:
            continue
        for val in v["values"]:
            yield (name, val)


def intervenable_var_names(
    world: Dict[str, Any], exclude: Optional[Set[str]] = None,
) -> List[str]:
    exclude = exclude or set()
    non_int = get_non_intervenable_names(world)
    return [
        v["name"] for v in world["variables"]
        if v["name"] not in non_int and v["name"] not in exclude
    ]


# ---------------------------------------------------------------------
# Effect table
# ---------------------------------------------------------------------

def compute_effect_table(
    ci: CausalInference,
    world: Dict[str, Any],
    target: str,
    scoring: Callable[[str], float],
    exclude: Optional[Set[str]] = None,
) -> Dict[Tuple[str, str], float]:
    """For every intervenable (X, v) (excluding target and `exclude`), compute
    E[scoring(target) | do(X=v)].  Returns dict keyed by (X, v).

    Lower value = better (under our unified scoring convention).
    """
    exclude = set(exclude or set())
    exclude.add(target)
    effects: Dict[Tuple[str, str], float] = {}
    for X, v in enumerate_intervenable_actions(world, exclude=exclude):
        effects[(X, v)] = expected_value_under_do(ci, target, {X: v}, scoring)
    return effects


# ---------------------------------------------------------------------
# Backdoor adjustment sets
# ---------------------------------------------------------------------

def get_minimal_backdoor_adjustment_sets(
    ci: CausalInference,
    X: str,
    Y: str,
    max_size: int = 4,
) -> List[FrozenSet[str]]:
    """Return subset-minimal backdoor adjustment sets between X and Y.

    Return values carry three distinct meanings:
      - `[frozenset()]`  -> empty set is a valid adjustment (no backdoor paths);
                            i.e. no conditioning is needed.
      - `[frozenset({...}), ...]` -> non-empty minimal adjustment sets.
      - `[]`             -> no valid adjustment exists (unidentifiable via
                            backdoor within `max_size`).

    Note pgmpy 1.0's `get_all_backdoor_adjustment_sets` returns an empty
    frozenset both when no adjustment is needed and when the effect is
    unidentifiable, so we disambiguate with `is_valid_backdoor_adjustment_set`.
    """
    try:
        empty_ok = bool(ci.is_valid_backdoor_adjustment_set(X, Y, frozenset()))
    except Exception:
        empty_ok = False
    if empty_ok:
        return [frozenset()]

    try:
        raw = ci.get_all_backdoor_adjustment_sets(X, Y)
    except Exception:
        return []
    sets: List[FrozenSet[str]] = [frozenset(s) for s in raw if len(s) > 0]
    sets = [s for s in sets if len(s) <= max_size]
    minimal: List[FrozenSet[str]] = []
    for s in sets:
        if any((other < s) for other in sets):
            continue
        if s not in minimal:
            minimal.append(s)
    return minimal


def is_valid_backdoor_set(
    ci: CausalInference, X: str, Y: str, Z: Set[str],
) -> bool:
    try:
        return bool(ci.is_valid_backdoor_adjustment_set(X, Y, frozenset(Z)))
    except Exception:
        return False


# ---------------------------------------------------------------------
# Mediator classification
# ---------------------------------------------------------------------

def classify_mediator(
    g: nx.DiGraph,
    T: str,
    M: str,
    O: str,
    cutoff: int = 6,  # accepted for backward compat; unused
) -> str:
    """Return one of 'only_through_M' / 'also_direct_or_other' / 'not_mediator'.

    - 'not_mediator': T has no directed path to O, OR no T→M→O chain exists.
    - 'only_through_M': M is on every directed T→O path (removing M from the
      graph disconnects T from O).
    - 'also_direct_or_other': T→M→O exists AND some T→O path bypasses M.

    Implementation is O(N + E) using reachability checks + subgraph removal,
    which is dramatically faster than enumerating all simple paths (the
    latter is exponential in the worst case on 30-node graphs).
    """
    if T == M or T == O or M == O:
        return "not_mediator"
    if not nx.has_path(g, T, O):
        return "not_mediator"
    if not (nx.has_path(g, T, M) and nx.has_path(g, M, O)):
        return "not_mediator"

    # Does a T->O path exist that bypasses M?  Remove M and recheck reachability.
    sub_nodes = [n for n in g.nodes if n != M]
    sub = g.subgraph(sub_nodes)
    bypasses_M = nx.has_path(sub, T, O)

    if bypasses_M:
        return "also_direct_or_other"
    return "only_through_M"


# ---------------------------------------------------------------------
# Directed-graph helpers
# ---------------------------------------------------------------------

def has_directed_path(g: nx.DiGraph, src: str, dst: str) -> bool:
    return src != dst and nx.has_path(g, src, dst)


def directed_ancestors(g: nx.DiGraph, node: str) -> Set[str]:
    return set(nx.ancestors(g, node))


def directed_descendants(g: nx.DiGraph, node: str) -> Set[str]:
    return set(nx.descendants(g, node))
