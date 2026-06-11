# world_gen_causal.py — Answer-first, stratified question generation for CAUSAL graphs
#
# Imports all world-generation machinery (LLM, variables, edges, DAG, CPDs,
# story, non-intervenable) from world_gen.py.  Replaces only the question
# generation with a causal-framing, balanced, answer-first approach:
#
#   3 groups  ×  2 questions each  =  6 questions per world
#
# Groups:
#   1. Causal Structure      (easy)        – causal_effect (Yes/No), all_causes_of / all_effects_of (list)
#   2. Marginal Independence (medium)      – 1 Yes + 1 No, verified by d-sep
#   3. Conditional Independence (medium-hard) – 1 Yes + 1 No, verified by d-sep
#
# Group 1 tests causal reasoning:
#   - Yes/No: "Does A cause B?" / "Does A have a causal effect on B?" etc.
#     Answer-first: we pre-decide Yes/No, then search for an ancestor pair that matches.
#   - List: "What causes X?" (ancestors) or "What does X causally affect?" (descendants)
#
# Edges in the generated BN represent real causal mechanisms (A → B means A causes B).
# Do(X=x) interventions affect descendants but NOT parents (do-calculus: incoming
# edges to X are severed, outgoing edges remain intact).

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Import everything we need from world_gen (world generation, graph helpers)
# ---------------------------------------------------------------------------
from world_gen import (
    QwenLLM,
    TOPICS,
    # World-building
    get_variables_chunked,
    get_edge_candidates_chunked,
    build_dag,
    build_bn_with_cpds,
    serialize_cpds,
    save_graph_png,
    generate_story,
    identify_non_intervenable_variables,
    # Graph helpers
    find_v_structures,
    find_chains,
    find_forks,
    find_ancestors,
    find_descendants,
    is_d_separated,
)


# ===================================================================
# Question-template pools (language diversity)
# ===================================================================

# --- Marginal Independence — independence-framing (Yes = independent) ---
_MARGINAL_INDEP_TEMPLATES = [
    "Are '{x}' and '{y}' statistically independent?",
    "Is '{x}' statistically independent of '{y}'?",
    "Is '{x}' independent of '{y}' in this system?",
]

# --- Marginal Independence — dependence-framing (Yes = dependent) ---
_MARGINAL_DEP_TEMPLATES = [
    "Are '{x}' and '{y}' statistically dependent?",
    "Is there a statistical dependence between '{x}' and '{y}'?",
    "Does knowing the value of '{x}' change the probability distribution of '{y}'?",
]

# --- Conditional Independence — 1 conditioning variable, independence-framing ---
_COND1_INDEP_TEMPLATES = [
    "Are '{x}' and '{y}' statistically independent given '{z}'?",
    "Once we know the value of '{z}', are '{x}' and '{y}' statistically independent?",
    "Conditioning on '{z}', is '{x}' statistically independent of '{y}'?",
]

# --- Conditional Independence — 1 conditioning variable, dependence-framing ---
_COND1_DEP_TEMPLATES = [
    "Are '{x}' and '{y}' statistically dependent given '{z}'?",
    "Does knowing the value of '{x}' change the probability distribution of '{y}', even after accounting for '{z}'?",
    "Even after accounting for '{z}', are '{x}' and '{y}' statistically dependent?",
]

# --- Conditional Independence — 2 conditioning variables, independence-framing ---
_COND2_INDEP_TEMPLATES = [
    "Are '{x}' and '{y}' statistically independent given both '{z1}' and '{z2}'?",
    "Once we know '{z1}' and '{z2}', are '{x}' and '{y}' statistically independent?",
]

# --- Conditional Independence — 2 conditioning variables, dependence-framing ---
_COND2_DEP_TEMPLATES = [
    "Are '{x}' and '{y}' statistically dependent given both '{z1}' and '{z2}'?",
    "Does knowing '{x}' change the probability distribution of '{y}', even after accounting for both '{z1}' and '{z2}'?",
]

# ===================================================================
# Post-hoc structure classifier (for analysis metadata only)
# ===================================================================

def _classify_structure(g: nx.DiGraph, x: str, y: str, conditioning_set: set) -> str:
    """
    Classify the independence relationship between x and y (given conditioning_set)
    into a structural motif label. Used only for analysis metadata — never for
    question generation or answer computation.

    Motif detection works by checking whether (x, y) appear as the two "tested"
    nodes in a chain, fork, or v-structure pattern that is (or is not) blocked by
    the conditioning set.  Falls back to "other_marginal" / "other_conditional".
    """
    if not conditioning_set:
        # Direct edge
        if g.has_edge(x, y) or g.has_edge(y, x):
            return "direct_marginal"
        # V-structure: parents a,b of collider c; a,b are marginally independent
        for a, b, c in find_v_structures(g):
            if {a, b} == {x, y}:
                return "v_structure_marginal"
        # Chain endpoints: a→b→c; a,c are marginally dependent
        for a, b, c in find_chains(g):
            if {a, c} == {x, y}:
                return "chain_marginal"
        # Fork branches: a←c→b; a,b are marginally dependent (c is confounder)
        for a, b, c in find_forks(g):
            if {a, b} == {x, y}:
                return "fork_marginal"
        return "other_marginal"
    else:
        z_set = conditioning_set
        # Chain: a→b→c, conditioning on mediator b; a ⊥ c | b
        for a, b, c in find_chains(g):
            if {a, c} == {x, y} and b in z_set:
                return "chain_conditional"
        # Fork: a←c→b, conditioning on confounder c; a ⊥ b | c
        for a, b, c in find_forks(g):
            if {a, b} == {x, y} and c in z_set:
                return "fork_conditional"
        # V-structure: a→c←b, conditioning on collider c (or descendant); a ⊥̸ b | c
        for a, b, c in find_v_structures(g):
            if {a, b} == {x, y} and c in z_set:
                return "v_structure_conditional"
        return "other_conditional"


# --- Causal Structure templates ---

# Yes/No: Does A have a causal effect on B? (A is ancestor of B)
_CAUSAL_EFFECT_TEMPLATES = [
    "Does '{x}' have a causal effect on '{y}'?",
    "Does '{x}' cause '{y}'?",
    "Is '{y}' caused by '{x}'?",
    "Would intervening on '{x}' affect '{y}'?",
    "Is '{x}' a cause of '{y}' in this causal network?",
    "Can '{x}' causally influence '{y}'?",
]

# List: all ancestors of a node
_ALL_CAUSES_OF_TEMPLATES = [
    "What are all the causes of '{node}' in this causal network (direct and indirect)?",
    "Which variables causally affect '{node}' in this network?",
]

# List: all descendants of a node
_ALL_EFFECTS_OF_TEMPLATES = [
    "What does '{node}' causally affect in this network (directly or indirectly)?",
    "List all variables that '{node}' has a causal effect on.",
]


# ===================================================================
# Per-group generators  (answer-first for Yes/No groups)
# ===================================================================

def _gen_causal_structure(
    g: nx.DiGraph, rng: random.Random, n: int = 2,
) -> List[Dict[str, Any]]:
    """
    Group 1: Causal Structure (easy).
    Produces 1 Yes/No causal effect question + 1 list (all_causes_of or all_effects_of).
    Answer-first for the Yes/No to guarantee ~50/50 balance across worlds.
    """
    questions: List[Dict[str, Any]] = []
    nodes = list(g.nodes())

    # --- Yes/No question: causal effect (A is ancestor of B) ---
    answer_yes = rng.choice([True, False])
    tmpl = rng.choice(_CAUSAL_EFFECT_TEMPLATES)

    if answer_yes:
        # Find a node with at least one ancestor
        shuffled = list(nodes)
        rng.shuffle(shuffled)
        for x in shuffled:
            ancs = find_ancestors(g, x)
            if ancs:
                a = rng.choice(list(ancs))
                questions.append({
                    "question": tmpl.format(x=a, y=x),
                    "answer": "Yes",
                    "question_type": "causal_effect",
                    "question_group": "Causal Structure",
                    "difficulty": "easy",
                })
                break
    else:
        # Find a pair (A, B) where A is NOT an ancestor of B
        for _ in range(100):
            a, b = rng.sample(nodes, 2)
            if a not in find_ancestors(g, b):
                questions.append({
                    "question": tmpl.format(x=a, y=b),
                    "answer": "No",
                    "question_type": "causal_effect",
                    "question_group": "Causal Structure",
                    "difficulty": "easy",
                })
                break

    # --- List question: all_causes_of or all_effects_of ---
    list_subtype = rng.choice(["all_causes_of", "all_effects_of"])
    shuffled = list(nodes)
    rng.shuffle(shuffled)

    if list_subtype == "all_causes_of":
        for node in shuffled:
            ancs = find_ancestors(g, node)
            if ancs:  # prefer nodes with non-empty ancestor sets
                tmpl_list = rng.choice(_ALL_CAUSES_OF_TEMPLATES)
                questions.append({
                    "question": tmpl_list.format(node=node),
                    "answer": sorted(ancs),
                    "question_type": "all_causes_of",
                    "question_group": "Causal Structure",
                    "difficulty": "easy",
                })
                break
        else:
            # Fallback: any node (empty answer is valid)
            node = rng.choice(nodes)
            ancs = find_ancestors(g, node)
            tmpl_list = rng.choice(_ALL_CAUSES_OF_TEMPLATES)
            questions.append({
                "question": tmpl_list.format(node=node),
                "answer": sorted(ancs),
                "question_type": "all_causes_of",
                "question_group": "Causal Structure",
                "difficulty": "easy",
            })
    else:  # all_effects_of
        for node in shuffled:
            descs = find_descendants(g, node)
            if descs:  # prefer nodes with non-empty descendant sets
                tmpl_list = rng.choice(_ALL_EFFECTS_OF_TEMPLATES)
                questions.append({
                    "question": tmpl_list.format(node=node),
                    "answer": sorted(descs),
                    "question_type": "all_effects_of",
                    "question_group": "Causal Structure",
                    "difficulty": "easy",
                })
                break
        else:
            node = rng.choice(nodes)
            descs = find_descendants(g, node)
            tmpl_list = rng.choice(_ALL_EFFECTS_OF_TEMPLATES)
            questions.append({
                "question": tmpl_list.format(node=node),
                "answer": sorted(descs),
                "question_type": "all_effects_of",
                "question_group": "Causal Structure",
                "difficulty": "easy",
            })

    return questions[:n]


def _gen_marginal_independence(
    g: nx.DiGraph, rng: random.Random, n: int = 2,
) -> List[Dict[str, Any]]:
    """
    Group 2: Marginal Independence (medium).

    Sampling-driven approach:
      1. Randomly sample node pairs (no structure hunting).
      2. For each pair, compute d-separation with empty conditioning set.
      3. Randomly choose whether the question asks about independence or dependence.
      4. Golden answer: "Yes" if (d_sep == asks_independence) else "No".
         — This guarantees ~50/50 Yes/No balance regardless of graph structure,
           because P(answer="Yes") = P(d_sep=True)*0.5 + P(d_sep=False)*0.5 = 0.5.
      5. Build a pool of ~n*15+20 candidates, then select n//2 "Yes" and n-n//2 "No".
      6. Post-hoc: classify structural motif for analysis only (not used for generation).

    Mathematical basis for golden answer:
      X ⊥⊥ Y  ⟺  is_d_separated(g, X, Y, {})  [Global Markov + Faithfulness]
      Template asks "are X,Y independent?" → Yes iff d_sep=True.
      Template asks "are X,Y dependent?"   → Yes iff d_sep=False.
    """
    nodes = list(g.nodes())
    if len(nodes) < 2:
        return []

    candidates: List[Dict[str, Any]] = []
    seen: set = set()

    for _ in range(n * 15 + 20):
        x, y = rng.sample(nodes, 2)
        if (x, y) in seen:
            continue
        seen.add((x, y))

        try:
            d_sep = is_d_separated(g, x, y, set())
        except Exception:
            continue

        # Randomly choose question framing: independence or dependence
        asks_independence = rng.choice([True, False])
        answer = "Yes" if (d_sep == asks_independence) else "No"

        if asks_independence:
            tmpl = rng.choice(_MARGINAL_INDEP_TEMPLATES)
        else:
            tmpl = rng.choice(_MARGINAL_DEP_TEMPLATES)

        q_type = _classify_structure(g, x, y, set())
        candidates.append({
            "question": tmpl.format(x=x, y=y),
            "answer": answer,
            "question_type": q_type,
            "question_group": "Marginal Independence",
            "difficulty": "medium",
        })

    # Select balanced Yes/No
    n_yes = n // 2
    n_no = n - n_yes
    yes_pool = [c for c in candidates if c["answer"] == "Yes"]
    no_pool  = [c for c in candidates if c["answer"] == "No"]
    rng.shuffle(yes_pool)
    rng.shuffle(no_pool)

    questions = yes_pool[:n_yes] + no_pool[:n_no]
    return questions


def _gen_conditional_independence(
    g: nx.DiGraph, rng: random.Random, n: int = 2,
) -> List[Dict[str, Any]]:
    """
    Group 3: Conditional Independence (medium-hard).

    Sampling-driven approach:
      1. Randomly choose n_cond ∈ {1, 2} conditioning variables per question.
         Falls back to n_cond=1 if graph has fewer than 4 distinct nodes.
      2. Sample (2 + n_cond) distinct nodes; first 2 are (x, y), rest form Z.
      3. Compute d-separation: d_sep = is_d_separated(g, x, y, set(Z)).
      4. Randomly choose independence or dependence framing.
      5. Golden answer: "Yes" if (d_sep == asks_independence) else "No".
         Same 50/50 balance argument as Group 2 applies.
      6. Build a pool of ~n*15+20 candidates, pick n//2 "Yes" and n-n//2 "No".
      7. Post-hoc: classify structural motif for analysis only.

    Mathematical basis for golden answer:
      X ⊥⊥ Y | Z  ⟺  is_d_separated(g, X, Y, Z)  [Global Markov + Faithfulness]
      Template asks "independent given Z?" → Yes iff d_sep=True.
      Template asks "dependent given Z?"   → Yes iff d_sep=False.
    """
    nodes = list(g.nodes())
    if len(nodes) < 3:
        return []

    candidates: List[Dict[str, Any]] = []
    seen: set = set()

    for _ in range(n * 15 + 20):
        # Choose number of conditioning variables
        n_cond = rng.choice([1, 2]) if len(nodes) >= 4 else 1

        sampled = rng.sample(nodes, 2 + n_cond)
        x, y = sampled[0], sampled[1]
        z_list = sampled[2:]
        z_set = set(z_list)

        key = (x, y, frozenset(z_set))
        if key in seen:
            continue
        seen.add(key)

        try:
            d_sep = is_d_separated(g, x, y, z_set)
        except Exception:
            continue

        # Randomly choose question framing
        asks_independence = rng.choice([True, False])
        answer = "Yes" if (d_sep == asks_independence) else "No"

        if n_cond == 1:
            z = z_list[0]
            if asks_independence:
                tmpl = rng.choice(_COND1_INDEP_TEMPLATES)
                question = tmpl.format(x=x, y=y, z=z)
            else:
                tmpl = rng.choice(_COND1_DEP_TEMPLATES)
                question = tmpl.format(x=x, y=y, z=z)
        else:
            z1, z2 = z_list[0], z_list[1]
            if asks_independence:
                tmpl = rng.choice(_COND2_INDEP_TEMPLATES)
                question = tmpl.format(x=x, y=y, z1=z1, z2=z2)
            else:
                tmpl = rng.choice(_COND2_DEP_TEMPLATES)
                question = tmpl.format(x=x, y=y, z1=z1, z2=z2)

        q_type = _classify_structure(g, x, y, z_set)
        candidates.append({
            "question": question,
            "answer": answer,
            "question_type": q_type,
            "question_group": "Conditional Independence",
            "difficulty": "medium",
        })

    # Select balanced Yes/No
    n_yes = n // 2
    n_no = n - n_yes
    yes_pool = [c for c in candidates if c["answer"] == "Yes"]
    no_pool  = [c for c in candidates if c["answer"] == "No"]
    rng.shuffle(yes_pool)
    rng.shuffle(no_pool)

    questions = yes_pool[:n_yes] + no_pool[:n_no]
    return questions


# ===================================================================
# Main question orchestrator
# ===================================================================

def generate_all_questions(
    g: nx.DiGraph,
    n_per_group: int = 2,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Stratified, answer-first question generation.

    Produces exactly n_per_group questions for each of the 3 groups
    (total = 3 × n_per_group).  Yes/No answer balance is guaranteed
    by choosing the answer first and then finding a matching question.

    Groups:
      1. Causal Structure     (easy)        – causal_effect Yes/No + ancestor/descendant list
      2. Marginal Independence (medium)     – 1 Yes + 1 No
      3. Conditional Independence (medium-hard) – 1 Yes + 1 No
    """
    rng = random.Random(seed)

    questions: List[Dict[str, Any]] = []
    questions.extend(_gen_causal_structure(g, rng, n_per_group))
    questions.extend(_gen_marginal_independence(g, rng, n_per_group))
    questions.extend(_gen_conditional_independence(g, rng, n_per_group))

    # Assign sequential IDs
    for i, q in enumerate(questions):
        q["id"] = i

    return questions


# ===================================================================
# Modified generate_world (uses new question generator)
# ===================================================================

def generate_world(
    llm: QwenLLM,
    topic: str,
    n_nodes: int,
    seed: int,
    outdir: str,
    max_parents: int = 3,
    edge_multiplier: float = 1.5,
    n_per_group: int = 2,
) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)

    # 1) Variables (chunked, stable)
    var_specs = get_variables_chunked(llm, topic, n_nodes, chunk_size=10, max_tries=6)
    names = [v["name"] for v in var_specs]

    # 2) Edge candidates (chunked)
    # For very small graphs (<=5 nodes), randomly pick 2 or 3 edges to get structural variety;
    # otherwise use the standard edge_multiplier formula.
    if n_nodes <= 5:
        target_edges = rng.choice([2, 3])
    else:
        target_edges = int(edge_multiplier * n_nodes)
    candidates = get_edge_candidates_chunked(
        llm, topic, names,
        target_edges=target_edges,
        chunk_edges=45,
        max_rounds=8 if n_nodes >= 30 else 5,
        max_tries=4,
    )

    # 3) Build DAG with constraints
    dag = build_dag(
        names=names,
        edge_candidates=candidates,
        max_parents=max_parents,
        target_edges=target_edges,
        ensure_connected=True,
        seed=seed,
    )

    # 4) Build BN + CPDs (fully defined)
    model = build_bn_with_cpds(dag, var_specs, seed=seed + 999)

    # 4.5) Generate story
    story = generate_story(
        llm=llm,
        topic=topic,
        var_specs=var_specs,
        edges=[(u, v) for (u, v) in dag.edges()],
    )

    # 4.6) Identify non-intervenable variables
    non_intervenable = identify_non_intervenable_variables(
        llm=llm,
        topic=topic,
        story=story,
        var_specs=var_specs,
    )

    # 5) Generate questions — stratified, answer-first, causal framing
    questions = generate_all_questions(
        g=dag,
        n_per_group=n_per_group,
        seed=seed + 1234,
    )

    # 6) Save
    graph_path = os.path.join(outdir, f"graph_{topic.replace(' ','_')}_n{n_nodes}_seed{seed}.png")
    save_graph_png(list(model.edges()), graph_path, title=f"{topic} | n={n_nodes}")

    world = {
        "meta": {
            "topic": topic,
            "n_nodes": n_nodes,
            "seed": seed,
            "llm_model": llm.model_name,
            "max_parents": max_parents,
            "target_edges": target_edges,
            "graph_image_path": graph_path,
            "n_per_group": n_per_group,
            "n_questions": len(questions),
        },
        "story": story,
        "non_intervenable_variables": non_intervenable,
        "variables": var_specs,
        "edges": [(u, v) for (u, v) in model.edges()],
        "cpds": serialize_cpds(model),
        "questions": questions,
    }

    json_path = os.path.join(outdir, f"world_{topic.replace(' ','_')}_n{n_nodes}_seed{seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(world, f, ensure_ascii=False, indent=2)
    world["meta"]["json_path"] = json_path
    return world


# ===================================================================
# CLI
# ===================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=str, default=None)
    ap.add_argument("--n_nodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="./out_bn_causal")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_parents", type=int, default=3)
    ap.add_argument("--edge_mult", type=float, default=1.5)
    ap.add_argument("--n_per_group", type=int, default=2,
                    help="Number of questions per group (default 2 → 6 questions per world)")
    args = ap.parse_args()

    topic = args.topic or random.choice(TOPICS)
    llm = QwenLLM(model_name=args.model, do_sample=False, max_new_tokens=900)

    world = generate_world(
        llm=llm,
        topic=topic,
        n_nodes=args.n_nodes,
        seed=args.seed,
        outdir=args.outdir,
        max_parents=args.max_parents,
        edge_multiplier=args.edge_mult,
        n_per_group=args.n_per_group,
    )

    print("Saved:")
    print(" JSON:", world["meta"]["json_path"])
    print(" PNG :", world["meta"]["graph_image_path"])
    print(" edges:", len(world["edges"]))
    print(" questions:", world["meta"]["n_questions"])
    print(" story:", world["story"][:120], "...")
    non_interv_names = [v["name"] for v in world.get("non_intervenable_variables", [])]
    print(" non-intervenable:", len(non_interv_names), non_interv_names)
    if world["questions"]:
        type_counts = {}
        answer_counts = {"Yes": 0, "No": 0, "list": 0}
        for q in world["questions"]:
            qt = q["question_type"]
            type_counts[qt] = type_counts.get(qt, 0) + 1
            if isinstance(q["answer"], str):
                answer_counts[q["answer"]] = answer_counts.get(q["answer"], 0) + 1
            else:
                answer_counts["list"] += 1
        print(" question types:", type_counts)
        print(" answer balance:", answer_counts)


if __name__ == "__main__":
    main()
