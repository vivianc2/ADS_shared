# world_gen_stable_bn.py
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


TOPICS = [
    "Screening & diagnosis",
    "Treatment effectiveness",
    "Hospital data",
    "Education",
    "Social Science",
    "Labor & Policy",
    "User Behavior",
    "Criminal Justice",
]


# ----------------------------
# LLM wrapper (Qwen by default)
# ----------------------------
SYSTEM_JSON = (
    "You must output ONLY one JSON object and nothing else. "
    "No markdown, no commentary, no trailing text."
)

@dataclass
class QwenLLM:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: Any = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    max_new_tokens: int = 900
    do_sample: bool = False  # deterministic JSON is more stable

    # def __post_init__(self):
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.model_name,
    #         dtype=self.dtype,
    #         device_map="auto" if self.device == "cuda" else None,
    #     )
    #     self.model.eval()
    def __post_init__(self):
        # Resolve to local cache path (works in offline mode if already cached)
        local_dir = snapshot_download(self.model_name, local_files_only=True)

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            dtype=self.dtype,      # or torch.bfloat16/torch.float16
            device_map="auto",
        )
        self.model.eval()

    def chat(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ----------------------------
# Robust JSON extraction + repair
# ----------------------------
def extract_first_balanced_json(text: str) -> Dict[str, Any]:
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for i in starts:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    cand = text[i : j + 1]
                    return json.loads(cand)
    return json.loads(text)

def repair_to_valid_json(llm: QwenLLM, bad_text: str, schema_hint: str) -> Dict[str, Any]:
    prompt = f"""You are a JSON repair tool.
{SYSTEM_JSON}

Fix the following into VALID JSON that matches the schema.

Schema hint:
{schema_hint}

Bad text:
{bad_text}
"""
    fixed = llm.chat(SYSTEM_JSON, prompt)
    return extract_first_balanced_json(fixed)


# ----------------------------
# Variable generation (chunked)
# ----------------------------
def _sanitize_pascal(name: str) -> str:
    name = "".join(ch for ch in name if ch.isalnum())
    if not name:
        name = "Var"
    if not name[0].isalpha():
        name = "V" + name
    # Make sure Pascal-ish
    return name[0].upper() + name[1:]

def unique_name(name: str, used: set) -> str:
    base = _sanitize_pascal(name)
    name2 = base
    k = 1
    while name2 in used:
        k += 1
        name2 = f"{base}{k}"
    used.add(name2)
    return name2

def prompt_vars(topic: str, k: int, avoid: List[str]) -> str:
    return f"""{SYSTEM_JSON}

Generate EXACTLY {k} plausible discrete variables for topic "{topic}".
Design each variable as it would appear in REAL-WORLD data collection.
Avoid these names: {avoid}

Return JSON:
{{
  "topic": "{topic}",
  "variables": [
    {{"name":"PascalCase","values":["val1","val2","val3",...],"desc":"short description without double-quotes"}},
    ...
  ]
}}

Rules:
- EXACTLY {k} items.
- Each variable should have a REALISTIC number of values (2-30 depending on what makes sense):
  * Binary variables (2 values): Yes/No, Male/Female, Treated/Untreated
  * Ordinal variables (3-7 values): Low/Medium/High, severity levels, age groups
  * Categorical variables (3-20+ values): medication types, diagnoses, meal sizes (e.g., 0g, 5g, 10g, ..., 100g)
  * Numeric discretized (5-30 values): glucose levels, blood pressure ranges, dosage amounts
- Values should be distinct, ordered meaningfully when applicable, and use realistic units/labels.
- Make them realistic for the topic (like what appears in real data).
- desc should be 1 sentence, do NOT include any double quotes.
""".strip()

def get_variables_chunked(
    llm: QwenLLM,
    topic: str,
    n_nodes: int,
    chunk_size: int = 10,
    max_tries: int = 6,
) -> List[Dict[str, Any]]:
    used = set()
    out: List[Dict[str, Any]] = []
    remaining = n_nodes

    while remaining > 0:
        k = min(chunk_size, remaining)
        avoid = sorted(list(used))
        schema_hint = f'keys: "topic" (string), "variables" (list length {k}), each item has name, values[2], desc'
        p = prompt_vars(topic, k, avoid)

        last_err: Optional[Exception] = None
        for _ in range(max_tries):
            raw = llm.chat(SYSTEM_JSON, p)
            try:
                js = extract_first_balanced_json(raw)
            except Exception as e:
                last_err = e
                try:
                    js = repair_to_valid_json(llm, raw, schema_hint)
                except Exception as e2:
                    last_err = e2
                    continue

            try:
                vars_list = js["variables"]
                if len(vars_list) < k:
                    raise ValueError(f"expected {k}, got {len(vars_list)}")
                # Tolerate the LLM returning a few extra items — just trim.
                if len(vars_list) > k:
                    vars_list = vars_list[:k]

                for v in vars_list:
                    name = unique_name(str(v["name"]), used)
                    vals = list(v["values"])
                    # Ensure at least 2 distinct values, max 30
                    if len(vals) < 2:
                        vals = [str(vals[0]) if vals else "Val0", "Val1"]
                    elif len(vals) > 30:
                        vals = vals[:30]
                    # Ensure all values are distinct
                    seen_vals = set()
                    unique_vals = []
                    for val in vals:
                        val_str = str(val)
                        if val_str not in seen_vals:
                            seen_vals.add(val_str)
                            unique_vals.append(val_str)
                    vals = unique_vals if len(unique_vals) >= 2 else unique_vals + ["Other"]
                    desc = str(v.get("desc", "")).replace('"', "'")
                    out.append({"name": name, "values": vals, "desc": desc})
                remaining -= k
                break
            except Exception as e:
                last_err = e
                continue
        else:
            raise RuntimeError(f"Failed to get variables chunk. Last error: {last_err}")

    return out


# ----------------------------
# Edge generation (LLM proposes, code enforces)
# ----------------------------
def prompt_edge_candidates(topic: str, names: List[str], m: int) -> str:
    names_str = ", ".join(names)
    return f"""{SYSTEM_JSON}

We are building a causal Bayesian network for topic "{topic}".
Variables:
[{names_str}]

Propose up to {m} DIRECTED edges (cause -> effect) that are plausible in the real world.
Keep it sparse and realistic.

Return JSON:
{{"edges":[["U","V"],["A","B"], ...]}}

Rules:
- U and V must be from the given list.
- No self-loops.
- Do not include duplicates.
- Prefer stable causal directions (avoid cycles conceptually).
""".strip()

def get_edge_candidates_chunked(
    llm: QwenLLM,
    topic: str,
    names: List[str],
    target_edges: int,
    chunk_edges: int = 40,
    max_rounds: int = 6,
    max_tries: int = 4,
) -> List[Tuple[str, str]]:
    """
    Ask for edges in multiple rounds to reduce failure rate and keep outputs short.
    """
    all_edges: List[Tuple[str, str]] = []
    seen = set()
    schema_hint = 'keys: "edges": list of [U,V] pairs'

    rounds = max_rounds
    for _ in range(rounds):
        if len(all_edges) >= target_edges * 2:
            break
        m = min(chunk_edges, max(10, target_edges - len(all_edges)))
        p = prompt_edge_candidates(topic, names, m)

        last_err: Optional[Exception] = None
        for _t in range(max_tries):
            raw = llm.chat(SYSTEM_JSON, p)
            try:
                js = extract_first_balanced_json(raw)
            except Exception as e:
                last_err = e
                try:
                    js = repair_to_valid_json(llm, raw, schema_hint)
                except Exception as e2:
                    last_err = e2
                    continue

            try:
                edges = js.get("edges", [])
                for pair in edges:
                    if not isinstance(pair, list) or len(pair) != 2:
                        continue
                    u, v = str(pair[0]), str(pair[1])
                    if u == v:
                        continue
                    if u not in names or v not in names:
                        continue
                    if (u, v) in seen:
                        continue
                    seen.add((u, v))
                    all_edges.append((u, v))
                break
            except Exception as e:
                last_err = e
                continue

    # If LLM gives too few, fall back to random plausible edges (still enforced later)
    if len(all_edges) < target_edges:
        rng = random.Random(0)
        for _ in range(target_edges * 10):
            u, v = rng.sample(names, 2)
            if u != v and (u, v) not in seen:
                seen.add((u, v))
                all_edges.append((u, v))
            if len(all_edges) >= target_edges:
                break

    return all_edges


# ----------------------------
# DAG construction with constraints
# ----------------------------
def add_edge_if_acyclic(g: nx.DiGraph, u: str, v: str) -> bool:
    if u == v or g.has_edge(u, v):
        return False
    g.add_edge(u, v)
    if not nx.is_directed_acyclic_graph(g):
        g.remove_edge(u, v)
        return False
    return True

def build_dag(
    names: List[str],
    edge_candidates: List[Tuple[str, str]],
    max_parents: int = 3,
    target_edges: Optional[int] = None,
    ensure_connected: bool = True,
    seed: int = 0,
) -> nx.DiGraph:
    rng = random.Random(seed)
    g = nx.DiGraph()
    g.add_nodes_from(names)

    if target_edges is None:
        # sparse by default: ~1.5*n
        target_edges = int(1.5 * len(names))

    # 1) Add candidate edges greedily under constraints
    rng.shuffle(edge_candidates)
    for (u, v) in edge_candidates:
        if g.in_degree(v) >= max_parents:
            continue
        added = add_edge_if_acyclic(g, u, v)
        if added and g.number_of_edges() >= target_edges:
            break

    # 2) If too sparse, add random edges (still constrained)
    tries = 0
    while g.number_of_edges() < target_edges and tries < target_edges * 50:
        u, v = rng.sample(names, 2)
        tries += 1
        if g.in_degree(v) >= max_parents:
            continue
        add_edge_if_acyclic(g, u, v)

    # 3) Avoid isolates / disconnected components (undirected connectivity)
    if ensure_connected:
        und = g.to_undirected()
        comps = list(nx.connected_components(und))
        # Connect components by adding edges from one comp to another (respect acyclicity)
        # We'll use topological order bias: connect from earlier to later by name order.
        if len(comps) > 1:
            comps = sorted(comps, key=lambda s: -len(s))
            base = list(comps[0])
            for comp in comps[1:]:
                a = rng.choice(base)
                b = rng.choice(list(comp))
                # try both directions to keep acyclic
                if not add_edge_if_acyclic(g, a, b):
                    add_edge_if_acyclic(g, b, a)
                base.extend(list(comp))

        # also discourage totally detached nodes (degree 0)
        for n in names:
            if g.in_degree(n) == 0 and g.out_degree(n) == 0:
                # connect it to something
                t = rng.choice([x for x in names if x != n])
                if not add_edge_if_acyclic(g, t, n):
                    add_edge_if_acyclic(g, n, t)

    return g


# ----------------------------
# CPD generation (fully defined probabilities)
# ----------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))

def clamp01(p: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return float(np.clip(p, lo, hi))

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def make_logistic_cpd(
    child: str,
    parents: List[str],
    rng: random.Random,
    state_names: Dict[str, List[str]],
    var_specs: List[Dict[str, Any]],
    base_rate_range: Tuple[float, float] = (0.15, 0.65),
    weight_range: Tuple[float, float] = (-1.2, 1.2),
) -> TabularCPD:
    """
    Multi-valued CPD using softmax regression:
    P(child=k | parents) = softmax(W[k] @ parent_features)[k]

    For binary: reduces to logistic regression
    For K>2: uses multinomial logistic (softmax) regression
    """
    # Get cardinality of child
    child_states = state_names[child]
    K = len(child_states)

    if not parents:
        # No parents: sample from Dirichlet to get random probabilities
        alpha = np.ones(K) * 2.0  # symmetric Dirichlet
        probs = rng.random() * np.ones(K)
        probs = probs / probs.sum()
        # Use Dirichlet-like sampling
        probs = np.random.dirichlet(alpha)
        vals = [[float(p)] for p in probs]
        return TabularCPD(child, K, vals, state_names=state_names)

    # Get parent cardinalities
    parent_cards = [len(state_names[p]) for p in parents]

    # Number of parent configurations
    n_configs = int(np.prod(parent_cards))

    # For each parent configuration, generate K probabilities using softmax
    # Create weight matrix: K x (1 + n_parents) for bias + parent effects
    # Simplified: each parent contributes additively to each class's logit

    # Generate base logits for each class (bias terms)
    base_logits = np.array([rng.uniform(-1.5, 1.5) for _ in range(K)])

    # Generate weight matrix: K x n_parents
    weights = np.array([[rng.uniform(*weight_range) for _ in parents] for _ in range(K)])

    # Generate probability for each parent configuration
    all_probs = []
    for config_idx in range(n_configs):
        # Decode configuration index to parent values
        parent_vals = []
        temp = config_idx
        for card in reversed(parent_cards):
            parent_vals.append(temp % card)
            temp //= card
        parent_vals = list(reversed(parent_vals))

        # Compute logits for this configuration
        logits = base_logits.copy()
        for k in range(K):
            for p_idx, p_val in enumerate(parent_vals):
                logits[k] += weights[k][p_idx] * p_val

        # Apply softmax to get probabilities
        probs = softmax(logits)
        all_probs.append(probs)

    # Convert to CPD format: values[i][j] = P(child=i | parents=config_j)
    values = [[float(all_probs[j][i]) for j in range(n_configs)] for i in range(K)]

    return TabularCPD(
        child, K, values,
        evidence=parents,
        evidence_card=parent_cards,
        state_names=state_names
    )

def build_bn_with_cpds(
    g: nx.DiGraph,
    var_specs: List[Dict[str, Any]],
    seed: int,
) -> DiscreteBayesianNetwork:
    rng = random.Random(seed)

    # state_names makes forward_sample return strings (stable + readable)
    state_names: Dict[str, List[str]] = {
        v["name"]: list(v["values"]) for v in var_specs
    }

    model = DiscreteBayesianNetwork(list(g.edges()))
    model.add_nodes_from(list(g.nodes()))

    cpds: List[TabularCPD] = []
    topo = list(nx.topological_sort(g))
    for node in topo:
        parents = list(g.predecessors(node))
        # keep probabilities plausible by keeping weights moderate
        cpd = make_logistic_cpd(
            child=node,
            parents=parents,
            rng=rng,
            state_names=state_names,
            var_specs=var_specs,
            base_rate_range=(0.12, 0.7),
            weight_range=(-1.0, 1.0),
        )
        cpds.append(cpd)

    model.add_cpds(*cpds)

    # This is the main correctness check.
    if not model.check_model():
        raise RuntimeError("model.check_model() failed (CPDs/graph mismatch).")

    return model


# ----------------------------
# Question Generation
# ----------------------------

# Difficulty levels
DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"


def find_v_structures(g: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """
    Find all v-structures (colliders): A -> C <- B where A and B are NOT adjacent.
    Returns list of (A, B, C) tuples where C is the collider.
    """
    v_structs = []
    for node in g.nodes():
        parents = list(g.predecessors(node))
        if len(parents) >= 2:
            for i, p1 in enumerate(parents):
                for p2 in parents[i + 1:]:
                    # Check if p1 and p2 are NOT adjacent (no edge in either direction)
                    if not g.has_edge(p1, p2) and not g.has_edge(p2, p1):
                        v_structs.append((p1, p2, node))
    return v_structs


def find_chains(g: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """
    Find chains: A -> B -> C where there is NO direct edge A -> C.
    Returns list of (A, B, C) tuples where B is the mediator.
    """
    chains = []
    for b in g.nodes():
        parents = list(g.predecessors(b))
        children = list(g.successors(b))
        for a in parents:
            for c in children:
                if not g.has_edge(a, c):  # Pure mediator (no direct effect)
                    chains.append((a, b, c))
    return chains


def find_forks(g: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """
    Find forks (confounders): A <- C -> B where C is common cause.
    Returns list of (A, B, C) tuples where C is the confounder.
    """
    forks = []
    for c in g.nodes():
        children = list(g.successors(c))
        if len(children) >= 2:
            for i, a in enumerate(children):
                for b in children[i + 1:]:
                    # Check that a and b are not directly connected
                    if not g.has_edge(a, b) and not g.has_edge(b, a):
                        forks.append((a, b, c))
    return forks


def get_markov_blanket(g: nx.DiGraph, node: str) -> set:
    """
    Get the Markov blanket of a node: parents + children + co-parents (parents of children).
    """
    parents = set(g.predecessors(node))
    children = set(g.successors(node))
    co_parents = set()
    for child in children:
        co_parents.update(g.predecessors(child))
    co_parents.discard(node)
    return parents | children | co_parents


def find_root_nodes(g: nx.DiGraph) -> List[str]:
    """Find nodes with no parents (root causes)."""
    return [n for n in g.nodes() if g.in_degree(n) == 0]


def find_leaf_nodes(g: nx.DiGraph) -> List[str]:
    """Find nodes with no children (terminal effects)."""
    return [n for n in g.nodes() if g.out_degree(n) == 0]


def is_d_separated(g: nx.DiGraph, x: str, y: str, z: set) -> bool:
    """
    Check if X and Y are d-separated given Z.
    Returns True if X _||_ Y | Z (conditionally independent).
    """
    # Use networkx's is_d_separator (d_separated is deprecated in v3.5+)
    try:
        return nx.is_d_separator(g, {x}, {y}, z)
    except AttributeError:
        # Fallback for older networkx versions
        return nx.d_separated(g, {x}, {y}, z)


def find_ancestors(g: nx.DiGraph, node: str) -> set:
    """Find all ancestors of a node."""
    return nx.ancestors(g, node)


def find_descendants(g: nx.DiGraph, node: str) -> set:
    """Find all descendants of a node."""
    return nx.descendants(g, node)


# ----------------------------
# Question Generators by Type
# ----------------------------

def _find_marginal_independent_pairs(g: nx.DiGraph, rng: random.Random, n: int) -> List[Tuple[str, str]]:
    """Find node pairs that ARE marginally d-separated (independent). Used for balancing."""
    nodes = list(g.nodes())
    pairs = []
    seen = set()
    for _ in range(n * 20):
        if len(pairs) >= n:
            break
        x, y = rng.sample(nodes, 2)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        try:
            if is_d_separated(g, x, y, set()):
                pairs.append((x, y))
        except:
            continue
    return pairs


def _find_cond_independent_triples(g: nx.DiGraph, rng: random.Random, n: int) -> List[Tuple[str, str, str]]:
    """Find (X, Y, Z) triples where X _||_ Y | Z. Used for balancing."""
    nodes = list(g.nodes())
    triples = []
    seen = set()
    for _ in range(n * 20):
        if len(triples) >= n:
            break
        if len(nodes) < 3:
            break
        x, y = rng.sample(nodes, 2)
        remaining = [nd for nd in nodes if nd not in {x, y}]
        if not remaining:
            continue
        z = rng.choice(remaining)
        key = (x, y, z)
        if key in seen:
            continue
        seen.add(key)
        try:
            if is_d_separated(g, x, y, {z}):
                triples.append((x, y, z))
        except:
            continue
    return triples


def generate_v_structure_questions(g: nx.DiGraph, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Generate questions about v-structures (colliders).
    Difficulty: MEDIUM

    v_structure_marginal: naturally balanced (other paths may or may not connect parents).
    v_structure_conditional: conditioning on collider always opens the path (always "No"),
      so we balance by also generating "Yes" questions from d-separated triples.
    """
    questions = []
    v_structs = find_v_structures(g)

    no_cond_count = 0  # track how many "No" conditional questions we generate

    for a, b, c in v_structs:
        # --- v_structure_marginal (naturally balanced) ---
        try:
            marginally_independent = is_d_separated(g, a, b, set())
        except:
            continue

        if marginally_independent:
            expl = (f"'{a}' and '{b}' form a v-structure at collider '{c}' (i.e., {a} -> {c} <- {b}). "
                    f"The collider blocks this path, and there are no other unblocked paths between them, so they are marginally independent.")
        else:
            expl = (f"'{a}' and '{b}' form a v-structure at collider '{c}' (i.e., {a} -> {c} <- {b}). "
                    f"Although the collider blocks this particular path, other unblocked paths exist in the graph, making them marginally dependent.")
        questions.append({
            "question": f"In this Bayesian network, are '{a}' and '{b}' marginally independent (without conditioning on anything)?",
            "answer": "Yes" if marginally_independent else "No",
            "explanation": expl,
            "difficulty": DIFFICULTY_MEDIUM,
            "question_type": "v_structure_marginal",
            "relevant_variables": [a, b, c],
        })

        # --- v_structure_conditional (inherently "No" for collider) ---
        try:
            cond_independent_given_c = is_d_separated(g, a, b, {c})
        except:
            continue

        if not cond_independent_given_c:
            expl2 = (f"'{c}' is a collider between '{a}' and '{b}'. Conditioning on a collider OPENS the path, "
                     f"making '{a}' and '{b}' dependent. This is known as the 'explaining away' effect.")
        else:
            expl2 = (f"'{c}' is a collider between '{a}' and '{b}'. Conditioning on '{c}' opens this v-structure path, "
                     f"but all paths between '{a}' and '{b}' are still blocked given '{c}', so they remain conditionally independent.")
        questions.append({
            "question": f"In this Bayesian network, are '{a}' and '{b}' conditionally independent given '{c}'?",
            "answer": "Yes" if cond_independent_given_c else "No",
            "explanation": expl2,
            "difficulty": DIFFICULTY_MEDIUM,
            "question_type": "v_structure_conditional",
            "relevant_variables": [a, b, c],
        })
        if not cond_independent_given_c:
            no_cond_count += 1

    # Balance v_structure_conditional: add "Yes" questions from d-separated triples
    if no_cond_count > 0:
        yes_triples = _find_cond_independent_triples(g, rng, no_cond_count)
        for x, y, z in yes_triples:
            questions.append({
                "question": f"In this Bayesian network, are '{x}' and '{y}' conditionally independent given '{z}'?",
                "answer": "Yes",
                "explanation": (f"Using d-separation: all paths between '{x}' and '{y}' are blocked when "
                                f"conditioning on '{z}', so they are conditionally independent."),
                "difficulty": DIFFICULTY_MEDIUM,
                "question_type": "v_structure_conditional",
                "relevant_variables": [x, y, z],
            })

    return questions


def generate_chain_questions(g: nx.DiGraph, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Generate questions about chains (mediators).
    Difficulty: MEDIUM

    chain_marginal: a chain A->B->C always has an open path, so marginal independence
      is always "No". We balance by adding "Yes" questions from d-separated pairs.
    chain_conditional: naturally balanced (other paths may keep A,C dependent given B).
    """
    questions = []
    chains = find_chains(g)

    no_marginal_count = 0

    for a, b, c in chains:
        # --- chain_marginal (inherently "No" for chain endpoints) ---
        try:
            marginally_independent = is_d_separated(g, a, c, set())
        except:
            continue

        if not marginally_independent:
            expl = (f"There is a chain {a} -> {b} -> {c}. The path through mediator '{b}' is open "
                    f"when not conditioning, making '{a}' and '{c}' marginally dependent.")
        else:
            expl = (f"There is a chain {a} -> {b} -> {c}, but all paths between '{a}' and '{c}' "
                    f"are blocked in the full graph (e.g., by colliders), so they are marginally independent.")
        questions.append({
            "question": f"In this Bayesian network, are '{a}' and '{c}' marginally independent?",
            "answer": "Yes" if marginally_independent else "No",
            "explanation": expl,
            "difficulty": DIFFICULTY_MEDIUM,
            "question_type": "chain_marginal",
            "relevant_variables": [a, b, c],
        })
        if not marginally_independent:
            no_marginal_count += 1

        # --- chain_conditional (naturally balanced) ---
        try:
            cond_independent_given_b = is_d_separated(g, a, c, {b})
        except:
            continue

        if cond_independent_given_b:
            expl2 = (f"'{b}' is a mediator on the chain from '{a}' to '{c}'. Conditioning on a mediator BLOCKS "
                     f"the path, making '{a}' and '{c}' conditionally independent given '{b}'.")
        else:
            expl2 = (f"'{b}' is a mediator on the chain from '{a}' to '{c}'. Conditioning on '{b}' blocks this chain path, "
                     f"but other open paths exist between '{a}' and '{c}' given '{b}', so they remain conditionally dependent.")
        questions.append({
            "question": f"In this Bayesian network, are '{a}' and '{c}' conditionally independent given '{b}'?",
            "answer": "Yes" if cond_independent_given_b else "No",
            "explanation": expl2,
            "difficulty": DIFFICULTY_MEDIUM,
            "question_type": "chain_conditional",
            "relevant_variables": [a, b, c],
        })

    # Balance chain_marginal: add "Yes" questions from d-separated pairs
    if no_marginal_count > 0:
        yes_pairs = _find_marginal_independent_pairs(g, rng, no_marginal_count)
        for x, y in yes_pairs:
            questions.append({
                "question": f"In this Bayesian network, are '{x}' and '{y}' marginally independent?",
                "answer": "Yes",
                "explanation": (f"Using d-separation: all paths between '{x}' and '{y}' are blocked "
                                f"(no open non-collider paths), so they are marginally independent."),
                "difficulty": DIFFICULTY_MEDIUM,
                "question_type": "chain_marginal",
                "relevant_variables": [x, y],
            })

    return questions


def generate_fork_questions(g: nx.DiGraph, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Generate questions about forks (confounders).
    Difficulty: MEDIUM

    fork_marginal: a fork A<-C->B always has an open path, so marginal independence
      is always "No". We balance by adding "Yes" questions from d-separated pairs.
    fork_conditional: naturally balanced (other paths may keep A,B dependent given C).
    """
    questions = []
    forks = find_forks(g)

    no_marginal_count = 0

    for a, b, c in forks:
        # --- fork_marginal (inherently "No" for fork children) ---
        try:
            marginally_independent = is_d_separated(g, a, b, set())
        except:
            continue

        if not marginally_independent:
            expl = (f"'{c}' is a common cause (confounder) of both '{a}' and '{b}' in the fork "
                    f"{a} <- {c} -> {b}. The path is open, creating dependence between them.")
        else:
            expl = (f"'{c}' is a common cause in the fork {a} <- {c} -> {b}, but all paths between "
                    f"'{a}' and '{b}' are blocked in the full graph, so they are marginally independent.")
        questions.append({
            "question": f"In this Bayesian network, are '{a}' and '{b}' marginally independent?",
            "answer": "Yes" if marginally_independent else "No",
            "explanation": expl,
            "difficulty": DIFFICULTY_MEDIUM,
            "question_type": "fork_marginal",
            "relevant_variables": [a, b, c],
        })
        if not marginally_independent:
            no_marginal_count += 1

        # --- fork_conditional (naturally balanced) ---
        try:
            cond_independent_given_c = is_d_separated(g, a, b, {c})
        except:
            continue

        if cond_independent_given_c:
            expl2 = (f"Conditioning on the common cause '{c}' blocks the fork path "
                     f"{a} <- {c} -> {b}, making '{a}' and '{b}' conditionally independent.")
        else:
            expl2 = (f"Conditioning on '{c}' blocks the fork path {a} <- {c} -> {b}, but other open paths "
                     f"exist between '{a}' and '{b}' given '{c}', so they remain conditionally dependent.")
        questions.append({
            "question": f"In this Bayesian network, are '{a}' and '{b}' conditionally independent given '{c}'?",
            "answer": "Yes" if cond_independent_given_c else "No",
            "explanation": expl2,
            "difficulty": DIFFICULTY_MEDIUM,
            "question_type": "fork_conditional",
            "relevant_variables": [a, b, c],
        })

    # Balance fork_marginal: add "Yes" questions from d-separated pairs
    if no_marginal_count > 0:
        yes_pairs = _find_marginal_independent_pairs(g, rng, no_marginal_count)
        for x, y in yes_pairs:
            questions.append({
                "question": f"In this Bayesian network, are '{x}' and '{y}' marginally independent?",
                "answer": "Yes",
                "explanation": (f"Using d-separation: all paths between '{x}' and '{y}' are blocked "
                                f"(no open non-collider paths), so they are marginally independent."),
                "difficulty": DIFFICULTY_MEDIUM,
                "question_type": "fork_marginal",
                "relevant_variables": [x, y],
            })

    return questions


def generate_markov_blanket_questions(g: nx.DiGraph, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Generate questions about Markov blankets.
    Difficulty: HARD (requires understanding conditional independence structure)
    """
    questions = []
    nodes = list(g.nodes())

    for node in nodes:
        mb = get_markov_blanket(g, node)
        if len(mb) >= 1:  # Only generate if non-trivial blanket
            mb_list = sorted(list(mb))
            other_nodes = [n for n in nodes if n != node and n not in mb]

            if other_nodes:
                # Question about what's in the Markov blanket
                q1 = {
                    "question": f"What is the Markov blanket of '{node}'? (List all variables that make '{node}' conditionally independent of all other variables when conditioned upon)",
                    "answer": mb_list,
                    "explanation": f"The Markov blanket consists of parents, children, and co-parents (other parents of children). For '{node}': {mb_list}",
                    "difficulty": DIFFICULTY_HARD,
                    "question_type": "markov_blanket",
                    "relevant_variables": [node] + mb_list,
                }
                questions.append(q1)

    return questions


def generate_structural_questions(g: nx.DiGraph, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Generate basic structural questions.
    Difficulty: EASY (just reading the graph)

    Questions: root_nodes, leaf_nodes, direct_edge
    """
    questions = []
    nodes = list(g.nodes())

    # Root nodes question
    roots = find_root_nodes(g)
    if roots:
        q1 = {
            "question": "Which variables have no parents (root causes) in this Bayesian network?",
            "answer": sorted(roots),
            "explanation": f"Root nodes are variables with in-degree 0. They represent exogenous variables or ultimate causes.",
            "difficulty": DIFFICULTY_EASY,
            "question_type": "root_nodes",
            "relevant_variables": roots,
        }
        questions.append(q1)

    # Leaf nodes question
    leaves = find_leaf_nodes(g)
    if leaves:
        q2 = {
            "question": "Which variables have no children (terminal effects) in this Bayesian network?",
            "answer": sorted(leaves),
            "explanation": f"Leaf nodes are variables with out-degree 0. They represent final outcomes that don't cause other measured variables.",
            "difficulty": DIFFICULTY_EASY,
            "question_type": "leaf_nodes",
            "relevant_variables": leaves,
        }
        questions.append(q2)

    # Direct edge questions - balanced: half from actual edges, half from non-edges
    edge_list = list(g.edges())
    n_edge_qs = min(5, len(nodes))
    n_yes = n_edge_qs // 2
    n_no = n_edge_qs - n_yes

    # Sample actual edges (answer = Yes)
    if edge_list and n_yes > 0:
        sampled_edges = rng.sample(edge_list, min(n_yes, len(edge_list)))
        for n1, n2 in sampled_edges:
            q3 = {
                "question": f"Is there a direct causal edge from '{n1}' to '{n2}' in this Bayesian network?",
                "answer": "Yes",
                "explanation": f"There is a direct causal edge from '{n1}' to '{n2}' in the graph.",
                "difficulty": DIFFICULTY_EASY,
                "question_type": "direct_edge",
                "relevant_variables": [n1, n2],
            }
            questions.append(q3)

    # Sample non-edges (answer = No)
    edge_set = set(g.edges())
    no_count = 0
    for _ in range(n_no * 10):
        if no_count >= n_no:
            break
        n1, n2 = rng.sample(nodes, 2)
        if (n1, n2) not in edge_set:
            q3 = {
                "question": f"Is there a direct causal edge from '{n1}' to '{n2}' in this Bayesian network?",
                "answer": "No",
                "explanation": f"There is no direct causal edge from '{n1}' to '{n2}' in the graph.",
                "difficulty": DIFFICULTY_EASY,
                "question_type": "direct_edge",
                "relevant_variables": [n1, n2],
            }
            questions.append(q3)
            no_count += 1

    return questions


def generate_ancestor_descendant_questions(g: nx.DiGraph, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Generate questions about ancestors and descendants.
    Difficulty: EASY to MEDIUM
    """
    questions = []
    nodes = list(g.nodes())

    # Ancestor questions
    for node in rng.sample(nodes, min(3, len(nodes))):
        ancestors = find_ancestors(g, node)
        if ancestors:
            q1 = {
                "question": f"List all ancestors (direct and indirect causes) of '{node}'.",
                "answer": sorted(list(ancestors)),
                "explanation": f"Ancestors are all nodes that have a directed path to '{node}'.",
                "difficulty": DIFFICULTY_EASY,
                "question_type": "ancestors",
                "relevant_variables": [node] + list(ancestors),
            }
            questions.append(q1)

    # Descendant questions
    for node in rng.sample(nodes, min(3, len(nodes))):
        descendants = find_descendants(g, node)
        if descendants:
            q2 = {
                "question": f"List all descendants (direct and indirect effects) of '{node}'.",
                "answer": sorted(list(descendants)),
                "explanation": f"Descendants are all nodes reachable via directed paths from '{node}'.",
                "difficulty": DIFFICULTY_EASY,
                "question_type": "descendants",
                "relevant_variables": [node] + list(descendants),
            }
            questions.append(q2)

    # Is-ancestor questions
    for _ in range(min(5, len(nodes))):
        n1, n2 = rng.sample(nodes, 2)
        is_anc = n1 in find_ancestors(g, n2)
        q3 = {
            "question": f"Is '{n1}' an ancestor of '{n2}'?",
            "answer": "Yes" if is_anc else "No",
            "explanation": f"{'There exists' if is_anc else 'There is no'} directed path from '{n1}' to '{n2}'.",
            "difficulty": DIFFICULTY_EASY,
            "question_type": "is_ancestor",
            "relevant_variables": [n1, n2],
        }
        questions.append(q3)

    return questions


def generate_marginal_independence_questions(g: nx.DiGraph, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Generate general marginal independence questions (not tied to specific structures).
    Difficulty: MEDIUM (requires understanding path analysis)

    Uses d-separation with empty conditioning set to verify answers.
    """
    questions = []
    nodes = list(g.nodes())

    if len(nodes) < 2:
        return questions

    # Generate random marginal independence queries
    for _ in range(min(10, len(nodes) * 2)):
        x, y = rng.sample(nodes, 2)

        # Compute marginal independence using d-separation with empty set
        try:
            marginally_independent = is_d_separated(g, x, y, set())
        except:
            continue

        q = {
            "question": f"Are '{x}' and '{y}' marginally independent (without conditioning on any variables)?",
            "answer": "Yes" if marginally_independent else "No",
            "explanation": f"Using d-separation with empty conditioning set: '{x}' and '{y}' are {'d-separated (marginally independent)' if marginally_independent else 'NOT d-separated (marginally dependent)'} - all paths between them are {'blocked' if marginally_independent else 'open or have at least one open path'}.",
            "difficulty": DIFFICULTY_MEDIUM,
            "question_type": "marginal_independence",
            "relevant_variables": [x, y],
        }
        questions.append(q)

    return questions


def generate_d_separation_questions(g: nx.DiGraph, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Generate general d-separation questions with non-trivial conditioning sets.
    Difficulty: HARD (requires understanding d-separation rules)

    Note: We always require a non-empty conditioning set (1-3 nodes).
    Marginal independence questions are handled by generate_marginal_independence_questions.
    """
    questions = []
    nodes = list(g.nodes())

    if len(nodes) < 4:  # Need at least 4 nodes for meaningful conditioning
        return questions

    # Generate random d-separation queries with non-empty conditioning sets
    for _ in range(min(15, len(nodes) * 2)):
        # Pick two nodes X and Y
        x, y = rng.sample(nodes, 2)

        # Pick a conditioning set Z (1-3 nodes, excluding X and Y) - NEVER empty
        remaining = [n for n in nodes if n not in {x, y}]
        if len(remaining) < 1:
            continue

        z_size = rng.randint(1, min(3, len(remaining)))  # At least 1 node
        z = set(rng.sample(remaining, z_size))

        # Compute d-separation
        try:
            d_sep = is_d_separated(g, x, y, z)
        except:
            continue

        z_str = ", ".join(f"'{v}'" for v in sorted(z))
        q = {
            "question": f"Are '{x}' and '{y}' conditionally independent given {{{z_str}}}?",
            "answer": "Yes" if d_sep else "No",
            "explanation": f"Using d-separation: '{x}' and '{y}' are {'d-separated (conditionally independent)' if d_sep else 'NOT d-separated (conditionally dependent)'} given {{{z_str}}}.",
            "difficulty": DIFFICULTY_HARD,
            "question_type": "d_separation",
            "relevant_variables": [x, y] + sorted(list(z)),
        }
        questions.append(q)

    return questions


def generate_all_questions(
    g: nx.DiGraph,
    n_questions: int,
    difficulty: Optional[str] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate questions for a Bayesian network.

    Args:
        g: The DAG
        n_questions: Target number of questions
        difficulty: Filter by difficulty level (easy/medium/hard) or None for all
        seed: Random seed for reproducibility

    Returns:
        List of question dictionaries
    """
    rng = random.Random(seed)

    # Generate all questions by type
    all_questions = []

    # Easy questions
    all_questions.extend(generate_structural_questions(g, rng))
    all_questions.extend(generate_ancestor_descendant_questions(g, rng))

    # Medium questions
    all_questions.extend(generate_v_structure_questions(g, rng))
    all_questions.extend(generate_chain_questions(g, rng))
    all_questions.extend(generate_fork_questions(g, rng))
    all_questions.extend(generate_marginal_independence_questions(g, rng))

    # Hard questions
    all_questions.extend(generate_markov_blanket_questions(g, rng))
    all_questions.extend(generate_d_separation_questions(g, rng))

    # Filter by difficulty if specified
    if difficulty:
        all_questions = [q for q in all_questions if q["difficulty"] == difficulty]

    # Shuffle and select n_questions
    rng.shuffle(all_questions)

    # If we don't have enough, return what we have
    selected = all_questions[:n_questions]

    # Add question IDs
    for i, q in enumerate(selected):
        q["id"] = i

    return selected


# ----------------------------
# Story & Non-Intervenable Variable Generation
# ----------------------------

def _prompt_story(topic: str, var_specs: List[Dict[str, Any]], edges: List[Tuple[str, str]]) -> str:
    var_summary = "\n".join(
        f"  - {v['name']}: {v['desc']} (values: {', '.join(v['values'][:5])}{'...' if len(v['values']) > 5 else ''})"
        for v in var_specs
    )
    edge_summary = ", ".join(f"{u}->{v}" for u, v in edges[:30])
    if len(edges) > 30:
        edge_summary += f", ... ({len(edges)} total)"

    return f"""{SYSTEM_JSON}

Generate a realistic research scenario / backstory for the following causal system.

Topic: "{topic}"

Variables:
{var_summary}

Causal edges (subset): {edge_summary}

Return JSON:
{{"story": "Your 3-5 sentence story here."}}

Rules for the story:
- Describe a realistic research scenario (e.g., a named study, hospital, institution, survey, or dataset).
- Name the specific dataset or study (e.g., 'the National Longitudinal Survey of Youth' or 'Mount Sinai Hospital Emergency Department records').
- Explain what kind of data is being collected and why.
- The story must be consistent with ALL the variables listed above.
- Write in second person: 'You are a researcher at...'
- Do NOT mention Bayesian networks, DAGs, or causal graphs.
- Do NOT use double quotes inside the story text; use single quotes if needed.
- 3-5 sentences maximum.
""".strip()


def generate_story(
    llm: QwenLLM,
    topic: str,
    var_specs: List[Dict[str, Any]],
    edges: List[Tuple[str, str]],
    max_tries: int = 4,
) -> str:
    """Generate a realistic research scenario/story for this world using the LLM."""
    schema_hint = 'keys: "story" (string, 3-5 sentences)'
    p = _prompt_story(topic, var_specs, edges)

    last_err: Optional[Exception] = None
    for _ in range(max_tries):
        raw = llm.chat(SYSTEM_JSON, p)
        try:
            js = extract_first_balanced_json(raw)
        except Exception as e:
            last_err = e
            try:
                js = repair_to_valid_json(llm, raw, schema_hint)
            except Exception as e2:
                last_err = e2
                continue

        story = js.get("story", "")
        if isinstance(story, str) and len(story) > 30:
            return story.strip()
        last_err = ValueError(f"Story too short or missing: {story!r}")

    # Fallback: generic story
    return (
        f"You are a researcher investigating causal relationships in the domain of {topic}. "
        f"This dataset contains {len(var_specs)} variables representing various factors in this domain. "
        f"Your goal is to understand the causal structure underlying these observations."
    )


def _prompt_non_intervenable(
    topic: str,
    story: str,
    var_specs: List[Dict[str, Any]],
) -> str:
    var_summary = "\n".join(
        f"  - {v['name']}: {v['desc']}"
        for v in var_specs
    )
    return f"""{SYSTEM_JSON}

Given the following research scenario and variables, identify which variables CANNOT be intervened upon (manipulated or randomized) in the real world.

Research scenario:
"{story}"

Topic: "{topic}"

Variables:
{var_summary}

Return JSON:
{{"non_intervenable": [
    {{"name": "VarName", "reason": "brief reason why this cannot be intervened upon"}},
    ...
]}}

Rules:
- A variable is NON-INTERVENABLE if it would be unethical, impossible, or impractical to manipulate in a real-world experiment.
- Examples of non-intervenable: gender, race, age, birth country, genetic conditions, historical events, parental background.
- Examples of intervenable: treatment type, dosage, policy decisions, school type, diet, exercise program.
- Each variable name must EXACTLY match one from the list above.
- Provide a 5-15 word reason for each.
- Typically 20-30% of variables should be non-intervenable, depending on the domain.
- Do NOT include variables that a researcher could plausibly randomize or assign in an experiment.
""".strip()


def identify_non_intervenable_variables(
    llm: QwenLLM,
    topic: str,
    story: str,
    var_specs: List[Dict[str, Any]],
    max_tries: int = 4,
) -> List[Dict[str, str]]:
    """Identify which variables cannot be intervened upon, consistent with the story."""
    valid_names = {v["name"] for v in var_specs}
    schema_hint = 'keys: "non_intervenable" (list of {name, reason})'
    p = _prompt_non_intervenable(topic, story, var_specs)

    last_err: Optional[Exception] = None
    for _ in range(max_tries):
        raw = llm.chat(SYSTEM_JSON, p)
        try:
            js = extract_first_balanced_json(raw)
        except Exception as e:
            last_err = e
            try:
                js = repair_to_valid_json(llm, raw, schema_hint)
            except Exception as e2:
                last_err = e2
                continue

        try:
            items = js["non_intervenable"]
            result = []
            for item in items:
                name = str(item["name"])
                reason = str(item.get("reason", "Cannot be manipulated"))
                if name in valid_names:
                    result.append({"name": name, "reason": reason})
            # Sanity check: at least 1 and at most 40% of variables
            if 1 <= len(result) <= int(0.4 * len(var_specs)):
                return result
            last_err = ValueError(
                f"Expected 1-{int(0.4 * len(var_specs))} non-intervenable, got {len(result)}"
            )
        except Exception as e:
            last_err = e
            continue

    # Fallback: return empty list (all variables intervenable)
    return []


# ----------------------------
# Save artifacts
# ----------------------------
def save_graph_png(
    edges: List[Tuple[str, str]],
    outpath: str,
    title: str,
    nodes: Optional[List[str]] = None,
) -> None:
    dg = nx.DiGraph()
    if nodes is not None:
        dg.add_nodes_from(nodes)
    dg.add_edges_from(edges)
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(dg, seed=0, k=0.8)
    nx.draw_networkx(dg, pos=pos, with_labels=True, arrows=True, node_size=900, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def serialize_cpds(model: DiscreteBayesianNetwork) -> List[Dict[str, Any]]:
    out = []
    for cpd in model.get_cpds():
        child = cpd.variable
        parents = list(cpd.variables[1:])
        # Get cardinality of child to reshape properly
        child_card = int(cpd.cardinality[0])  # Convert numpy int64 to Python int
        n_parent_configs = cpd.values.size // child_card
        values = np.array(cpd.values).reshape(child_card, n_parent_configs).tolist()
        out.append({"child": child, "parents": parents, "values": values, "cardinality": child_card})
    return out


# ----------------------------
# Main generation routine
# ----------------------------
def generate_world(
    llm: QwenLLM,
    topic: str,
    n_nodes: int,
    seed: int,
    outdir: str,
    max_parents: int = 3,
    edge_multiplier: float = 1.5,
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)

    # 1) Variables (chunked, stable)
    var_specs = get_variables_chunked(llm, topic, n_nodes, chunk_size=10, max_tries=6)
    names = [v["name"] for v in var_specs]

    # 2) Edge candidates (chunked)
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

    # 5) Generate questions
    n_questions = max(1, n_nodes // 10)
    questions = generate_all_questions(
        g=dag,
        n_questions=n_questions,
        difficulty=difficulty,
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
            "difficulty_filter": difficulty,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=str, default=None)
    ap.add_argument("--n_nodes", type=int, default=30, choices=[10, 20, 30])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="./out_bn")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_parents", type=int, default=3)
    ap.add_argument("--edge_mult", type=float, default=1.5)
    ap.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=[None, "easy", "medium", "hard"],
        help="Filter questions by difficulty level (easy/medium/hard). Default: all difficulties.",
    )
    args = ap.parse_args()
    

    topic = args.topic or random.choice(TOPICS)
    llm = QwenLLM(model_name=args.model, do_sample=False, max_new_tokens=900) #1200)

    world = generate_world(
        llm=llm,
        topic=topic,
        n_nodes=args.n_nodes,
        seed=args.seed,
        outdir=args.outdir,
        max_parents=args.max_parents,
        edge_multiplier=args.edge_mult,
        difficulty=args.difficulty,
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
        # Show question type breakdown
        type_counts = {}
        for q in world["questions"]:
            qt = q["question_type"]
            type_counts[qt] = type_counts.get(qt, 0) + 1
        print(" question types:", type_counts)


if __name__ == "__main__":
    main()
