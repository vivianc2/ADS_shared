#!/usr/bin/env python3
"""Maps free-text requests to the world's signals and controls."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from scm import WorldSCM

logger = logging.getLogger("resolver")


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set:
    return set(_norm(s).split())


@dataclass
class Resolution:
    kind: str
    ok: bool
    interpretation: str
    target_id: Optional[str] = None
    value: Any = None
    reason: str = ""
    candidates: List[str] = field(default_factory=list)
    method: str = "alias"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class Resolver:
    def __init__(self, scm: WorldSCM, llm: Any = None, display: bool = False):
        self.scm = scm
        self.llm = llm
        self.display = display
        self.var_aliases: Dict[str, List[str]] = {}
        for name, spec in scm.variables.items():
            if spec["kind"] in ("observable", "outcome") or spec.get("measurable"):
                al = [name] + list(spec.get("aliases", []))
                self.var_aliases[name] = al
        self.act_aliases: Dict[str, List[str]] = {}
        for aid, act in scm.actuators.items():
            self.act_aliases[aid] = [aid] + list(act.get("aliases", []))
        self.hidden_aliases: Dict[str, List[str]] = {}
        for name, spec in scm.variables.items():
            if not (spec["kind"] in ("observable", "outcome") or spec.get("measurable")):
                self.hidden_aliases[name] = [name] + list(spec.get("aliases", []))

    _STOP = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "with",
             "please", "let", "set", "add", "adjust", "change", "increase", "reduce",
             "lower", "raise", "measure", "check", "read", "test", "assay", "level",
             "content", "some", "more", "less", "high", "low", "moderate", "medium",
             "max", "min", "up", "down", "its", "value", "rate", "me", "what", "is"}

    _SHORT_ENGLISH = {"do", "no", "on", "off", "up", "it", "is", "be", "go", "so",
                      "of", "to", "in", "at", "we", "me", "my", "an", "as", "or"}

    def _phrase_in(self, phrase: str, text: str) -> bool:
        if not phrase:
            return False
        return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None

    def _doc_freq(self, alias_map: Dict[str, List[str]]) -> Dict[str, int]:
        df: Dict[str, int] = {}
        for cid, aliases in alias_map.items():
            seen = set()
            for al in aliases:
                seen |= (_tokens(al) - self._STOP)
            for tok in seen:
                df[tok] = df.get(tok, 0) + 1
        return df

    @staticmethod
    def _stem_match(a: str, b: str) -> bool:
        if a == b:
            return True
        n = min(len(a), len(b))
        if n < 5:
            return False
        k = 0
        while k < n and a[k] == b[k]:
            k += 1
        return k >= 5

    def _best_alias(self, text: str, alias_map: Dict[str, List[str]]) -> List[Tuple[str, float]]:
        t = _norm(text)
        req_tok = _tokens(text) - self._STOP
        df = self._doc_freq(alias_map)
        n_entries = max(1, len(alias_map))
        def distinctive_tok(w: str) -> bool:
            return len(w) >= 4 and df.get(w, 0) <= max(1, n_entries // 5)

        scored: List[Tuple[str, float]] = []
        for cid, aliases in alias_map.items():
            best = 0.0
            for al in aliases:
                na = _norm(al)
                if not na:
                    continue
                al_tok = _tokens(al) - self._STOP
                if not al_tok:
                    continue
                if len(na) >= 3 and self._phrase_in(na, t):
                    best = max(best, 0.9 + min(len(na), 30) / 300.0)
                    continue
                if (len(na) <= 3 and na not in self._SHORT_ENGLISH
                        and na in al_tok and df.get(na, 0) <= 1
                        and self._phrase_in(na, t)):
                    best = max(best, 0.9)
                    continue
                matched = []
                matched_req = set()
                any_distinctive = False
                for at in al_tok:
                    for rt in req_tok:
                        if self._stem_match(at, rt):
                            matched.append(at)
                            matched_req.add(rt)
                            if distinctive_tok(at) or distinctive_tok(rt):
                                any_distinctive = True
                            break
                if matched:
                    coverage = len(matched) / len(al_tok)
                    n_distinct_matched = sum(1 for w in (matched_req) if distinctive_tok(w))
                    req_distinct = {w for w in req_tok if distinctive_tok(w)} or req_tok
                    req_cov = len(matched_req & req_distinct) / max(1, len(req_distinct))
                    if any_distinctive:
                        score = 0.72 + 0.2 * coverage
                        if req_cov < 0.5:
                            score = min(score, 0.50 + 0.06 * n_distinct_matched)
                    else:
                        score = 0.4 * coverage
                    best = max(best, score)
            if best > 0:
                scored.append((cid, best))
        scored.sort(key=lambda x: -x[1])
        return scored

    @staticmethod
    def _answer_variants(text: str) -> List[str]:
        t = (text or "").strip()
        if not t:
            return []
        variants = []
        def add(s):
            s = s.strip(" \t-–—:;,/")
            if s and s.lower() not in {v.lower() for v in variants}:
                variants.append(s)
        head = re.split(r"[(\[:—–\-/]| - ", t, maxsplit=1)[0]
        add(head)
        for m in re.findall(r"[(\[]([^)\]]*)[)\]]", t):
            add(m)
        for part in re.split(r"[(){}\[\]:—–/]|\s-\s", t):
            add(part)
        add(t)
        return variants

    def resolve_answer_term(self, text: str, kind: str = "measure",
                            value: Any = None) -> Resolution:
        def _resolve(v):
            return (self.resolve_intervene(v, value) if kind == "intervene"
                    else self.resolve_measure(v))

        def _rare_token_hit(form: str):
            alias_map = self.var_aliases if kind == "measure" else self.act_aliases
            df = self._doc_freq(alias_map)
            ftoks = _tokens(form) - self._STOP
            hits = {}
            for cid, aliases in alias_map.items():
                atoks = set()
                for al in aliases:
                    atoks |= (_tokens(al) - self._STOP)
                for tok in ftoks & atoks:
                    if len(tok) <= 3 and tok in self._SHORT_ENGLISH:
                        continue
                    if df.get(tok, 0) == 1:
                        hits.setdefault(cid, set()).add(tok)
            return next(iter(hits)) if len(hits) == 1 else None

        variants = self._answer_variants(text)
        if not variants:
            return _resolve(text or "")

        head = variants[0]
        r = _resolve(head)
        if r.ok and r.target_id and not r.candidates:
            return r
        rare = _rare_token_hit(head)
        if rare is not None:
            return (self._finalize_intervene(rare, value, head, "alias") if kind == "intervene"
                    else Resolution("measure", True, f"measure '{rare}'", target_id=rare, method="alias"))
        if self.llm is not None:
            r = (self._llm_pick(head, kind, value=value) if kind == "intervene"
                 else self._llm_pick(head, kind))
            if r is not None and r.ok and r.target_id:
                if kind == "intervene" and r.value is None:
                    r = self._finalize_intervene(r.target_id, value, head, r.method)
                return r

        for v in variants[1:]:
            r = _resolve(v)
            if r.ok and r.target_id and not r.candidates:
                return r
            rare = _rare_token_hit(v)
            if rare is not None:
                return (self._finalize_intervene(rare, value, v, "alias") if kind == "intervene"
                        else Resolution("measure", True, f"measure '{rare}'", target_id=rare, method="alias"))
        return _resolve(text or "")

    def resolve_measure(self, request: str) -> Resolution:
        if self.llm is not None:
            r = self._llm_pick(request, "measure")
            if r is not None:
                return r
        meas = self._best_alias(request, self.var_aliases)
        hid = self._best_alias(request, self.hidden_aliases)
        meas_best = meas[0] if meas else (None, 0.0)
        hid_best = hid[0] if hid else (None, 0.0)

        if hid_best[1] >= 0.85 and hid_best[1] >= meas_best[1]:
            return Resolution("reject", False, f"no assay for '{hid_best[0]}'",
                              reason="That quantity is not directly measurable on this rig; "
                                     "there is no assay for it.", method="alias")

        if meas_best[1] >= 0.85:
            return Resolution("measure", True, f"measure '{meas_best[0]}'",
                              target_id=meas_best[0], method="alias")

        top = [c for c, s in meas if s >= 0.6][:4]
        if len(top) == 1 and meas_best[1] >= hid_best[1]:
            return Resolution("measure", True, f"measure '{top[0]}'", target_id=top[0], method="alias")
        if len(top) > 1:
            return Resolution("measure", False, "ambiguous measurement request",
                              candidates=top, reason="Please name the quantity more specifically.",
                              method="alias")
        if hid_best[1] >= 0.6:
            return Resolution("reject", False, f"no assay for '{hid_best[0]}'",
                              reason="That quantity is not directly measurable on this rig; "
                                     "there is no assay for it.", method="alias")
        return Resolution("reject", False, "unrecognized measurement",
                          reason="No instrument on this system measures that; it is not part of "
                                 "the available signals.", method="alias")

    def resolve_intervene(self, request: str, value: Any = None) -> Resolution:
        if self.llm is not None:
            r = self._llm_pick(request, "intervene", value=value)
            if r is not None:
                return r
        scored = self._best_alias(request, self.act_aliases)
        if scored and scored[0][1] >= 0.85:
            aid = scored[0][0]
            return self._finalize_intervene(aid, value, request, "alias")
        strong = [c for c, s in scored if s >= 0.6][:4]
        weak = [c for c, s in scored if 0.4 <= s < 0.6][:4]
        if len(strong) == 1:
            return self._finalize_intervene(strong[0], value, request, "alias")
        if len(strong) > 1 or weak:
            return Resolution("intervene", False, "ambiguous intervention request",
                              candidates=(strong or weak), reason="Please specify which control/additive you mean.",
                              method="alias")
        hid = self._best_alias(request, {**self.hidden_aliases, **self.var_aliases})
        if hid and hid[0][1] >= 0.6:
            return Resolution("reject", False, f"no actuator for '{hid[0][0]}'",
                              reason="There is no way to directly set that on this system; "
                                     "no actuator is connected to it.", method="alias")
        return Resolution("reject", False, "unrecognized intervention",
                          reason="That equipment/additive is not present on this line.",
                          method="alias")

    def _range(self, act: Dict[str, Any]):
        u = act.get("units") if self.display else None
        return [u["lo"], u["hi"]] if u else act.get("range", [0, 100])

    def _finalize_intervene(self, aid: str, value: Any, request: str, method: str) -> Resolution:
        act = self.scm.actuators[aid]
        v = self._coerce_value(act, value, request)
        desc = act.get("description", aid)
        if v is None:
            v = act.get("default")
            u = act.get("units") if self.display else None
            if u is not None and isinstance(v, (int, float)):
                v = round(u["lo"] + float(v) / 100.0 * (u["hi"] - u["lo"]), 4)
        return Resolution("intervene", True,
                          f"apply '{aid}' ({desc}) at {v}", target_id=aid, value=v, method=method)

    _QUAL_FRAC = {"max": 1.0, "maximum": 1.0, "highest": 1.0, "full": 1.0,
                  "high": 0.75, "elevated": 0.75, "increase": 0.75, "raise": 0.75,
                  "moderate": 0.5, "medium": 0.5, "mid": 0.5, "normal": 0.5,
                  "low": 0.25, "reduce": 0.25, "lower": 0.25, "decrease": 0.25,
                  "min": 0.0, "minimum": 0.0, "off": 0.0, "zero": 0.0, "none": 0.0}

    def _coerce_value(self, act: Dict[str, Any], value: Any, request: str) -> Any:
        cont = act.get("dtype", "continuous") == "continuous"

        def _from_qualitative(text: str) -> Any:
            t = _norm(text)
            if cont:
                lo, hi = self._range(act)
                for word, frac in self._QUAL_FRAC.items():
                    if self._phrase_in(word, t):
                        return round(lo + frac * (hi - lo), 2)
                return None
            vals = act.get("values", ["off", "on"])
            for vv in vals:
                if self._phrase_in(_norm(vv), t):
                    return vv
            if "on" in t.split() or "enable" in t:
                return vals[-1]
            if "off" in t.split() or "disable" in t:
                return vals[0]
            return None

        if value is not None:
            if cont:
                try:
                    fv = float(value)
                    lo, hi = self._range(act)
                    return round(float(max(lo, min(hi, fv))), 2)
                except (TypeError, ValueError):
                    q = _from_qualitative(str(value))
                    if q is not None:
                        return q
            else:
                vals = act.get("values", ["off", "on"])
                if value in vals:
                    return value
                q = _from_qualitative(str(value))
                if q is not None:
                    return q

        if cont:
            m = re.search(r"(-?\d+(?:\.\d+)?)", request)
            if m:
                lo, hi = self._range(act)
                return round(float(max(lo, min(hi, float(m.group(1))))), 2)
        return _from_qualitative(request)

    def _llm_pick(self, request: str, kind: str, value: Any = None) -> Optional[Resolution]:
        meas_list = "\n".join(f"  M:{cid}: {', '.join(al[:4])}" for cid, al in self.var_aliases.items())
        act_list = "\n".join(f"  A:{aid}: {act.get('description', aid)} "
                             f"[{', '.join(self.act_aliases[aid][:3])}]"
                             for aid, act in self.scm.actuators.items())
        hid_list = "\n".join(f"  H:{cid}: {', '.join(al[:4])}" for cid, al in self.hidden_aliases.items())
        verb = "MEASURE (observe a quantity)" if kind == "measure" else "INTERVENE (apply a control/additive)"
        prompt = (
            f'A scientist asked to {verb}: "{request}".\n\n'
            f"MEASURABLE variables (have an assay):\n{meas_list or '  (none)'}\n\n"
            f"ACTUATORS (things you can set/apply):\n{act_list or '  (none)'}\n\n"
            f"REAL but NOT directly measurable/controllable states:\n{hid_list or '  (none)'}\n\n"
            "Decide what the scientist meant, mapping to the SINGLE best catalog "
            "entry, and classify the outcome. Rules:\n"
            "- If it clearly refers to a MEASURABLE variable and kind is measure -> measure it.\n"
            "- If it clearly refers to an ACTUATOR and kind is intervene -> apply it.\n"
            "- If it refers to a REAL state that has no assay (measure) / no actuator "
            "(intervene) -> reject as no_assay / no_actuator.\n"
            "- If it refers to nothing in these lists -> reject as not_in_world.\n"
            "- Do NOT force a match on a superficial shared word (e.g. 'network link "
            "speed' is NOT 'fan speed'). Prefer a rejection over a wrong mapping.\n"
            'Reply as JSON only: {"target":"<id or empty>","outcome":'
            '"measure|intervene|no_assay|no_actuator|not_in_world"}'
        )
        try:
            raw = self.llm.generate(
                "You map a scientist's free-text request to a fixed catalog of a "
                "hidden world. Be precise; never invent equipment. JSON only.",
                prompt, max_new_tokens=2048)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            obj = json.loads(m.group(0)) if m else {}
        except Exception as e:
            logger.error("RESOLVER LLM CALL FAILED for %r (%s: %s) -> LEXICAL FALLBACK. "
                         "Scores will be deflated by unresolved actions.",
                         request, type(e).__name__, e)
            return None
        tid, outcome = str(obj.get("target", "")).strip(), obj.get("outcome", "")
        for pfx in ("A:", "M:", "H:", "a:", "m:", "h:"):
            if tid.startswith(pfx):
                tid = tid[len(pfx):].strip()
                break
        if outcome == "measure" and tid in self.var_aliases:
            return Resolution("measure", True, f"measure '{tid}' (llm)", target_id=tid, method="llm")
        if outcome == "intervene" and tid in self.scm.actuators:
            return self._finalize_intervene(tid, value, request, "llm")
        if outcome == "no_assay":
            return Resolution("reject", False, f"no assay (llm)",
                              reason="That quantity is not directly measurable on this rig.", method="llm")
        if outcome == "no_actuator":
            return Resolution("reject", False, f"no actuator (llm)",
                              reason="There is no actuator connected to that on this system.", method="llm")
        if outcome == "not_in_world":
            return Resolution("reject", False, "not present (llm)",
                              reason="That is not part of this system.", method="llm")
        logger.warning("RESOLVER LLM returned unusable output for %r (target=%r outcome=%r raw=%r) "
                       "-> LEXICAL FALLBACK.", request, tid, outcome, (raw or "")[:200])
        return None
