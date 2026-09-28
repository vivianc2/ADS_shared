#!/usr/bin/env python3
"""Regression tests locking box2's 2026-09-08 pipeline fixes (run before any eval/train).
   PYTHONPATH=../rpg_v9 python -m pytest test_parse_regression.py  (or just run it)."""
import sys, os
sys.path[:0]=[os.path.dirname(__file__), os.path.join(os.path.dirname(__file__),"..","rpg_v9")]
from run_agent_v6 import _parse_action
from oracle_v6 import _canon_sign

def check(name, cond):
    print(("PASS " if cond else "FAIL ")+name); assert cond, name

# 1) <think>-block draft actions must NOT be parsed (only the committed action)
a,_=_parse_action('<think>let me try <action type="intervene">{"actions":[]}</action></think>\n<action type="measure">{"ids":["m0"]}</action>')
check("think-block action ignored -> committed 'measure'", a=="measure")
# 2) multiple committed actions -> parse failure (not silent-first), but a terminal answer is honored
a,_=_parse_action('<action type="measure">{"ids":["m0"]}</action>\n<action type="intervene">{"actions":[]}</action>')
check("multi non-terminal action -> parse fail (None)", a is None)
a,_=_parse_action('<action type="measure">{}</action>\n<action type="answer">{"proxy":"m3"}</action>')
check("multi with terminal -> honor answer", a=="answer")
# 3) sign canonicalization: int/word 'no effect' must equal gold string "0"
check("_canon_sign(0)=='0'", _canon_sign(0)=="0")
check("_canon_sign('none')=='0'", _canon_sign("none")=="0")
check("_canon_sign(1)=='+'", _canon_sign(1)=="+")
check("_canon_sign('-1')=='-'", _canon_sign("-1")=="-")
# 4) SkyRL/Qwen3.5: the opening <think> is in the PROMPT, the output has only </think>. A draft action
#    inside the reasoning must not turn a valid committed action into a multi-action parse failure.
from env import _restore_think_open
_sk = 'draft: <action type="intervene">{"actions":[]}</action>\n</think>\n<action type="measure">{"ids":["m0"]}</action>\n<memory>x</memory>'
a,_=_parse_action(_restore_think_open(_sk))
check("SkyRL output (no <think> open) with draft action -> committed 'measure'", a=="measure")
check("_restore_think_open is a no-op on balanced / think-free text",
      _restore_think_open('<think>a</think><action type="measure">{}</action>')=='<think>a</think><action type="measure">{}</action>'
      and _restore_think_open('<action type="measure">{}</action>')=='<action type="measure">{}</action>')
print("ALL PARSE/GRADE REGRESSION TESTS PASS")
if __name__=="__main__": pass
