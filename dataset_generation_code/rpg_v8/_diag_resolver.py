"""Diagnose the resolver: is the 0-LLM-resolution failure qwen-specific, or a
resolver-logic/prompt/parse bug that would ALSO break Opus 4.8?

Builds a Resolver with each candidate LLM and calls _llm_pick on real requests that
the lexical path rejected, printing the RAW model output and the parsed Resolution.
"""
import json, sys, os
sys.path.insert(0, "/home/ec2-user/SageMaker/vivian/ADS_shared/framework_code")
sys.path.insert(0, "/home/ec2-user/SageMaker/vivian/ADS_shared/dataset_generation_code/rpg_rl")
from resolver import Resolver
from engine import WorldSCM

# a world + a request its lexical resolver rejected
world = json.load(open("/home/ec2-user/SageMaker/vivian/data/rpg_v9_val_worlds/world_v7_agronomy_competing__20000086.json"))
scm = WorldSCM.from_dict(world["scm"])
requests = ["inject acid to lower irrigation pH", "add chelated iron supplement",
            "reduce micronutrient lockout"]   # last one is an actual alias (control)

print("ACTUATORS:", {aid: a.get("aliases") for aid, a in scm.actuators.items()})


def wrap(name, llm):
    print("\n" + "=" * 70 + f"\nRESOLVER LLM = {name}\n" + "=" * 70)
    r = Resolver(scm, llm=llm)
    # monkey-patch to capture raw
    orig = llm.generate if llm else None
    for req in requests:
        if llm is not None:
            raw_holder = {}
            def gen(sys_p, usr_p, max_new_tokens=None, _o=orig, _h=raw_holder):
                out = _o(sys_p, usr_p, max_new_tokens=max_new_tokens)
                _h["raw"] = out
                return out
            llm.generate = gen
        res = r.resolve_intervene(req, value=50)
        raw = raw_holder.get("raw", "(n/a)") if llm else "(no llm)"
        print(f"\n  request: {req!r}")
        print(f"    -> kind={res.kind} ok={res.ok} target={res.target_id} method={res.method}")
        print(f"    RAW LLM OUTPUT: {raw!r}"[:500])
        if llm is not None:
            llm.generate = orig


# 1) Bedrock Opus 4.8 (needs boto3 — present in ADS-rpg)
try:
    from bedrock_llm import BedrockLLM
    wrap("Bedrock Opus 4.8", BedrockLLM(model_id="us.anthropic.claude-opus-4-8", max_new_tokens=200))
except Exception as e:
    print("Opus resolver load FAILED:", type(e).__name__, e)

# 2) Nautilus qwen3-small (what the run actually degraded to)
try:
    from openai_llm import OpenAILLM
    q = OpenAILLM(model_name="qwen3-small", base_url=os.environ.get("NAUTILUS_BASE_URL", "https://ellm.nrp-nautilus.io/v1"),
                  api_key=os.environ.get("NAUTILUS_API_KEY"), max_new_tokens=200)
    wrap("Nautilus qwen3-small", q)
except Exception as e:
    print("qwen resolver load FAILED:", type(e).__name__, e)
