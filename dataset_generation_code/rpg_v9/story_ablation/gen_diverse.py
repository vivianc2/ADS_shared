import importlib.util, pyarrow.parquet as pq, json, os, re, random, sys, time
J="/home/ec2-user/.claude/jobs/97ff74aa/tmp"
from bedrock_llm import BedrockLLM
SK=(lambda s: (s.loader.exec_module(m:=importlib.util.module_from_spec(s)) or m))(importlib.util.spec_from_file_location("skins","skins.py")).SKINS
NF=(lambda s: (s.loader.exec_module(m:=importlib.util.module_from_spec(s)) or m))(importlib.util.spec_from_file_location("nf",f"{J}/neutral_fills.py")).NEUTRAL
llm=BedrockLLM(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", temperature=1.0)
SRC="/opt/dlami/nvme/rpg_data/rl_a4_base_ds"
CACHE=f"{J}/diverse_stories.json"
cache=json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def neutral_scenario(skin):
    v=SK[skin]; return v["scenario"].format(**{**v["fills"], **NF.get(skin,{})})
SYS=("You rewrite an industrial-diagnosis situation briefing into a DISTINCTIVE variant. "
     "Rules: keep the SAME scenario meaning — same domain, same outcome metric declining, same neutral stance. "
     "Vary wording, concrete operational detail, and framing so no two briefings read alike. "
     "CRITICAL: do NOT introduce, name, favor, or hint at any specific cause, culprit, or fix — the team has no consensus. "
     "Do NOT mention specific measurable signals or controls by name. One paragraph, 4-6 sentences, same professional register. "
     "End with an instruction to determine, through experiments, what is really driving the change and what to do about it. "
     "Output ONLY the paragraph, no preamble.")
seeds=[]
for fn in ["train.parquet","validation.parquet"]:
    for r in pq.read_table(f"{SRC}/{fn}").to_pylist():
        seeds.append((str(r["extra_info"]["seed"]), r["extra_info"]["skin"]))
print(f"total worlds: {len(seeds)}; cached: {len(cache)}", flush=True)
n=0
for seed,skin in seeds:
    if seed in cache: continue
    base=neutral_scenario(skin)
    for attempt in range(3):
        try:
            out=llm.generate(SYS, f"Situation briefing to rewrite (domain: {skin}):\n\n{base}", max_new_tokens=400).strip()
            # basic non-leak / format guards
            bad=any(w in out.lower() for w in ["the cause is","the culprit","root cause is","caused by","because the","the true cause","the real cause"])
            if len(out)>150 and len(out)<1600 and not bad and "\n\n" not in out:
                cache[seed]=out; break
        except Exception as e:
            time.sleep(2)
    n+=1
    if n%25==0:
        json.dump(cache,open(CACHE,"w")); print(f"...{n} done, {len(cache)} cached", flush=True)
json.dump(cache,open(CACHE,"w"))
print(f"DONE gen: {len(cache)}/{len(seeds)} stories cached -> {CACHE}", flush=True)
