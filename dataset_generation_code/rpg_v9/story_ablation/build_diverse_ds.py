import pyarrow as pa, pyarrow.parquet as pq, json, os, re
J="/home/ec2-user/.claude/jobs/97ff74aa/tmp"
stories=json.load(open(f"{J}/diverse_stories.json"))
SRC="/opt/dlami/nvme/rpg_data/rl_a4_base_ds"; DST="/opt/dlami/nvme/rpg_data/rl_diverse_ds"; os.makedirs(DST,exist_ok=True)
pat=re.compile(r"(SITUATION\n)(.*?)(\n\nOUTCOME OF INTEREST:)", re.DOTALL)
rep={}; missing=set()
for fn in ["train.parquet","validation.parquet"]:
    t=pq.read_table(f"{SRC}/{fn}"); rows=t.to_pylist(); changed=0
    for r in rows:
        seed=str(r["extra_info"]["seed"])
        if seed not in stories: missing.add(seed); continue
        story=stories[seed].strip()
        rc=False
        for msg in r["prompt"]:
            if "SITUATION" not in msg["content"]: continue
            new,ncount=pat.subn(lambda m: m.group(1)+story+m.group(3), msg["content"], count=1)
            if ncount==1: msg["content"]=new; rc=True
        changed+= 1 if rc else 0
    assert not missing, f"missing stories for {len(missing)} seeds e.g. {list(missing)[:5]}"
    assert changed==len(rows), f"{fn}: {changed}/{len(rows)} replaced"
    pq.write_table(pa.Table.from_pylist(rows,schema=t.schema), f"{DST}/{fn}")
    rep[fn]={"rows":len(rows),"changed":changed}
print(json.dumps(rep))
# integrity: OUTCOME/catalog unchanged vs source, only SITUATION differs
o=pq.read_table(f"{SRC}/validation.parquet").to_pylist(); n=pq.read_table(f"{DST}/validation.parquet").to_pylist()
def after_outcome(row): 
    c=[m["content"] for m in row["prompt"] if "SITUATION" in m["content"]][0]; return c.split("OUTCOME OF INTEREST",1)[1]
print("OUTCOME+catalog identical (all val rows):", all(after_outcome(a)==after_outcome(b) for a,b in zip(o,n)))
print("extra_info identical:", all(a["extra_info"]==b["extra_info"] for a,b in zip(o,n)))
print("\n--- sample new SITUATION (val row0) ---")
c=[m["content"] for m in n[0]["prompt"] if "SITUATION" in m["content"]][0]
print(c.split("SITUATION",1)[1].split("OUTCOME OF INTEREST",1)[0].strip()[:500])
