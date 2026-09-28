import importlib.util, pyarrow as pa, pyarrow.parquet as pq, os, json, sys
J="/home/ec2-user/.claude/jobs/97ff74aa/tmp"
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
SK=load("skins","skins.py").SKINS
NF=load("nf",f"{J}/neutral_fills.py").NEUTRAL
SRC="/opt/dlami/nvme/rpg_data/rl_a4_base_ds"; DST="/opt/dlami/nvme/rpg_data/rl_neutral_ds"; os.makedirs(DST,exist_ok=True)
rep={}
for fn in ["train.parquet","validation.parquet"]:
    t=pq.read_table(f"{SRC}/{fn}"); rows=t.to_pylist(); changed=0
    for r in rows:
        skin=r["extra_info"]["skin"]; orig=SK[skin]["fills"]; neu=NF.get(skin,{})
        repls=[(orig[k],neu[k]) for k in ("naive_theories","surface_clue") if k in orig and k in neu and orig[k]]
        rc=False
        for msg in r["prompt"]:
            if "SITUATION" not in msg["content"]: continue
            c=msg["content"]
            for old,new in repls:
                if old in c: c=c.replace(old,new); rc=True
            msg["content"]=c
        changed+= 1 if rc else 0
    assert changed==len(rows), f"{fn}: {changed}/{len(rows)}"
    pq.write_table(pa.Table.from_pylist(rows,schema=t.schema), f"{DST}/{fn}")
    rep[fn]={"rows":len(rows),"changed":changed}
print(json.dumps(rep))
