import json, sys, glob

wid_sub = sys.argv[1]
rdir = sys.argv[2] if len(sys.argv) > 2 else "/work/results_v9_validation_27b"
f = [x for x in glob.glob(f"{rdir}/result_*.json") if wid_sub in x][0]
d = json.load(open(f))
print("WORLD:", d["world_id"], "| answered:", d.get("answered"), "| turns:", len(d["turns"]), "| hit_cap:", d.get("hit_turn_cap"))
g = d.get("grade") or {}
print("GRADE: accepted=%s partA=%s benefit=%s partB=%s" % (g.get("accepted"), g.get("part_a_utility_ok"), g.get("benefit_recovered"), g.get("battery_fraction")))
print("gold_intervention:", g.get("gold_intervention"))
print("recommended_intervention (resolved):", g.get("recommended_intervention"))
print("battery_items:", g.get("battery_items"))
print("=" * 90)
for t in d["turns"]:
    at = t.get("action_type", t.get("turn"))
    pay = t.get("payload") or ""
    res = t.get("result") or ""
    if isinstance(res, (dict, list)):
        res = json.dumps(res)
    print(f"\n### turn {t.get('turn')}  action={at}  finish={t.get('finish_reason')}")
    if pay:
        print("  payload:", json.dumps(pay)[:300] if isinstance(pay, (dict, list)) else str(pay)[:300])
    if res:
        print("  result:", str(res)[:500])
# full raw of the terminal/answer turn(s)
print("\n" + "=" * 90 + "\nFULL RAW OF ANSWER / LAST 2 TURNS:\n" + "=" * 90)
for t in d["turns"][-2:]:
    print(f"\n----- turn {t.get('turn')} action={t.get('action_type')} -----")
    print(t.get("raw") or "(no raw)")
