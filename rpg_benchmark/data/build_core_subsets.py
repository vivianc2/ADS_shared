#!/usr/bin/env python3
"""Build the core subsets of data/test: core54 (6 worlds per structure) and core108 (12 per structure, contains core54).

    python data/build_core_subsets.py            # writes data/core108, checks data/core54 is reproduced
    python data/build_core_subsets.py --check    # only checks both folders against the rule

Rule. Within each (structure, domain) cell, worlds are ordered by sha1 of their file name.
- core54: per structure, 6 of the 10 domains (the existing core54 rotation), first world of each cell.
- core108 = core54 + 54 worlds:
  * the first world of each of the 4 domains core54 left out for that structure (36 worlds; every
    (structure, domain) cell now has one world);
  * the second world of 2 more cells per structure (18 worlds). Structures are taken in sorted order; each picks the
    2 domains with the fewest second-world picks so far (ties by sha1 of "structure/domain"), among cells with at
    least two worlds. This gives every domain 10 or 11 worlds.
"""
import argparse, collections, gzip, hashlib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def h(s):
    return hashlib.sha1(s.encode()).hexdigest()


def stem(f):
    return f[len("world_"):-len(".json.gz")]


def load(split):
    return json.load(open(os.path.join(HERE, split, "index.json")))


def build():
    test = load("test")
    core54 = load("core54")
    cells = collections.defaultdict(list)
    for e in test:
        cells[(e["archetype"], e["domain"])].append(e)
    for k in cells:
        cells[k].sort(key=lambda e: h(e["file"]))
    for e in core54:   # core54 is the first world of its cell
        assert cells[(e["archetype"], e["domain"])][0]["file"] == e["file"], e["file"]

    domains = sorted({e["domain"] for e in test})
    structures = sorted({e["archetype"] for e in test})
    in54 = {(e["archetype"], e["domain"]) for e in core54}
    picked = list(core54)
    for a in structures:
        picked += [cells[(a, d)][0] for d in domains if (a, d) not in in54]
    second = collections.Counter()
    for a in structures:
        ok = [d for d in domains if len(cells[(a, d)]) >= 2]
        for d in sorted(ok, key=lambda d: (second[d], h(f"{a}/{d}")))[:2]:
            second[d] += 1
            picked.append(cells[(a, d)][1])
    order = {a: i for i, a in enumerate(structures)}
    picked.sort(key=lambda e: (order[e["archetype"]], e["domain"], e["file"]))
    return picked


def write(split, entries):
    out = os.path.join(HERE, split)
    os.makedirs(out, exist_ok=True)
    for e in entries:
        src, dst = os.path.join(HERE, "test", e["file"]), os.path.join(out, e["file"])
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    json.dump(entries, open(os.path.join(out, "index.json"), "w"), indent=1)
    open(os.path.join(out, "worlds.txt"), "w").write("".join(stem(e["file"]) + "\n" for e in entries))


def check(split, entries):
    out = os.path.join(HERE, split)
    want = {e["file"] for e in entries}
    have_idx = {e["file"] for e in load(split)}
    have_gz = {f for f in os.listdir(out) if f.endswith(".json.gz")}
    have_txt = {f"world_{l.strip()}.json.gz" for l in open(os.path.join(out, "worlds.txt")) if l.strip()}
    bad = [n for n, s in (("index.json", have_idx), ("world files", have_gz), ("worlds.txt", have_txt)) if s != want]
    for f in want & have_gz:   # copies are byte-identical to data/test
        a, b = os.path.join(out, f), os.path.join(HERE, "test", f)
        if open(a, "rb").read() != open(b, "rb").read():
            bad.append(f"{f} differs from data/test")
    print(f"{split}: {len(want)} worlds, " + ("OK" if not bad else "MISMATCH: " + "; ".join(bad)))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    c108 = build()
    c54 = load("core54")
    assert len(c108) == 108 and len({e["file"] for e in c108}) == 108
    assert {e["file"] for e in c54} <= {e["file"] for e in c108}
    if not args.check:
        write("core108", c108)
    ok = check("core54", c54) & check("core108", c108)
    print("per structure:", dict(collections.Counter(e["archetype"] for e in c108)))
    print("per domain:   ", dict(sorted(collections.Counter(e["domain"] for e in c108).items())))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
