#!/usr/bin/env python3
"""Child entry point for the hardened sandbox when it runs on a separate interpreter (RLV10_SANDBOX_PY).

Why: the SkyRL container's Python has other pandas/numpy/scipy versions and no statsmodels or sklearn, so agent code
would behave differently in training than in evaluation. The sandbox therefore runs on a dedicated env with exactly
the evaluation versions (`~/rpgrl/envs/rpg_sandbox`, pins in rl_v10/README), in training and in our evals.

Protocol (sandbox_hardened.run_code): argv = [bench_dir, rl_v10_dir]; stdin = pickle of
{"code", "csv_map", "carried": {name: pickled bytes}, "wd"}. The child runs the release `_worker` behind the hardened
guard, then writes pickle((out, {name: pickled bytes})) to <wd>/.rlv10_result.pkl (inside the analysis folder, so the
audit hook allows it). Carried variables stay pickled bytes in the parent, which never unpickles agent objects.
"""
import os
import pickle
import sys


class _Q:
    def put(self, x):
        self.x = x


def main():
    bench, here = sys.argv[1], sys.argv[2]
    sys.path[:0] = [bench, here]
    payload = pickle.loads(sys.stdin.buffer.read())
    carried = {}
    for k, b in payload["carried"].items():
        try:
            carried[k] = pickle.loads(b)
        except Exception:
            pass
    import sandbox_hardened
    q = _Q()
    sandbox_hardened._hardened_worker(payload["code"], payload["csv_map"], carried, q, payload["wd"])
    out, new_vars = q.x
    enc = {}
    for k, v in new_vars.items():
        try:
            enc[k] = pickle.dumps(v)
        except Exception:
            pass
    with open(os.path.join(payload["wd"], ".rlv10_result.pkl"), "wb") as f:
        pickle.dump((out, enc), f)


if __name__ == "__main__":
    main()
