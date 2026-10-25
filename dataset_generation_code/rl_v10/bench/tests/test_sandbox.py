#!/usr/bin/env python3
"""Sandbox checks (v10.2): normal analysis works, file access outside the analysis folder is refused (and, since
2026-10-24, ctypes file access, the environment, /proc secrets and UDP)."""
import os, sys, tempfile, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from code_sandbox import run_code

WORLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test", "index.json")
TESTS = [  # (name, code, expect substring in last line, should_pass)
    ("describe wide", "print(experiment_1_df.describe())", "max", True),
    ("read_csv bare name", "df=pd.read_csv('experiment_1_csv'); print(df.shape)", "(50, 15)", True),
    ("read_csv file name", "df=pd.read_csv('experiment_1.csv'); print(df.shape)", "(50, 15)", True),
    ("read_csv variable", "df=pd.read_csv(experiment_1_csv); print(df.shape)", "(50, 15)", True),
    ("statsmodels", "import statsmodels.api as sm\nX=sm.add_constant(experiment_1_df[['Temp']])\nprint('r2', sm.OLS(experiment_1_df.Yield,X).fit().rsquared>-1)", "r2 True", True),
    ("statsmodels formula", "import statsmodels.formula.api as smf\nprint(smf.ols('Yield ~ Temp', data=experiment_1_df).fit().params.shape)", "(2,)", True),
    ("sklearn", "from sklearn.linear_model import LinearRegression\nprint(LinearRegression().fit(experiment_1_df[['Temp']],experiment_1_df.Yield).coef_.shape)", "(1,)", True),
    ("listdir .", "import os; print(sorted(os.listdir('.')))", "experiment_1.csv", True),
    ("write+read local", "open('notes.txt','w').write('x'); print(open('notes.txt').read())", "x", True),
    ("listdir parent", "import os; print(os.listdir('..'))", "PermissionError", False),
    ("open world index", f"print(open({WORLD!r}).read()[:40])", "PermissionError", False),
    ("pandas read world", f"print(pd.read_json({WORLD!r}).shape)", "PermissionError", False),
    ("glob home (blocked: empty)", "import glob; print(glob.glob('/home/*/ADS_folder/*'))", "[]", False),
    ("subprocess", "import subprocess; print(subprocess.run(['ls','/'],capture_output=True).stdout[:30])", "not allowed", False),
    ("os.system", "import os; os.system('ls /')", "not allowed", False),
    # 2026-10-24 hardening: libc file access through ctypes, the environment, /proc secrets, UDP
    ("ctypes libc fopen", f"import ctypes; libc=ctypes.CDLL(None); libc.fopen.restype=ctypes.c_void_p; print(libc.fopen({WORLD!r}.encode(), b'r'))", "not allowed", False),
    ("ctypes pythonapi", f"import ctypes; f=ctypes.pythonapi.fopen; print(f)", "not allowed", False),
    ("os.environ scrubbed", "import os; print(sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k or 'SECRET' in k))", "[]", True),
    ("/proc/self/environ", "print(open('/proc/self/environ','rb').read()[:20])", "PermissionError", False),
    ("udp send", "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b'x', ('127.0.0.1', 9))", "not allowed", False),
    ("numeric idioms", "print(round(float(((experiment_1_df-experiment_1_df.mean())/experiment_1_df.std()).abs().max().max()),0)>0)", "True", True),
]

def main():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "experiment_1.csv")
    cols = {"unit_id": [1000 + i for i in range(50)], "Yield": np.random.rand(50), "Temp": np.random.rand(50)}
    cols.update({f"ColumnNumber{i}": np.random.rand(50) for i in range(12)})
    pd.DataFrame(cols).to_csv(p, index=False)
    before = len([x for x in os.listdir('/tmp') if x.startswith('rpg_code_')])
    bad = 0
    for name, code, expect, should_pass in TESTS:
        t0 = time.time()
        out, _ = run_code(code, {"experiment_1_csv": p})
        ok = expect in out
        bad += not ok
        last = out.strip().splitlines()[-1][:110] if out.strip() else "(empty)"
        print(f"{'PASS' if ok else 'FAIL'} {name:22s} {time.time()-t0:4.1f}s  {last}")
    out, _ = run_code("print(experiment_1_df.shape)", {"experiment_1_csv": p})
    wide, _ = run_code("print(experiment_1_df.head(2))", {"experiment_1_csv": p})
    print("wide print elided?", "..." in wide.splitlines()[0])
    carried = {}
    for code in ("best = 5", "best = 7", "print('best', best)"):
        o, nv = run_code(code, {"experiment_1_csv": p}, carried)
        carried.update(nv)
    ok = "best 7" in o
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} carried reassignment   {o.strip()}")
    after = len([x for x in os.listdir('/tmp') if x.startswith('rpg_code_')])
    print("temp folders left behind:", after - before)
    print("FAILURES:", bad)
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
