#!/usr/bin/env python3
"""run `python3 build.py`"""
core = open("core.js").read()
core = core.replace(
    'if (typeof module !== "undefined") module.exports = '
    '{ World, Space, Learner, Shaper, makeRng, phiOf, llSign, logNdtr, erfc };\n', '')
html = (open("template.html").read()
        .replace("/*__CORE__*/", core)
        .replace("/*__UI__*/", open("ui.js").read()))
open("causal_bench.html", "w").write(html)
print("wrote causal_bench.html", len(html), "bytes")
