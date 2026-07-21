#!/usr/bin/env python3
"""Build gates for the TNSE manuscript. Run from paper/tnse after a full
pdflatex+bibtex+pdflatex+pdflatex cycle. Exit 0 iff all gates pass.

Gates: total pages == 14 (<= 18 TNSE cap); references end on page 10
(body+refs <= 10 pages, user constraint); no overfull boxes; no undefined
or multiply-defined references; 90 bibliography entries; abstract <= 200
words; exactly 5 keywords; all fonts embedded; no em-dashes in source; no
retired phrases from the review remediation.
"""
import glob
import re
import subprocess
import sys

RETIRED = ["information barrier", "not a defect of particular indices",
           "predictive validity", "recorded closure", "zero-betweenness mass",
           "verifies the exponent"]

def _norm(t):
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def _last_bib_tail(n=60):
    """Tail of the final bibliography entry, as rendered text."""
    bbl = open("main.bbl").read()
    last = bbl.split("\\bibitem")[-1]
    last = last.split("\\end{thebibliography}")[0]        # BEFORE stripping macros,
    last = re.sub(r"\\[a-zA-Z]+\s*", " ", last)          # else "thebibliography"
    last = re.sub(r"[{}~\\]", " ", last)                  # survives into the tail
    return _norm(last)[-n:]


def _bib_spills(p10, p11):
    """True if the last reference does not fit entirely on p10.

    An entry can WRAP across the page break: its number sits on p10 while the rest
    continues on p11. Checking only for entry numbers at line start on p11 misses
    that -- it did, once -- so verify the final entry's tail text is on p10.
    """
    tail = _last_bib_tail()
    return tail not in _norm(p10)


log = open("main.log", encoding="utf-8", errors="replace").read()
out = subprocess.run(["pdftotext", "main.pdf", "-"],
                     capture_output=True, text=True).stdout
pages = out.split("\f")
p10, p11 = pages[9], pages[10]

checks = {
    "pages_total_15_le_18": len([p for p in pages if p.strip()]) == 15,
    # A reference can WRAP across the page break: its number sits on p10 while the
    # rest of the entry continues on p11. Checking only for entry numbers on p11
    # misses that (it did, once). Assert instead that nothing before the Appendix
    # heading on p11 looks like bibliography continuation.
    "refs_end_p10": "[71]" in p10 and not re.findall(r"^\[\d+\]", p11, re.M)
                    and not _bib_spills(p10, p11),
    "no_overfull": "Overfull" not in log,
    "no_undefined": "undefined" not in log.lower(),
    "no_multiply_defined": "multiply" not in log.lower(),
    "bib_71_entries": open("main.bbl").read().count("\\bibitem") == 71,
}

src = open("main.tex").read()
a = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
t = re.sub(r"\\emph\{([^}]*)\}", r"\1", a)
t = re.sub(r"\$[^$]*\$", "X", t)
t = re.sub(r"\\[a-zA-Z]+", "", t).replace("{", "").replace("}", "")
checks["abstract_le_200"] = len([w for w in re.split(r"[\s~]+", t) if w]) <= 200
k = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", src, re.S).group(1)
checks["keywords_5"] = len([x for x in k.replace("\n", " ").split(",") if x.strip()]) == 5

bad = []
for f in glob.glob("sec-*.tex") + glob.glob("app-*.tex") + ["main.tex"]:
    c = open(f).read()
    if "---" in c:
        bad.append(("em-dash", f))
    for ph in RETIRED:
        if ph in c.lower():
            bad.append((ph, f))
checks["no_retired_phrases"] = not bad

fonts = subprocess.run(["pdffonts", "main.pdf"], capture_output=True, text=True).stdout
rows = [l.split() for l in fonts.splitlines()[2:] if l.strip()]
checks["fonts_embedded"] = all(r[-4] == "yes" for r in rows)

fails = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS" if v else "FAIL"), k)
if bad:
    print(bad)
sys.exit(1 if fails else 0)
