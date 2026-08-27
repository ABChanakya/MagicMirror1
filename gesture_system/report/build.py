#!/usr/bin/env python3
"""
build.py — wrap the report fragment into a standalone HTML document.

scaling_report.html is written as a *fragment*: no <!doctype>, <html>, <head> or
<body>, because the Claude artifact host supplies those itself. Opened directly
from disk or served by GitHub Pages that is malformed, so this generates
index.html with the full document structure around the same content.

Run after editing scaling_report.html:

    python build.py

Keeping index.html generated rather than hand-maintained means the two copies
cannot drift apart.
"""

from pathlib import Path
import re

HERE = Path(__file__).parent
SRC = HERE / "scaling_report.html"
OUT = HERE / "index.html"

fragment = SRC.read_text(encoding="utf-8")

# Pull <title> and any <link rel="stylesheet"|"preconnect"> up into <head>;
# everything else stays in <body>.
head_bits = []


def lift(pattern):
    global fragment
    for m in re.finditer(pattern, fragment, re.I):
        head_bits.append(m.group(0))
    fragment = re.sub(pattern, "", fragment, flags=re.I)


lift(r'<title>.*?</title>')
lift(r'<link\b[^>]*>')

# <style> can stay where it is, but head is tidier
style_m = re.search(r'<style>.*?</style>', fragment, re.S | re.I)
style = ""
if style_m:
    style = style_m.group(0)
    fragment = fragment.replace(style, "", 1)

title = next((b for b in head_bits if b.lower().startswith("<title")), "<title>Report</title>")
links = [b for b in head_bits if not b.lower().startswith("<title")]

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Cross-validated data-scaling experiment for the MagicMirror gesture classifier: a null recording is worth about 27x more per clip than another swipe.">
{title}
{chr(10).join(links)}
{style}
</head>
<body>
{fragment.strip()}
</body>
</html>
"""

OUT.write_text(doc, encoding="utf-8")
kb = OUT.stat().st_size / 1024
print(f"wrote {OUT.relative_to(HERE.parent)}  ({kb:.0f} KB)")
print("open it with:  xdg-open report/index.html")
