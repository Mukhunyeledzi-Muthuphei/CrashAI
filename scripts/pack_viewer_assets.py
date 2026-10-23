"""Wrap the viewer's binary assets (car model, HDR sky) in .js files as base64.

The published artifact only serves web file types (no .glb/.hdr), and a page opened from disk
can't fetch files, so the viewer loads these with plain <script> tags instead.

  python scripts/pack_viewer_assets.py
"""

import base64
import os

ASSETS = os.path.join(os.path.dirname(__file__), "..", "viewer", "assets")

for name in ("ferrari.glb", "venice_sunset_1k.hdr", "ferrari_ao.png"):
    with open(os.path.join(ASSETS, name), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    out = os.path.join(ASSETS, name + ".js")
    with open(out, "w") as f:
        f.write(f'(window.CRASHAI_ASSETS = window.CRASHAI_ASSETS || {{}})["{name}"] = "{b64}";\n')
    print(f"{out}: {os.path.getsize(out) // 1024} KB")
