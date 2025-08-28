# -*- coding: utf-8 -*-
from pathlib import Path

def save_show(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    print("Figure saved to:", out_path)
