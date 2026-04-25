"""
generate_network_diagram.py — Draw MHWRiskModel architecture diagram for README.

No GCS access needed. Draws from hardcoded arch spec matching src/models/.

Output: docs/assets/figures/network_architecture.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrow

# ── colours ───────────────────────────────────────────────────────────────────
C_CNN   = "#4C72B0"
C_TRANS = "#55A868"
C_GATE  = "#DD8452"
C_HEAD  = "#8172B3"
C_INPUT = "#999999"
C_OUT   = "#C44E52"
C_SVAR  = "#A03040"
BG      = "#FAFAFA"
TXT     = "#1a1a1a"

# ── box registry (cx, cy, w, h) ───────────────────────────────────────────────
# All in axes [0,1] coords. pad=0 in boxstyle so visual edge == stated edge.
BOXES = {
    "hycom_in": (0.23, 0.890, 0.38, 0.055),
    "wn2_in":   (0.77, 0.890, 0.38, 0.055),
    "cnn":      (0.23, 0.760, 0.38, 0.100),
    "trans":    (0.77, 0.760, 0.38, 0.100),
    "gate":     (0.50, 0.575, 0.52, 0.100),
    "head":     (0.50, 0.415, 0.34, 0.070),
    "sdd":      (0.50, 0.280, 0.52, 0.060),
    "svar":     (0.50, 0.145, 0.52, 0.060),
}

def top(k):    cx, cy, w, h = BOXES[k]; return cx, cy + h / 2
def bottom(k): cx, cy, w, h = BOXES[k]; return cx, cy - h / 2
def left(k):   cx, cy, w, h = BOXES[k]; return cx - w / 2, cy
def right(k):  cx, cy, w, h = BOXES[k]; return cx + w / 2, cy


def _box(ax, key, color, label, sublabel="", fontsize=10, subfontsize=8):
    cx, cy, w, h = BOXES[key]
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0",       # pad=0: visual edge == stated coordinate
        facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.92,
        zorder=3,
    )
    ax.add_patch(rect)
    y_label = cy + 0.010 if sublabel else cy
    ax.text(cx, y_label, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.022, sublabel,
                ha="center", va="center", fontsize=subfontsize,
                color="white", alpha=0.93, zorder=4)


def _arrow(ax, x1, y1, x2, y2, label="", color="#777777"):
    """Draw edge-to-edge arrow. Points are exact box edge coordinates."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                        mutation_scale=14),
        zorder=5,
    )
    if label:
        mx = (x1 + x2) / 2 + 0.015
        my = (y1 + y2) / 2
        ax.text(mx, my, label, ha="left", va="center",
                fontsize=7, color="#666666", zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(10, 14))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── side block arrow (data flow) ──────────────────────────────────────────
    # FancyArrow(x, y, dx, dy, ...) — tail at (x,y), tip at (x+dx, y+dy)
    ax.add_patch(FancyArrow(
        0.945, 0.875,            # tail: top of arrow
        0, -0.730,               # direction: straight down
        width=0.022,
        head_width=0.048,
        head_length=0.045,
        facecolor="#cccccc",
        edgecolor="#aaaaaa",
        alpha=0.65,
        zorder=1,
        length_includes_head=True,
    ))
    ax.text(0.945, 0.520, "Data\nFlow", ha="center", va="center",
            fontsize=8, color="#888888", rotation=90, style="italic")

    # ── title ─────────────────────────────────────────────────────────────────
    ax.text(0.48, 0.970, "MHWRiskModel Architecture",
            ha="center", va="top", fontsize=14, fontweight="bold", color=TXT)
    ax.text(0.48, 0.950, "Dual-stream CNN + Transformer → LeakyGate → SDD → SVaR",
            ha="center", va="top", fontsize=9, color="#555555")

    # ── input boxes ───────────────────────────────────────────────────────────
    _box(ax, "hycom_in", C_INPUT,
         "Ocean Depth Profiles (HYCOM)",
         "(batch, 64, depth=11, ch=4)",
         fontsize=9, subfontsize=7.5)
    _box(ax, "wn2_in", C_INPUT,
         "Atmospheric Sequences (WeatherNext 2, ERA5)",
         "(batch, 64, time=90, feat=5)",
         fontsize=8.5, subfontsize=7.5)

    # ── encoder boxes ─────────────────────────────────────────────────────────
    _box(ax, "cnn", C_CNN,
         "CNN1dEncoder",
         "Conv1d(4→32)+BN+ReLU  ·  Conv1d(32→64)+BN+ReLU\n"
         "Conv1d(64→128)+BN+ReLU  ·  residual skip(32→128)\n"
         "AdaptiveAvgPool1d → (batch×64, 128)",
         fontsize=9, subfontsize=7.5)
    _box(ax, "trans", C_TRANS,
         "TransformerEncoder",
         "Linear(5→128) + sinusoidal PE\n"
         "4× PreNorm layers  (8-head attn, d_ff=256)\n"
         "mean-pool over 90 days → (batch×64, 128)",
         fontsize=9, subfontsize=7.5)

    # ── gate box ──────────────────────────────────────────────────────────────
    _box(ax, "gate", C_GATE,
         "LeakyGate",
         "g = α + (1–2α)·sigmoid(Linear([depth_feat ‖ time_feat])),  α=0.1\n"
         "fused = g · depth_feat + (1–g) · time_feat     g ∈ [0.1, 0.9]",
         fontsize=9, subfontsize=8)

    # Gate regime annotation — positioned below gate box
    gate_bottom_y = BOXES["gate"][1] - BOXES["gate"][3] / 2
    ax.text(0.50, gate_bottom_y - 0.013,
            "g→1: stratification-driven MHW   |   g→0: atmospheric-forcing-driven",
            ha="center", va="top", fontsize=7.5, color="#888888", style="italic")

    # ── head box ──────────────────────────────────────────────────────────────
    _box(ax, "head", C_HEAD,
         "Output Head",
         "Linear(128→1)  +  Softplus  →  SDD ≥ 0",
         fontsize=9, subfontsize=8)

    # ── SDD output box ────────────────────────────────────────────────────────
    _box(ax, "sdd", C_OUT,
         "SDD Predictions",
         "sdd: (batch, 64)   ·   latent: (batch, 64, 128)   ·   gate: (batch, 64)",
         fontsize=9, subfontsize=8)

    # ── SVaR box ──────────────────────────────────────────────────────────────
    _box(ax, "svar", C_SVAR,
         "SVaR Estimation",
         "SVaR_95 = quantile(sdd, 0.95, dim=member)  →  loss exceedance curve",
         fontsize=9, subfontsize=8)

    # ── arrows: input → encoder (vertical, same x column) ────────────────────
    bx_h, by_h = bottom("hycom_in")
    tx_c, ty_c = top("cnn")
    _arrow(ax, bx_h, by_h, tx_c, ty_c)

    bx_w, by_w = bottom("wn2_in")
    tx_t, ty_t = top("trans")
    _arrow(ax, bx_w, by_w, tx_t, ty_t)

    # ── arrows: encoder → gate (converging diagonals) ─────────────────────────
    # CNN bottom-center hits gate top at x=0.33 (1/3 from left of gate)
    gate_cx, gate_cy, gate_w, gate_h = BOXES["gate"]
    gate_top_y = gate_cy + gate_h / 2
    cnn_bx, cnn_by = bottom("cnn")          # (0.23, 0.710)
    _arrow(ax, cnn_bx, cnn_by,
           gate_cx - gate_w * 0.28, gate_top_y)

    trans_bx, trans_by = bottom("trans")    # (0.77, 0.710)
    _arrow(ax, trans_bx, trans_by,
           gate_cx + gate_w * 0.28, gate_top_y)

    # Dimension labels on diagonal arrows
    ax.text(0.31, 0.660, "(batch×64, 128)", ha="center", va="center",
            fontsize=7, color="#666666")
    ax.text(0.69, 0.660, "(batch×64, 128)", ha="center", va="center",
            fontsize=7, color="#666666")

    # ── arrows: gate → head → sdd → svar (vertical center spine) ─────────────
    g_bx, g_by = bottom("gate")
    h_tx, h_ty = top("head")
    _arrow(ax, g_bx, g_by, h_tx, h_ty, label="(batch×64, 128)")

    h_bx, h_by = bottom("head")
    s_tx, s_ty = top("sdd")
    _arrow(ax, h_bx, h_by, s_tx, s_ty, label="(batch×64, 1)")

    sv_tx, sv_ty = top("svar")
    s_bx, s_by  = bottom("sdd")
    _arrow(ax, s_bx, s_by, sv_tx, sv_ty)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C_CNN,   label="CNN1dEncoder — HYCOM vertical profiles"),
        mpatches.Patch(color=C_TRANS, label="TransformerEncoder — WN2/ERA5 sequences"),
        mpatches.Patch(color=C_GATE,  label="LeakyGate — learnable stream fusion"),
        mpatches.Patch(color=C_HEAD,  label="Output head — SDD regression (Softplus)"),
    ]
    ax.legend(handles=legend_items, loc="lower center", fontsize=8.5,
              framealpha=0.88, edgecolor="#cccccc",
              bbox_to_anchor=(0.48, 0.01), ncol=2)

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "network_architecture.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
