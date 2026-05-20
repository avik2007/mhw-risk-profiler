"""
export_dashboard_pngs.py — Render static PNGs of the dashboard charts.

Mirrors the exact styling and data from docs/index.html:
  - loss_curve.png     — ERA5 + WN2 training/val loss
  - gate_chart.png     — LeakyGate α over epochs
  - svar_map_era5.png  — SVaR₉₅ map, ERA5 model
  - svar_map_wn2.png   — SVaR₉₅ map, WN2 model

Output: docs/assets/plots/dashboard/
"""
import json, os
import plotly.graph_objects as go

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "docs", "data")
OUT  = os.path.join(BASE, "docs", "assets", "plots", "dashboard")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(DATA, "training.json")) as f:
    training = json.load(f)
with open(os.path.join(DATA, "svar_map.json")) as f:
    svar_map = json.load(f)

ERA5_COLOR = "#1d6fe8"
WN2_COLOR  = "#0ea472"
FONT       = dict(color="#5a6e8c", family="Inter, system-ui, sans-serif", size=13)
GRID       = "#e8eff8"
LINE       = "#dde4f0"
LEGEND     = dict(bgcolor="rgba(255,255,255,.9)", bordercolor=LINE, borderwidth=1)
LAYOUT_BASE = dict(
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    font=FONT,
    xaxis=dict(gridcolor=GRID, linecolor=LINE, tickcolor="#c0cfe4", zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor=LINE, tickcolor="#c0cfe4", zeroline=False),
    legend=LEGEND,
)

W, H = 1200, 600


# ── Loss curve ───────────────────────────────────────────────────────────────
e, w = training["era5"], training["wn2"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=e["epochs"], y=e["train_loss"], name="ERA5 train",
    mode="lines", line=dict(color=ERA5_COLOR, width=2.5)))
fig.add_trace(go.Scatter(x=e["epochs"], y=e["val_loss"], name="ERA5 val",
    mode="lines", line=dict(color=ERA5_COLOR, width=2.5, dash="dot")))
fig.add_trace(go.Scatter(x=w["epochs"], y=w["train_loss"], name="WN2 train",
    mode="lines", line=dict(color=WN2_COLOR, width=2.5)))
fig.add_trace(go.Scatter(x=w["epochs"], y=w["val_loss"], name="WN2 val",
    mode="lines", line=dict(color=WN2_COLOR, width=2.5, dash="dot")))
fig.add_trace(go.Scatter(x=[e["best_epoch"]], y=[e["best_val_loss"]], mode="markers",
    showlegend=False, marker=dict(color=ERA5_COLOR, size=12, symbol="star",
    line=dict(color="#fff", width=1.5))))
fig.add_trace(go.Scatter(x=[w["best_epoch"]], y=[w["best_val_loss"]], mode="markers",
    showlegend=False, marker=dict(color=WN2_COLOR, size=12, symbol="star",
    line=dict(color="#fff", width=1.5))))
fig.update_layout(
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=FONT,
    width=W, height=H,
    margin=dict(l=70, r=30, t=50, b=70),
    title=dict(text="Training & Validation Loss", font=dict(size=17, color="#1a2540"), x=0.5),
    xaxis=dict(gridcolor=GRID, linecolor=LINE, tickcolor="#c0cfe4", zeroline=False,
               title=dict(text="Epoch", standoff=10)),
    yaxis=dict(gridcolor=GRID, linecolor=LINE, tickcolor="#c0cfe4", zeroline=False,
               title=dict(text="MSE Loss (normalised)", standoff=10)),
    legend=dict(**LEGEND, orientation="h", y=-0.22, x=0),
)
path = os.path.join(OUT, "loss_curve.png")
fig.write_image(path, scale=2)
print(f"  wrote {path}")


# ── Gate chart ───────────────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(x=e["epochs"], y=e["gate_mean"],
    name="ERA5  (synthetic ensemble)", mode="lines+markers",
    line=dict(color=ERA5_COLOR, width=2.5), marker=dict(size=7, color=ERA5_COLOR)))
fig.add_trace(go.Scatter(x=w["epochs"], y=w["gate_mean"],
    name="WeatherNext 2 (physical ensemble)", mode="lines+markers",
    line=dict(color=WN2_COLOR, width=2.5), marker=dict(size=7, color=WN2_COLOR)))
fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=0.5, y1=0.5,
    line=dict(color="#c0cfe4", width=1.5, dash="dot"))
fig.add_annotation(x=0.97, y=0.53, xref="paper", yref="y",
    text="← balanced →", showarrow=False, font=dict(color="#c0cfe4", size=11))
fig.update_layout(
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=FONT,
    width=W, height=H,
    margin=dict(l=70, r=30, t=50, b=70),
    title=dict(text="LeakyGate Coefficient (α) — Which Stream Dominates?",
               font=dict(size=17, color="#1a2540"), x=0.5),
    xaxis=dict(gridcolor=GRID, linecolor=LINE, tickcolor="#c0cfe4", zeroline=False,
               title=dict(text="Epoch", standoff=10)),
    yaxis=dict(gridcolor=GRID, linecolor=LINE, tickcolor="#c0cfe4", zeroline=False,
               title=dict(text="α  (0 = atmosphere · 1 = ocean)", standoff=10),
               range=[0, 0.65]),
    legend=dict(**LEGEND, orientation="h", y=-0.22, x=0),
)
path = os.path.join(OUT, "gate_chart.png")
fig.write_image(path, scale=2)
print(f"  wrote {path}")


# ── SVaR maps ────────────────────────────────────────────────────────────────
COLORSCALE = [
    [0.0,  "#dbeafe"],
    [0.3,  "#60a5fa"],
    [0.6,  "#f59e0b"],
    [0.85, "#ef4444"],
    [1.0,  "#7f1d1d"],
]

GEO = dict(
    scope="north america",
    projection=dict(type="mercator"),
    center=dict(lat=43.0, lon=-68.5),
    lonaxis=dict(range=[-73, -64]),
    lataxis=dict(range=[39.5, 46.5]),
    showland=True,    landcolor="#e8f4ea",
    showocean=True,   oceancolor="#dbeafe",
    showcoastlines=True, coastlinecolor="#94a3b8",
    showsubunits=True, subunitcolor="#c0cfe4",
    showrivers=False, showframe=False,
    bgcolor="#ffffff", resolution=50,
)

for source, color, label in [
    ("era5", ERA5_COLOR, "ERA5 Model — reanalysis + synthetic ensemble"),
    ("wn2",  WN2_COLOR,  "WeatherNext 2 — 64-member physical ensemble"),
]:
    cells  = svar_map[source]
    vals   = [c["svar95"] for c in cells]
    vmin, vmax = min(vals), max(vals)
    pad = (vmax - vmin) * 0.1

    fig = go.Figure(go.Scattergeo(
        lon=[c["lon"] for c in cells],
        lat=[c["lat"] for c in cells],
        mode="markers",
        marker=dict(
            size=10, color=vals,
            colorscale=COLORSCALE,
            cmin=vmin - pad, cmax=vmax + pad,
            colorbar=dict(
                title=dict(text="SVaR₉₅<br>[°C·day]", font=dict(color="#5a6e8c", size=13)),
                tickfont=dict(color="#5a6e8c"),
                bgcolor="rgba(255,255,255,.85)",
                bordercolor=LINE, len=0.75,
            ),
            symbol="square",
            line=dict(color="#fff", width=0.8),
            opacity=0.92,
        ),
        text=[f"SVaR₉₅: {c['svar95']:.1f} °C·day" for c in cells],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        width=W, height=700,
        margin=dict(l=0, r=0, t=60, b=0),
        title=dict(
            text=f"<b>{label}</b><br><sup>SVaR₉₅ [°C·day] · Illustrative — real model weights, synthetic inputs</sup>",
            font=dict(size=15, color="#1a2540"), x=0.5,
        ),
        geo=GEO,
    )
    path = os.path.join(OUT, f"svar_map_{source}.png")
    fig.write_image(path, scale=2)
    print(f"  wrote {path}")

print("Done.")
