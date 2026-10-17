"""Plotly figurleri. Koyu tema app tarafinda acilir."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

from schema import CORNERS

COLORS = {
    "fl": "#4CC9F0",
    "fr": "#F72585",
    "rl": "#90E0A0",
    "rr": "#FFB703",
    "inv1": "#4CC9F0",
    "inv2": "#FFB703",
    "power": "#FF4B4B",
    "lv": "#C77DFF",
    "steer": "#FFD166",
    "apps": "#06D6A0",
    "brake": "#EF476F",
}


def _base(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#0B0E14",
        margin=dict(l=48, r=24, t=56, b=40),
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#243041")
    fig.update_yaxes(showgrid=True, gridcolor="#243041")
    return fig


def time_series(df: pd.DataFrame, columns: list[tuple[str, str, str]], title: str, hlines=None) -> go.Figure:
    fig = go.Figure()
    for column, name, color in columns:
        if column not in df.columns:
            continue
        fig.add_trace(go.Scatter(x=df["time_s"], y=df[column], name=name, line=dict(color=color, width=1.6)))
    for y, label, color in hlines or []:
        fig.add_hline(y=y, line_dash="dash", line_color=color, annotation_text=label, annotation_position="right")
    return _base(fig, title)


def four_corners(df: pd.DataFrame, prefix: str, title: str, scale: float = 1.0) -> go.Figure:
    cols = [(f"{prefix}_{c}", c.upper(), COLORS[c]) for c in CORNERS]
    fig = time_series(df, cols, title)
    if scale != 1.0:
        fig.update_traces(y=None)
    return fig


def torque_rpm(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Tork komutu / geri bildirim", "Devir"))
    for corner in CORNERS:
        fig.add_trace(
            go.Scatter(x=df["time_s"], y=df[f"tq_cmd_{corner}"], name=f"{corner.upper()} cmd", line=dict(color=COLORS[corner], width=1.4)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["time_s"], y=df[f"tq_fb_{corner}"], name=f"{corner.upper()} fb",
                line=dict(color=COLORS[corner], width=1.2, dash="dot"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["time_s"], y=df[f"rpm_{corner}"], name=f"{corner.upper()} rpm", line=dict(color=COLORS[corner], width=1.4), showlegend=False),
            row=2, col=1,
        )
    fig.update_yaxes(title_text="Nm", row=1, col=1)
    fig.update_yaxes(title_text="rpm", row=2, col=1)
    fig.update_xaxes(title_text="s", row=2, col=1)
    return _base(fig, "4 tekerlek tork ve devir")


def slip_fig(df: pd.DataFrame) -> go.Figure:
    fig = time_series(
        df,
        [(f"slip_{c}", c.upper(), COLORS[c]) for c in CORNERS],
        "Slip oranı  ( + patinaj, − kilitlenme )",
        hlines=[(0.08, "+0.08", "#FFB703"), (-0.08, "−0.08", "#4CC9F0")],
    )
    return fig


def inverter_fig(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("IGBT", "DC akım veya mekanik güç"))
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["inv1_igbt_c"], name="Inverter_1 IGBT", line=dict(color=COLORS["inv1"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["inv2_igbt_c"], name="Inverter_2 IGBT", line=dict(color=COLORS["inv2"])), row=1, col=1)
    use_dc = float(np.nanmax(np.abs(df[[f"dc_i_{c}" for c in CORNERS]].to_numpy(dtype=float)))) > 0.5
    if use_dc:
        fig.add_trace(go.Scatter(x=df["time_s"], y=df["dc_i_fl"] + df["dc_i_fr"], name="Inv1 DC akım", line=dict(color=COLORS["inv1"])), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["time_s"], y=df["dc_i_rl"] + df["dc_i_rr"], name="Inv2 DC akım", line=dict(color=COLORS["inv2"])), row=2, col=1)
        fig.update_yaxes(title_text="A", row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=df["time_s"], y=df["mech_w_fl"] + df["mech_w_fr"], name="Inv1 mekanik", line=dict(color=COLORS["inv1"])), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["time_s"], y=df["mech_w_rl"] + df["mech_w_rr"], name="Inv2 mekanik", line=dict(color=COLORS["inv2"])), row=2, col=1)
        fig.update_yaxes(title_text="W", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.add_hline(y=85, line_dash="dash", line_color="#FFB703", row=1, col=1)
    return _base(fig, "2 inverter sıcaklık ve yük")


def power_fig(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Paket gücü", "Paket gerilimi ve akımı"))
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["power_w"] / 1000.0, name="V×I", line=dict(color=COLORS["power"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["pwr_scale"] * 80.0, name="scale × 80 kW", line=dict(color="#8ECAE6", dash="dot")), row=1, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="#FFD166", row=1, col=1)
    fig.add_hline(y=78, line_dash="dot", line_color="#90E0A0", row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["pack_v"], name="V", line=dict(color="#4CC9F0")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["pack_i"], name="I", line=dict(color="#FFB703")), row=2, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="V / A", row=2, col=1)
    return _base(fig, "600 V HV — 80 kW ve gerilim çökmesi")


def lv_fig(df: pd.DataFrame) -> go.Figure | None:
    if "lv_v" not in df.columns:
        return None
    cols = [("lv_v", "LV gerilim", COLORS["lv"])]
    if "lv_i" in df.columns:
        cols.append(("lv_i", "LV akım", "#FFB703"))
    fig = time_series(df, cols, "24 V LV kararlılığı", hlines=[(21.5, "21.5 V", "#FF4B4B"), (24, "24 V", "#90E0A0")])
    return fig


def driver_fig(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Gaz, fren, direksiyon", "Hız"))
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["apps_pct"], name="APPS %", line=dict(color=COLORS["apps"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["brake_f_bar"], name="Ön fren bar", line=dict(color=COLORS["brake"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["steer_deg"], name="Direksiyon °", line=dict(color=COLORS["steer"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["vx_kmh"], name="km/h", line=dict(color="#E8EEF7")), row=2, col=1)
    coast = (df["apps_pct"] < 5) & (df["brake_f_bar"] < 3) & (df["vx_mps"] > 6)
    if coast.any():
        fig.add_trace(
            go.Scatter(x=df.loc[coast, "time_s"], y=df.loc[coast, "vx_kmh"], name="coasting", mode="markers", marker=dict(size=4, color="#C77DFF")),
            row=2, col=1,
        )
    return _base(fig, "Pilotaj izi")


def gg_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df["ay_mps2"] / 9.81,
            y=df["ax_mps2"] / 9.81,
            mode="markers",
            marker=dict(size=4, color=df["time_s"], colorscale="Turbo", showscale=True, colorbar=dict(title="s")),
            name="G-G",
        )
    )
    fig.update_xaxes(title="ay [g]")
    fig.update_yaxes(title="ax [g]")
    return _base(fig, "G-G (ax türev, ay = yaw × vx; IMU accel logda yoksa)")


def piezo_fft(df: pd.DataFrame) -> go.Figure | None:
    if "piezo1" not in df.columns and "piezo2" not in df.columns:
        return None
    dt = float(np.median(np.diff(df["time_s"]))) if len(df) > 3 else 0.01
    if dt <= 0:
        return None
    fs = 1.0 / dt
    fig = go.Figure()
    for column, color in (("piezo1", "#4CC9F0"), ("piezo2", "#FFB703")):
        if column not in df.columns:
            continue
        values = df[column].fillna(0).to_numpy(dtype=float)
        if len(values) < 64:
            continue
        freq, power = signal.welch(values - np.mean(values), fs=fs, nperseg=min(512, len(values)))
        fig.add_trace(go.Scatter(x=freq, y=power, name=column, line=dict(color=color)))
    fig.update_xaxes(title="Hz", range=[0, min(80, fs / 2)])
    fig.update_yaxes(title="PSD", type="log")
    return _base(fig, "Piezo FFT")


def blackbox_fig(window: pd.DataFrame, t_event: float) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Güç ve LV", "Gaz / SDC ucu", "RR slip"))
    fig.add_trace(go.Scatter(x=window["time_s"], y=window["power_w"] / 1000.0, name="kW", line=dict(color=COLORS["power"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=window["time_s"], y=window["pack_i"], name="HV A", line=dict(color="#FFB703")), row=1, col=1)
    if "lv_v" in window.columns:
        fig.add_trace(go.Scatter(x=window["time_s"], y=window["lv_v"], name="LV V", line=dict(color=COLORS["lv"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=window["time_s"], y=window["apps_pct"], name="APPS", line=dict(color=COLORS["apps"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=window["time_s"], y=window["sdc_end_closed"].astype(float) * 100, name="SDC kapalı ×100", line=dict(color="#E8EEF7")), row=2, col=1)
    if "slip_rr" in window.columns:
        fig.add_trace(go.Scatter(x=window["time_s"], y=window["slip_rr"], name="slip RR", line=dict(color=COLORS["rr"])), row=3, col=1)
    for row in (1, 2, 3):
        fig.add_vline(x=t_event, line_dash="dash", line_color="#FF4B4B", row=row, col=1)
    return _base(fig, "Arıza öncesi 500 ms")
