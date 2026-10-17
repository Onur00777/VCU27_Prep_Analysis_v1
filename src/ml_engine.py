"""IsolationForest anomali taramasi ve darboaz siralamasi."""

from __future__ import annotations

import numpy as np
import pandas as pd

from schema import CORNERS


def run_ml(df: pd.DataFrame, analysis: dict) -> dict:
    bottlenecks = _bottlenecks(analysis)
    anomalies = _anomalies(df)
    return {"bottlenecks": bottlenecks, "anomalies": anomalies}


def _bottlenecks(analysis: dict) -> list[dict]:
    hv = analysis["hv"]
    lv = analysis["lv"]
    slip = analysis["slip"]
    thermal = analysis["thermal"]
    tv = analysis["tv"]
    apps = analysis["apps"]["bits"]
    segments = analysis["segments"]
    items = []

    if hv["exceed_s"] > 0:
        over_kw = max(0.0, hv["peak_w"] / 1000.0 - 80.0)
        items.append(_item("80 kW aşımı", hv["exceed_s"] * (1.0 + over_kw), f"{hv['exceed_s']*1000:.0f} ms, pik {hv['peak_w']/1000:.1f} kW"))
    if hv["early_derate_s"] > 0.05:
        items.append(_item("Erken güç kısma", hv["early_derate_s"] * 1.4, f"{hv['early_derate_s']*1000:.0f} ms, limitin altında scale < 0.90"))
    spin_s = sum(ch["spin_s"] for ch in slip["wheels"].values())
    if spin_s > 0.05:
        worst = max(slip["wheels"].items(), key=lambda kv: kv[1]["peak_slip"])
        items.append(_item("Tekerlek patinajı", spin_s * (1.0 + max(0.0, worst[1]["peak_slip"])), f"{worst[0].upper()} pik slip {worst[1]['peak_slip']:.2f}, toplam {spin_s:.2f} s"))
    lock_s = sum(ch["lock_s"] for ch in slip["wheels"].values())
    if lock_s > 0.05:
        items.append(_item("Tekerlek kilitlenmesi", lock_s * 1.2, f"Toplam {lock_s:.2f} s negatif slip"))
    if tv["understeer_fraction"] > 0.25 and tv["corner_s"] > 0.4:
        items.append(_item("Understeer", tv["corner_s"] * tv["understeer_fraction"], f"Viraj örneklerinin %{tv['understeer_fraction']*100:.0f}'i"))
    if tv["oversteer_fraction"] > 0.2 and tv["corner_s"] > 0.4:
        items.append(_item("Oversteer", tv["corner_s"] * tv["oversteer_fraction"] * 1.3, f"Viraj örneklerinin %{tv['oversteer_fraction']*100:.0f}'i"))
    if thermal["imbalance_warn"]:
        items.append(_item("İnverter sıcaklık farkı", thermal["imbalance_max_c"] / 8.0, f"Tepe fark {thermal['imbalance_max_c']:.1f} °C"))
    derate_s = sum(ch["derate_s"] for ch in thermal["channels"].values())
    if derate_s > 0.05:
        items.append(_item("Termal tork kısma", derate_s * 1.5, f"{derate_s:.2f} s sıcakken tork geri beslemesi komutun %70'inin altında"))
    if lv.get("available") and lv["brownout_s"] > 0.02:
        items.append(_item("24 V brownout", lv["brownout_s"] * 3.0, f"min {lv['min_v']:.2f} V, {lv['brownout_s']*1000:.0f} ms"))
    if apps["implaus_s"] + apps["deviation_s"] + apps["brake_cut_s"] > 0.05:
        items.append(
            _item(
                "APPS kesmesi",
                (apps["implaus_s"] + apps["brake_cut_s"]) * 2.0 + apps["deviation_s"],
                f"sapma {apps['deviation_s']*1000:.0f} ms, implaus {apps['implaus_s']*1000:.0f} ms, fren-gaz {apps['brake_cut_s']*1000:.0f} ms",
            )
        )
    if segments["coast_candidate_s"] > 0.4:
        items.append(_item("Coasting kaybı", segments["coast_candidate_s"] * 0.6, f"{segments['coast_candidate_s']:.2f} s gazsız ve frensiz"))
    if not items:
        items.append(_item("Belirgin darboğaz yok", 0.0, "Eşikler aşılmadı"))
    items.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(items, start=1):
        row["rank"] = rank
    return items


def _item(name: str, score: float, detail: str) -> dict:
    return {"name": name, "score": float(score), "detail": detail}


def _anomalies(df: pd.DataFrame) -> dict:
    columns = [
        "pack_v",
        "pack_i",
        "power_w",
        "pwr_scale",
        "apps_pct",
        "steer_deg",
        "yaw_dps",
        "vx_mps",
        "lv_v",
        "brake_f_bar",
        "brake_r_bar",
    ]
    columns += [f"igbt_c_{c}" for c in CORNERS]
    columns += [f"motor_c_{c}" for c in CORNERS]
    columns += [f"slip_{c}" for c in CORNERS]
    use = [c for c in columns if c in df.columns]
    frame = df[use].replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 80:
        return {"available": False, "reason": "Anomali modeli için satır sayısı yetersiz.", "events": [], "drivers": []}
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"available": False, "reason": "scikit-learn kurulu değil.", "events": [], "drivers": []}

    scaler = StandardScaler()
    matrix = scaler.fit_transform(frame.to_numpy(dtype=float))
    model = IsolationForest(n_estimators=200, contamination=0.03, random_state=27, n_jobs=1)
    pred = model.fit_predict(matrix)
    flagged = pred == -1
    times = df.loc[frame.index, "time_s"].to_numpy(dtype=float)
    z = np.abs(matrix)
    z_mean = z[flagged].mean(axis=0) if flagged.any() else np.zeros(len(use))
    order = np.argsort(z_mean)[::-1][:5]
    drivers = [{"signal": use[i], "z": float(z_mean[i])} for i in order if z_mean[i] > 0]
    events = []
    if flagged.any():
        mask = np.zeros(len(df), dtype=bool)
        mask[frame.index.to_numpy()] = flagged
        events = _clusters(df["time_s"].to_numpy(dtype=float), mask)
    return {
        "available": True,
        "count": int(flagged.sum()),
        "fraction": float(flagged.mean()),
        "drivers": drivers,
        "events": events[:8],
        "score_times": times[flagged][:400].tolist(),
    }


def _clusters(time_s: np.ndarray, mask: np.ndarray) -> list[dict]:
    clusters = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            clusters.append({"t0_s": float(time_s[start]), "t1_s": float(time_s[i - 1])})
            start = None
    if start is not None:
        clusters.append({"t0_s": float(time_s[start]), "t1_s": float(time_s[-1])})
    clusters.sort(key=lambda item: item["t1_s"] - item["t0_s"], reverse=True)
    return clusters
