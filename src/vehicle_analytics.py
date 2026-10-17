"""4WD, TV, slip, HV 600 V, LV 24 V ve shutdown blackbox hesabi."""

from __future__ import annotations

import numpy as np
import pandas as pd

from schema import CORNERS, bit_labels, label, vehicle

INV1 = ("fl", "fr")
INV2 = ("rl", "rr")


def analyze(df: pd.DataFrame) -> dict:
    cfg = vehicle()
    work = _enrich(df, cfg)
    dt = _dt(work)
    return {
        "df": work,
        "session": _session(work, dt, cfg),
        "tv": _tv(work, dt, cfg),
        "slip": _slip(work, dt, cfg),
        "thermal": _thermal(work, dt, cfg),
        "hv": _hv(work, dt, cfg),
        "lv": _lv(work, dt, cfg),
        "apps": _apps(work, cfg),
        "faults": _faults(work, cfg),
        "segments": _segments(work),
    }


def _enrich(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    geo = cfg.get("geometry", {})
    radius = float(geo.get("wheel_radius_m", 0.20574))
    track = float(geo.get("track_width_m", 1.22))
    wheelbase = float(geo.get("wheelbase_m", 1.53))
    ratio = float(geo.get("steer_ratio_handwheel_per_road", 5.5))

    out["power_w"] = out["pack_v"] * out["pack_i"]
    for corner in CORNERS:
        out[f"wheel_mps_{corner}"] = out[f"rpm_{corner}"] * (2.0 * np.pi / 60.0) * radius
        out[f"mech_w_{corner}"] = out[f"tq_fb_{corner}"] * out[f"rpm_{corner}"] * (2.0 * np.pi / 60.0)
    out["vx_mps"] = out[[f"wheel_mps_{c}" for c in CORNERS]].mean(axis=1)
    out["vx_kmh"] = out["vx_mps"] * 3.6
    den = out["vx_mps"].abs().clip(lower=1.0)
    for corner in CORNERS:
        out[f"slip_re_{corner}"] = (out[f"wheel_mps_{corner}"] - out["vx_mps"]) / den
    out["tq_left"] = out["tq_cmd_fl"] + out["tq_cmd_rl"]
    out["tq_right"] = out["tq_cmd_fr"] + out["tq_cmd_rr"]
    out["tq_delta_nm"] = out["tq_right"] - out["tq_left"]
    out["yaw_moment_nm"] = out["tq_delta_nm"] * track / (2.0 * radius)
    out["steer_road_deg"] = out["steer_deg"] / ratio
    delta = np.deg2rad(out["steer_road_deg"])
    out["yaw_kin_dps"] = np.rad2deg(out["vx_mps"] * np.tan(delta) / wheelbase)
    out["inv1_igbt_c"] = out[[f"igbt_c_{c}" for c in INV1]].max(axis=1)
    out["inv2_igbt_c"] = out[[f"igbt_c_{c}" for c in INV2]].max(axis=1)
    out["inv1_motor_c"] = out[[f"motor_c_{c}" for c in INV1]].max(axis=1)
    out["inv2_motor_c"] = out[[f"motor_c_{c}" for c in INV2]].max(axis=1)
    out["mech_w"] = out[[f"mech_w_{c}" for c in CORNERS]].sum(axis=1)
    if "ax_mps2" not in out.columns:
        out["ax_mps2"] = np.gradient(out["vx_mps"].to_numpy(dtype=float), out["time_s"].to_numpy(dtype=float))
    if "ay_mps2" not in out.columns:
        out["ay_mps2"] = np.deg2rad(out["yaw_dps"]) * out["vx_mps"]
    out["sdc_end_closed"] = (out["sdc_flags"].fillna(0).astype(int) & 0x01) != 0
    out["sdc_imd_ok"] = (out["sdc_flags"].fillna(0).astype(int) & 0x04) != 0
    out["sdc_ams_ok"] = (out["sdc_flags"].fillna(0).astype(int) & 0x08) != 0
    return out


def _trap(y, x) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def _dt(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.01
    diffs = np.diff(df["time_s"].to_numpy(dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 0.01


def _session(df: pd.DataFrame, dt: float, cfg: dict) -> dict:
    power = df["power_w"].to_numpy(dtype=float)
    t = df["time_s"].to_numpy(dtype=float)
    vx = df["vx_mps"].to_numpy(dtype=float)
    distance = _trap(np.clip(vx, 0, None), t) if len(t) > 1 else 0.0
    e_dis = _trap(np.clip(power, 0, None), t) / 3600.0 if len(t) > 1 else 0.0
    e_regen = _trap(np.clip(-power, 0, None), t) / 3600.0 if len(t) > 1 else 0.0
    return {
        "duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
        "distance_m": distance,
        "vmax_kmh": float(np.nanmax(df["vx_kmh"])) if len(df) else 0.0,
        "pmax_w": float(np.nanmax(power)) if len(power) else 0.0,
        "pmin_w": float(np.nanmin(power)) if len(power) else 0.0,
        "energy_discharge_wh": e_dis,
        "energy_regen_wh": e_regen,
        "regen_ratio": (e_regen / e_dis) if e_dis > 1.0 else 0.0,
        "zero_to_100_s": _zero_to_speed(df, 100.0 / 3.6),
        "dt_s": dt,
        "power_limit_w": float(cfg.get("hv", {}).get("power_limit_w", 80000)),
    }


def _zero_to_speed(df: pd.DataFrame, target: float) -> float | None:
    moving = df["vx_mps"] > 0.5
    if not moving.any():
        return None
    start = float(df.loc[moving, "time_s"].iloc[0])
    hit = df[(df["time_s"] >= start) & (df["vx_mps"] >= target)]
    if hit.empty:
        return None
    return float(hit["time_s"].iloc[0] - start)


def _tv(df: pd.DataFrame, dt: float, cfg: dict) -> dict:
    ratio = float(cfg.get("geometry", {}).get("steer_ratio_handwheel_per_road", 5.5))
    corner = (df["steer_deg"].abs() > float(cfg.get("corner_steer_deg", 15))) & (df["vx_mps"] > 5.0) & (df["state"] == 4)
    kin_ok = corner & (df["yaw_kin_dps"].abs() > 4.0)
    balance = df.loc[kin_ok, "yaw_dps"] / df.loc[kin_ok, "yaw_kin_dps"]
    under = float((balance < 0.85).mean()) if len(balance) else 0.0
    over = float((balance > 1.15).mean()) if len(balance) else 0.0
    lag_s, corr = _lag(df.loc[corner, "steer_deg"], df.loc[corner, "tq_delta_nm"], dt)
    mean_delta = float(df.loc[corner, "tq_delta_nm"].abs().mean()) if corner.any() else 0.0
    return {
        "corner_s": float(corner.sum() * dt),
        "mean_abs_tq_delta_nm": mean_delta,
        "mean_abs_yaw_moment_nm": float(df.loc[corner, "yaw_moment_nm"].abs().mean()) if corner.any() else 0.0,
        "tv_inactive": bool(corner.any() and mean_delta < float(cfg.get("tv_inactive_nm", 1.5))),
        "lag_ms": None if lag_s is None else float(lag_s * 1000.0),
        "lag_corr": corr,
        "lag_warn": bool(lag_s is not None and abs(lag_s) * 1000.0 > float(cfg.get("tv_lag_warn_ms", 100))),
        "understeer_fraction": under,
        "oversteer_fraction": over,
        "balance_median": float(balance.median()) if len(balance) else None,
        "steer_ratio_assumed": ratio,
        "samples": int(kin_ok.sum()),
    }


def _lag(x: pd.Series, y: pd.Series, dt: float, max_lag_s: float = 0.4):
    if len(x) < 20 or dt <= 0:
        return None, None
    a = x.to_numpy(dtype=float)
    b = y.to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 20 or np.std(a) < 1e-3 or np.std(b) < 1e-3:
        return None, None
    a = a - a.mean()
    b = b - b.mean()
    max_lag = int(max_lag_s / dt)
    best_lag = 0
    best_corr = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(a[-lag:], b[:lag])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(a[:-lag], b[lag:])[0, 1]
        else:
            corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr) and abs(corr) > abs(best_corr):
            best_corr = float(corr)
            best_lag = lag
    return best_lag * dt, best_corr


def _slip(df: pd.DataFrame, dt: float, cfg: dict) -> dict:
    spin_th = float(cfg.get("slip_spin", 0.08))
    lock_th = float(cfg.get("slip_lock", -0.08))
    wheels = {}
    tc_miss = []
    for corner in CORNERS:
        slip = df[f"slip_{corner}"]
        spin = slip > spin_th
        lock = slip < lock_th
        others = [c for c in CORNERS if c != corner]
        cmd = df[f"tq_cmd_{corner}"]
        peer = df[[f"tq_cmd_{c}" for c in others]].mean(axis=1)
        uncontrolled = spin & (cmd > 3.0) & (cmd > 0.85 * peer.clip(lower=0.1))
        if uncontrolled.any():
            tc_miss.append(corner)
        wheels[corner] = {
            "peak_slip": float(slip.max()) if len(slip) else 0.0,
            "min_slip": float(slip.min()) if len(slip) else 0.0,
            "spin_s": float(spin.sum() * dt),
            "lock_s": float(lock.sum() * dt),
            "tc_not_cutting": bool(uncontrolled.any()),
        }
    return {"wheels": wheels, "tc_miss_corners": tc_miss, "spin_threshold": spin_th, "lock_threshold": lock_th}


def _thermal(df: pd.DataFrame, dt: float, cfg: dict) -> dict:
    igbt_warn = float(cfg.get("igbt_warn_c", 85))
    igbt_crit = float(cfg.get("igbt_crit_c", 100))
    motor_warn = float(cfg.get("motor_warn_c", 110))
    imbalance_th = float(cfg.get("thermal_imbalance_c", 8))
    channels = {}
    for corner in CORNERS:
        igbt = df[f"igbt_c_{corner}"]
        motor = df[f"motor_c_{corner}"]
        cmd = df[f"tq_cmd_{corner}"]
        fb = df[f"tq_fb_{corner}"]
        hot = (igbt > igbt_warn) | (motor > motor_warn)
        derate = hot & (cmd > 5.0) & (fb < 0.7 * cmd)
        channels[corner] = {
            "igbt_max_c": float(igbt.max()) if len(igbt) else 0.0,
            "motor_max_c": float(motor.max()) if len(motor) else 0.0,
            "igbt_warn_s": float((igbt > igbt_warn).sum() * dt),
            "igbt_crit_s": float((igbt > igbt_crit).sum() * dt),
            "derate_s": float(derate.sum() * dt),
        }
    gap = (df["inv1_igbt_c"] - df["inv2_igbt_c"]).abs()
    dc_i_span = float(np.nanmax(np.abs(df[[f"dc_i_{c}" for c in CORNERS]].to_numpy(dtype=float))))
    return {
        "channels": channels,
        "inv1_igbt_max_c": float(df["inv1_igbt_c"].max()) if len(df) else 0.0,
        "inv2_igbt_max_c": float(df["inv2_igbt_c"].max()) if len(df) else 0.0,
        "imbalance_max_c": float(gap.max()) if len(gap) else 0.0,
        "imbalance_mean_c": float(gap.mean()) if len(gap) else 0.0,
        "imbalance_warn": bool(len(gap) and gap.max() > imbalance_th),
        "dc_current_logged": bool(dc_i_span > 0.5),
        "dc_voltage_saturated": _dc_saturated(df),
        "dc_voltage_empty": _dc_empty(df),
    }


def _dc_saturated(df: pd.DataFrame) -> bool:
    """int16 tavanı 327.67 V. 580 V gibi gercek degerler tasma degildir."""
    volts = df[[f"dc_v_{c}" for c in CORNERS]]
    return bool(((volts.abs() - 327.67).abs() < 0.2).any().any())


def _dc_empty(df: pd.DataFrame) -> bool:
    return bool(df[[f"dc_v_{c}" for c in CORNERS]].abs().max().max() < 1.0)


def _hv(df: pd.DataFrame, dt: float, cfg: dict) -> dict:
    hv = cfg.get("hv", {})
    limit = float(hv.get("power_limit_w", 80000))
    hold_lo = float(hv.get("hold_low_w", 78000))
    hold_hi = float(hv.get("hold_high_w", 79900))
    early_below = float(hv.get("early_derate_power_w", 70000))
    full_lo = float(hv.get("full_soc_min_v", 580))
    full_hi = float(hv.get("full_soc_max_v", 600))
    power = df["power_w"]
    exceed = power > limit
    moving = (df["state"] == 4) & (df["apps_pct"] >= 70) & (df["vx_mps"] > 5) & (df["brake_f_bar"] < 5)
    limited = moving & (df["pwr_scale"] < 0.98)
    in_band = limited & (power >= hold_lo) & (power <= hold_hi)
    early = moving & (df["pwr_scale"] < 0.90) & (power < early_below) & (power > 5000)
    late = exceed
    rint, voc, n_fit = _rint(df, float(hv.get("sag_i_min_a", 20)))
    events = _runs(df["time_s"], exceed, dt)
    return {
        "limit_w": limit,
        "exceed_s": float(exceed.sum() * dt),
        "exceed_events": events,
        "peak_w": float(power.max()) if len(power) else 0.0,
        "hold_fraction": float(in_band.sum() / limited.sum()) if limited.any() else None,
        "limited_s": float(limited.sum() * dt),
        "early_derate_s": float(early.sum() * dt),
        "late_cut_s": float(late.sum() * dt),
        "vmin_v": float(df["pack_v"].min()) if len(df) else 0.0,
        "vmax_v": float(df["pack_v"].max()) if len(df) else 0.0,
        "full_band": [full_lo, full_hi],
        "below_full_s": float((df["pack_v"] < full_lo).sum() * dt),
        "rint_ohm": rint,
        "voc_v": voc,
        "rint_samples": n_fit,
        "cell_vmin_v": float(df["cell_vmin"].min()) if "cell_vmin" in df.columns and df["cell_vmin"].notna().any() else None,
        "logged_power_drops_regen": bool((df["pack_power_w_logged"].fillna(0) == 0).any() and (power < -500).any()),
    }


def _rint(df: pd.DataFrame, i_min: float):
    mask = (df["pack_i"] > i_min) & (df["power_w"] > 0) & df["pack_v"].notna()
    if int(mask.sum()) < 30:
        return None, None, int(mask.sum())
    current = df.loc[mask, "pack_i"].to_numpy(dtype=float)
    voltage = df.loc[mask, "pack_v"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(current, voltage, 1)
    return float(-slope), float(intercept), int(mask.sum())


def _runs(time_s: pd.Series, mask: pd.Series, dt: float) -> list[dict]:
    values = mask.fillna(False).to_numpy(dtype=bool)
    times = time_s.to_numpy(dtype=float)
    events = []
    start = None
    for i, flag in enumerate(values):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            events.append({"t0_s": float(times[start]), "t1_s": float(times[i - 1]), "duration_s": float((i - start) * dt)})
            start = None
    if start is not None:
        events.append({"t0_s": float(times[start]), "t1_s": float(times[-1]), "duration_s": float((len(values) - start) * dt)})
    return events[:12]


def _lv(df: pd.DataFrame, dt: float, cfg: dict) -> dict:
    lv = cfg.get("lv", {})
    brownout = float(lv.get("brownout_v", 21.5))
    if "lv_v" not in df.columns or df["lv_v"].notna().sum() < 5:
        return {"available": False, "brownout_v": brownout, "events": [], "min_v": None, "brownout_s": 0.0}
    low = df["lv_v"] < brownout
    return {
        "available": True,
        "brownout_v": brownout,
        "min_v": float(df["lv_v"].min()),
        "max_v": float(df["lv_v"].max()),
        "mean_v": float(df["lv_v"].mean()),
        "brownout_s": float(low.sum() * dt),
        "events": _runs(df["time_s"], low, dt),
        "nominal_band": [float(lv.get("nominal_min_v", 24)), float(lv.get("nominal_max_v", 28))],
    }


def _apps(df: pd.DataFrame, cfg: dict) -> dict:
    apps = cfg.get("apps", {})
    status = df["apps_status"].fillna(0).astype(int)
    bits = {
        "deviation_s": float(((status & 0x04) != 0).sum() * _dt(df)),
        "implaus_s": float(((status & 0x08) != 0).sum() * _dt(df)),
        "brake_cut_s": float(((status & 0x10) != 0).sum() * _dt(df)),
        "oor_s": float(((status & 0x02) != 0).sum() * _dt(df)),
        "latch_s": float(((status & 0x01) != 0).sum() * _dt(df)),
    }
    raw_note = None
    if "apps_raw1" in df.columns and "apps_raw2" in df.columns:
        p1 = df["apps_raw1"].map(lambda v: _pot_percent(v, apps))
        p2 = df["apps_raw2"].map(lambda v: _pot_percent(v, apps))
        both = p1.notna() & p2.notna()
        if both.any():
            dev = (p1 - p2).abs()
            raw_note = {
                "samples": int(both.sum()),
                "max_deviation_pp": float(dev[both].max()),
                "over_10_s": float(((dev > float(apps.get("deviation_pp", 10))) & both).sum() * _dt(df)),
            }
    return {"bits": bits, "raw": raw_note}


def _pot_percent(raw, apps: dict):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return np.nan
    lo = float(apps.get("raw_min", 8000))
    hi = float(apps.get("raw_max", 52000))
    dead = float(apps.get("dead_pct", 10))
    if value < lo or value > hi or hi <= lo:
        return np.nan
    pct = (value - lo) * 100.0 / (hi - lo)
    if pct < dead:
        return 0.0
    return (pct - dead) * 100.0 / (100.0 - dead)


def _faults(df: pd.DataFrame, cfg: dict) -> list[dict]:
    pre_s = float(cfg.get("blackbox_pre_ms", 500)) / 1000.0
    post_s = float(cfg.get("blackbox_post_ms", 200)) / 1000.0
    times = df["time_s"].to_numpy(dtype=float)
    marks = []
    reason = df["fault_reason"].fillna(0).astype(int)
    state = df["state"].fillna(0).astype(int)
    marks += _edges(times, (reason > 0) & (reason.shift(1).fillna(0).astype(int) == 0), "fault_reason")
    marks += _edges(times, (state == 5) & (state.shift(1).fillna(0).astype(int) != 5), "state_fault")
    end = df["sdc_end_closed"].astype(bool)
    imd = df["sdc_imd_ok"].astype(bool)
    ams = df["sdc_ams_ok"].astype(bool)
    marks += _edges(times, (~end) & end.shift(1).fillna(False), "sdc_end_open")
    marks += _edges(times, (~imd) & imd.shift(1).fillna(False), "imd_not_ok")
    marks += _edges(times, (~ams) & ams.shift(1).fillna(False), "ams_not_ok")
    for corner in CORNERS:
        inv = df[f"inv_state_{corner}"].fillna(0).astype(int)
        marks += _edges(times, (inv == 5) & (inv.shift(1).fillna(0).astype(int) != 5), f"inv_error_{corner}")
    status = df["apps_status"].fillna(0).astype(int)
    marks += _edges(times, ((status & 0x19) != 0) & ((status.shift(1).fillna(0).astype(int) & 0x19) == 0), "apps_cut")

    marks.sort(key=lambda item: item["t_s"])
    deduped = []
    for mark in marks:
        if deduped and abs(mark["t_s"] - deduped[-1]["t_s"]) < 0.3 and mark["kind"] in {"fault_reason", "state_fault", "sdc_end_open"}:
            deduped[-1]["also"] = deduped[-1].get("also", []) + [mark["kind"]]
            continue
        deduped.append(mark)

    events = []
    for mark in deduped:
        window = df[(df["time_s"] >= mark["t_s"] - pre_s) & (df["time_s"] <= mark["t_s"] + post_s)].copy()
        before = df[(df["time_s"] >= mark["t_s"] - pre_s) & (df["time_s"] < mark["t_s"])]
        cause, details = _cause(before, df.iloc[(df["time_s"] - mark["t_s"]).abs().argmin()], cfg)
        at = df.iloc[(df["time_s"] - mark["t_s"]).abs().argmin()]
        events.append(
            {
                "t_s": mark["t_s"],
                "kind": mark["kind"],
                "also": mark.get("also", []),
                "fault_reason": label("fault_reason", at["fault_reason"]),
                "state": label("state", at["state"]),
                "sdc": bit_labels("sdc_flags_bits", at["sdc_flags"]),
                "cause": cause,
                "details": details,
                "window": window,
            }
        )
    return events


def _edges(times: np.ndarray, mask: pd.Series, kind: str) -> list[dict]:
    idx = np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool))
    return [{"t_s": float(times[i]), "kind": kind} for i in idx]


def _cause(before: pd.DataFrame, at: pd.Series, cfg: dict) -> tuple[str, list[str]]:
    if before.empty:
        return "Pencere boş", []
    details = []
    scored = []
    i_peak = float(before["pack_i"].max())
    p_peak = float(before["power_w"].max())
    limit = float(cfg.get("hv", {}).get("power_limit_w", 80000))
    if i_peak > 200:
        scored.append((3, f"HV akım piki {i_peak:.0f} A"))
    if p_peak > limit:
        scored.append((3, f"Güç {p_peak/1000:.1f} kW, 80 kW tavanı aşıldı"))
    if "lv_v" in before.columns and before["lv_v"].notna().any():
        lv_min = float(before["lv_v"].min())
        if lv_min < float(cfg.get("lv", {}).get("brownout_v", 21.5)):
            scored.append((4, f"LV {lv_min:.2f} V, 21.5 V brownout altı"))
    status = int(before["apps_status"].fillna(0).astype(int).max()) | int(at["apps_status"])
    if status & 0x1F:
        scored.append((3, "APPS: " + ", ".join(bit_labels("apps_status_bits", status))))
    for corner in CORNERS:
        if (before[f"inv_state_{corner}"].fillna(0).astype(int) == 5).any():
            scored.append((4, f"{corner.upper()} inverter ERROR"))
    if (~before["sdc_imd_ok"]).any() or (not bool(at["sdc_imd_ok"])):
        scored.append((5, "IMD_OK düştü"))
    if (~before["sdc_ams_ok"]).any() or (not bool(at["sdc_ams_ok"])):
        scored.append((5, "AMS_OK düştü"))
    if (~before["sdc_end_closed"]).any() or (not bool(at["sdc_end_closed"])):
        scored.append((5, "SDC ucu açıldı"))
    v_drop = float(before["pack_v"].iloc[0] - before["pack_v"].min()) if len(before) else 0.0
    if v_drop > 15:
        scored.append((2, f"Paket gerilimi {v_drop:.1f} V çöktü"))
    if not scored:
        scored.append((1, f"Kayıtlı sebep {label('fault_reason', at['fault_reason'])}"))
    scored.sort(key=lambda item: item[0], reverse=True)
    details = [text for _, text in scored]
    return details[0], details


def _segments(df: pd.DataFrame) -> dict:
    brake = (df["brake_f_bar"] >= 8) | (df["brake_r_bar"] >= 8)
    corner = df["steer_deg"].abs() > 15
    accel = (df["ax_mps2"] > 1.0) & ~brake & ~corner
    return {
        "accel_s": float((accel).sum() * _dt(df)),
        "brake_s": float(brake.sum() * _dt(df)),
        "corner_s": float(corner.sum() * _dt(df)),
        "coast_candidate_s": float(((df["apps_pct"] < 5) & ~brake & (df["vx_mps"] > 6)).sum() * _dt(df)),
    }
