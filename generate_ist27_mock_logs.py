"""IST27 mock VCU CSV. Sutunlar datalogger.c basligi ile ayni; LV/piezo ek sutun."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from schema import signals  # noqa: E402

LOG_DIR = ROOT / "logs_input"
R_WHEEL = 0.20574
WHEELBASE = 1.530
STEER_RATIO = 5.5

FIRMWARE_COLUMNS = [spec["csv"] for spec in signals() if spec.get("logged")]
EXTRA_COLUMNS = ["lv_v_10mV", "lv_i_10mA", "apps_raw1", "apps_raw2", "piezo1", "piezo2", "ax_cmps2", "ay_cmps2"]


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    clean = build_clean_lap()
    fault = build_fault_session()
    write_csv(LOG_DIR / "IST27_mock_clean_lap.csv", clean)
    write_csv(LOG_DIR / "IST27_mock_fault_session.csv", fault)
    print(f"Yazildi: {LOG_DIR / 'IST27_mock_fault_session.csv'} ({len(fault['time_ms'])} satir)")
    print(f"Yazildi: {LOG_DIR / 'IST27_mock_clean_lap.csv'} ({len(clean['time_ms'])} satir)")


def write_csv(path: Path, columns: dict):
    order = FIRMWARE_COLUMNS + EXTRA_COLUMNS
    missing = [name for name in order if name not in columns]
    if missing:
        raise RuntimeError("Eksik sutun: " + ", ".join(missing))
    lines = [";".join(order)]
    n = len(columns["time_ms"])
    for i in range(n):
        cells = [str(int(round(float(columns[name][i])))) for name in order]
        lines.append(";".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fault_session() -> dict:
    rng = np.random.default_rng(27)
    t = np.arange(0.0, 26.0, 0.01)
    n = len(t)
    steer = np.interp(t, [0, 8, 9.2, 12.2, 14.6, 15.6, 26], [0, 0, -140, -80, -20, 0, 0])
    apps = np.zeros(n)
    apps[(t >= 3.0) & (t < 7.8)] = np.interp(t[(t >= 3.0) & (t < 7.8)], [3, 7.8], [0, 100])
    apps[(t >= 7.8) & (t < 8.5)] = 100
    apps[(t >= 8.5) & (t < 13.7)] = 90
    apps[(t >= 10.20) & (t < 10.35)] = 0
    apps[(t >= 13.7) & (t < 13.9)] = np.interp(t[(t >= 13.7) & (t < 13.9)], [13.7, 13.9], [90, 0])
    apps[(t >= 17.2) & (t < 18.5)] = 40
    apps[(t >= 18.5) & (t < 19.2)] = 75
    apps[(t >= 19.2) & (t < 21.45)] = 55
    apps[(t >= 21.45) & (t < 22.0)] = 80

    vx = np.interp(t, [0, 3, 8, 12.2, 14.6, 16.4, 21.4, 22.3, 26], [0, 0, 32, 24, 20, 11, 15, 0, 0])
    factor = np.ones((n, 4))
    exit_mask = (t >= 12.2) & (t < 13.6)
    factor[exit_mask, 3] = 1.25

    state = np.ones(n, dtype=int)
    state[t < 1.0] = 2
    state[(t >= 1.0) & (t < 2.0)] = 3
    state[(t >= 2.0) & (t < 22.0)] = 4
    state[t >= 22.0] = 5
    mode = np.where(t >= 2.0, 2, 0)
    status = np.zeros(n, dtype=int)
    status[(t >= 10.20) & (t < 10.35)] = 0x04 | 0x08

    brake_f = np.zeros(n)
    brake_r = np.zeros(n)
    braking = (t >= 15.0) & (t < 16.3)
    brake_f[braking] = 42
    brake_r[braking] = 26

    scale = np.full(n, 1000.0)
    scale[(t >= 7.75) & (t < 11.0)] = 960
    scale[(t >= 18.5) & (t < 18.9)] = 620

    pack_i = 8.0 + 0.55 * apps
    pack_i[(t >= 7.55) & (t < 7.75)] = 141.0
    pack_i[(t >= 7.75) & (t < 11.0)] = 135.7
    pack_i[(t >= 15.05) & (t < 16.2)] = -26.0
    pack_i[(t >= 18.5) & (t < 18.9)] = 72.0
    spike = (t >= 21.55) & (t < 22.0)
    pack_i[spike] = np.interp(t[spike], [21.55, 21.85, 22.0], [80, 250, 40])
    pack_i[t >= 22.0] = 0.0
    pack_v = 596.0 - 0.09 * pack_i + rng.normal(0, 0.12, n)

    lv_v = np.full(n, 26.8) + rng.normal(0, 0.05, n)
    lv_i = np.full(n, 3.5)
    dip = (t >= 16.4) & (t < 17.6)
    lv_v[dip] = 20.4
    lv_i[dip] = 22
    pre = (t >= 21.5) & (t < 22.3)
    lv_v[pre] = 20.15
    lv_i[pre] = 24
    lv_v[t >= 22.6] = 26.4
    lv_i[t >= 22.6] = 2.0

    sdc = np.full(n, 0x1F, dtype=int)
    sdc[t >= 22.0] = 0x1E
    fault = np.zeros(n, dtype=int)
    fault[t >= 22.0] = 1

    return _assemble(t, steer, apps, vx, factor, state, mode, status, brake_f, brake_r, scale, pack_v, pack_i, lv_v, lv_i, sdc, fault, rng, hot_rr=True)


def build_clean_lap() -> dict:
    rng = np.random.default_rng(28)
    t = np.arange(0.0, 18.0, 0.01)
    n = len(t)
    steer = np.interp(t, [0, 6, 8, 11, 13, 18], [0, 0, -55, -40, 0, 0])
    apps = np.zeros(n)
    apps[(t >= 1.5) & (t < 6)] = np.interp(t[(t >= 1.5) & (t < 6)], [1.5, 6], [0, 85])
    apps[(t >= 6) & (t < 12.2)] = 70
    apps[(t >= 12.2) & (t < 12.5)] = np.interp(t[(t >= 12.2) & (t < 12.5)], [12.2, 12.5], [70, 0])
    vx = np.interp(t, [0, 1.5, 7.5, 12, 14.5, 18], [0, 0, 30, 22, 14, 8])
    factor = np.ones((n, 4))
    state = np.full(n, 4, dtype=int)
    state[t < 0.8] = 3
    mode = np.full(n, 2, dtype=int)
    status = np.zeros(n, dtype=int)
    brake_f = np.zeros(n)
    brake_r = np.zeros(n)
    braking = (t >= 12.6) & (t < 14.2)
    brake_f[braking] = 28
    brake_r[braking] = 16
    scale = np.full(n, 980.0)
    scale[(t >= 4.5) & (t < 10.5)] = 970
    pack_i = 20.0 + 0.35 * apps
    pack_i[(t >= 4.5) & (t < 10.5)] = 135.7
    pack_i[braking] = -18.0
    pack_v = 596.0 - 0.09 * pack_i
    lv_v = np.full(n, 26.7)
    lv_i = np.full(n, 4.0)
    sdc = np.full(n, 0x1F, dtype=int)
    fault = np.zeros(n, dtype=int)
    return _assemble(t, steer, apps, vx, factor, state, mode, status, brake_f, brake_r, scale, pack_v, pack_i, lv_v, lv_i, sdc, fault, rng, hot_rr=False, tv_lag=0.04, tv_gain=0.20)


def _assemble(t, steer, apps, vx, factor, state, mode, status, brake_f, brake_r, scale, pack_v, pack_i, lv_v, lv_i, sdc, fault, rng, hot_rr: bool, tv_lag: float = 0.18, tv_gain: float = 0.12):
    n = len(t)
    delay = int(round(tv_lag / 0.01))
    steer_d = np.concatenate([np.zeros(delay), steer[:-delay]]) if delay else steer.copy()
    tq_delta = np.where(np.abs(steer) > 5, -tv_gain * steer_d, 0.0)
    tq_delta[state != 4] = 0.0
    per = np.clip(apps / 100.0 * 21.0 * (scale / 1000.0), 0, 21)
    per[state != 4] = 0.0
    cmd = np.column_stack([per, per, per, per])
    cmd[:, 0] -= tq_delta / 4.0
    cmd[:, 2] -= tq_delta / 4.0
    cmd[:, 1] += tq_delta / 4.0
    cmd[:, 3] += tq_delta / 4.0
    cmd = np.clip(cmd, -21, 21)

    vw = vx[:, None] * factor
    rpm = vw / R_WHEEL * 60.0 / (2.0 * np.pi)
    vref = vw.mean(axis=1)
    den = np.maximum(np.abs(vref), 1.0)
    slip = (vw - vref[:, None]) / den[:, None]
    slip[vx < 1.0] = 0.0

    fb = cmd * 0.98
    igbt = np.column_stack([
        49 + 3 * np.sin(t / 3),
        50 + 2 * np.sin(t / 2.5),
        56 + 2 * np.sin(t / 4),
        57 + 2 * np.sin(t / 4),
    ])
    if hot_rr:
        heat = (t >= 12.2) & (t < 15.5)
        igbt[heat, 3] = np.interp(t[heat], [12.2, 13.4, 15.5], [60, 93, 78])
        fb[heat, 3] = cmd[heat, 3] * 0.72
    motor = igbt - 16.0
    inv_state = np.where(state == 4, 4, 1).astype(int)
    inv = np.column_stack([inv_state, inv_state, inv_state, inv_state])

    delta = np.deg2rad(steer / STEER_RATIO)
    yaw_kin = np.rad2deg(vx * np.tan(delta) / WHEELBASE)
    yaw = 0.95 * yaw_kin
    if hot_rr:
        yaw[(t >= 8.0) & (t < 12.2)] = 0.72 * yaw_kin[(t >= 8.0) & (t < 12.2)]
        yaw[(t >= 12.2) & (t < 13.8)] = 1.40 * yaw_kin[(t >= 12.2) & (t < 13.8)]
    yaw[vx < 3] = 0.0

    power = pack_v * pack_i
    ax = np.gradient(vx, t)
    ay = np.deg2rad(yaw) * vx
    piezo1 = 180 + rng.normal(0, 25, n)
    piezo2 = 160 + rng.normal(0, 20, n)
    burst = (t >= 9.15) & (t < 9.55)
    piezo1[burst] += 1800 * np.sin(2 * np.pi * 38 * t[burst])
    piezo2[burst] += 900 * np.sin(2 * np.pi * 38 * t[burst])

    raw1 = _raw_from_pct(np.clip(apps, 0, 100))
    raw2 = raw1 + 180
    if hot_rr:
        pushed = (t >= 10.20) & (t < 10.35)
        raw1[pushed] = _raw_from_pct(np.full(pushed.sum(), 88))
        raw2[pushed] = _raw_from_pct(np.full(pushed.sum(), 70))

    share = np.clip(cmd, 0, None)
    share_sum = share.sum(axis=1, keepdims=True)
    share_sum[share_sum < 0.1] = 1.0
    dc_i = share / share_sum * pack_i[:, None]

    cols = {
        "time_ms": np.round(t * 1000).astype(int),
        "state": state,
        "drive_mode": mode,
        "apps_status": status,
        "apps_pct": np.round(apps).astype(int),
        "steer_cdeg": np.round(steer * 100),
        "yaw_cdps": np.round(yaw * 100),
        "pack_v_10mV": np.round(pack_v * 100),
        "pack_i_10mA": np.round(pack_i * 100),
        "pack_power_w": np.round(np.clip(power, 0, None)),
        "pwr_scale_permille": np.round(scale),
        "brake_f_bar": np.round(brake_f),
        "brake_r_bar": np.round(brake_r),
        "sdc_flags": sdc,
        "fault_reason": fault,
        "lv_v_10mV": np.round(lv_v * 100),
        "lv_i_10mA": np.round(lv_i * 100),
        "apps_raw1": raw1,
        "apps_raw2": raw2,
        "piezo1": np.round(piezo1),
        "piezo2": np.round(piezo2),
        "ax_cmps2": np.round(ax * 100),
        "ay_cmps2": np.round(ay * 100),
    }
    names = ("fl", "fr", "rl", "rr")
    mapping = {
        "tq_cmd_cNm": np.round(cmd * 100),
        "tq_fb_cNm": np.round(fb * 100),
        "rpm": np.round(rpm),
        "slip_permille": np.round(slip * 1000),
        "dc_v_10mV": np.round(pack_v[:, None] * 100 * np.ones((1, 4))),
        "dc_i_10mA": np.round(dc_i * 100),
        "igbt_cdeg": np.round(igbt * 100),
        "motor_cdeg": np.round(motor * 100),
        "inv_state": inv,
    }
    for prefix, matrix in mapping.items():
        for index, name in enumerate(names):
            cols[f"{prefix}_{name}"] = matrix[:, index]
    return cols


def _raw_from_pct(pct: np.ndarray) -> np.ndarray:
    sensor = 10.0 + 0.9 * pct
    return np.round(8000 + sensor / 100.0 * (52000 - 8000)).astype(int)


if __name__ == "__main__":
    main()
