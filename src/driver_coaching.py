"""Pilot pedal/direksiyon koçluğu. Metinler olculen logdan uretilir."""

from __future__ import annotations

import numpy as np
import pandas as pd

from schema import CORNERS, vehicle


def coach(df: pd.DataFrame, analysis: dict) -> dict:
    cfg = vehicle()
    dt = float(analysis["session"]["dt_s"])
    pedal = _pedal(df, dt, cfg)
    trail = _trail(df, dt)
    coast = _coast(df, dt)
    steer_score = _steer_smooth(df, dt)
    scores = {
        "pedal": pedal["score"],
        "steering": steer_score,
        "trail_brake": trail["score"],
        "coasting": coast["score"],
    }
    scores["overall"] = float(np.mean(list(scores.values())))
    pilot = _pilot_lines(pedal, trail, coast, steer_score, analysis)
    vcu = _vcu_lines(analysis)
    alerts = _alert_lines(analysis)
    events = pedal["events"] + coast["events"] + trail["events"]
    events.sort(key=lambda item: item["t_s"])
    return {"scores": scores, "pilot": pilot, "vcu": vcu, "alerts": alerts, "events": events}


def _pedal(df: pd.DataFrame, dt: float, cfg: dict) -> dict:
    apps = df["apps_pct"].to_numpy(dtype=float)
    steer = df["steer_deg"].to_numpy(dtype=float)
    rate = np.abs(np.diff(apps, prepend=apps[0]) / max(dt, 1e-3))
    moving = df["vx_mps"].to_numpy(dtype=float) > 5
    typical = float(np.percentile(rate[moving], 90)) if moving.any() else 0.0
    score = float(np.clip(100.0 - max(0.0, typical - 80.0) * 0.35, 0, 100))
    spin_th = float(cfg.get("slip_spin", 0.08))
    apps_th = float(cfg.get("corner_exit_apps_pct", 70))
    steer_th = float(cfg.get("corner_steer_deg", 15))
    events = []
    inner = np.where(steer >= 0, df["slip_rl"].to_numpy(dtype=float), df["slip_rr"].to_numpy(dtype=float))
    inner_name = np.where(steer >= 0, "RL", "RR")
    bad = (np.abs(steer) > steer_th) & (apps > apps_th) & (inner > spin_th) & (df["state"].to_numpy() == 4)
    for t0, t1 in _true_runs(df["time_s"].to_numpy(dtype=float), bad):
        if t1 - t0 < 0.15:
            continue
        sl = df[(df["time_s"] >= t0) & (df["time_s"] <= t1)]
        wheel = inner_name[(df["time_s"] - t0).abs().argmin()]
        peak = float(sl[f"slip_{wheel.lower()}"].max())
        events.append(
            {
                "t_s": float(t0),
                "kind": "corner_exit",
                "text": (
                    f"{t0:.2f}–{t1:.2f} s: direksiyon {sl['steer_deg'].mean():.0f}° iken gaz %{sl['apps_pct'].mean():.0f}. "
                    f"İç arka {wheel} kayması {peak:.2f}. Çıkışta gazı tekerlek tutununca aç."
                ),
            }
        )
    return {"score": score, "rms_pct_s": typical, "events": events}


def _steer_smooth(df: pd.DataFrame, dt: float) -> float:
    steer = df["steer_deg"].to_numpy(dtype=float)
    rate = np.diff(steer, prepend=steer[0]) / max(dt, 1e-3)
    moving = df["vx_mps"].to_numpy(dtype=float) > 5
    if not moving.any():
        return 100.0
    rms = float(np.sqrt(np.mean(rate[moving] ** 2)))
    return float(np.clip(100.0 - max(0.0, rms - 40.0) * 0.25, 0, 100))


def _coast(df: pd.DataFrame, dt: float) -> dict:
    mask = (df["apps_pct"] < 5) & (df["brake_f_bar"] < 3) & (df["brake_r_bar"] < 3) & (df["vx_mps"] > 6) & (df["state"] == 4)
    events = []
    coast_s = 0.0
    for t0, t1 in _true_runs(df["time_s"].to_numpy(dtype=float), mask.to_numpy(dtype=bool)):
        dur = t1 - t0
        if dur < 0.2:
            continue
        coast_s += dur
        events.append(
            {
                "t_s": float(t0),
                "kind": "coast",
                "text": f"{t0:.2f}–{t1:.2f} s: {dur:.2f} s boşta (ne gaz ne fren) ve hız > 6 m/s. Bu aralık tahrik de rejenerasyon da üretmiyor.",
            }
        )
    moving_s = float(((df["vx_mps"] > 6) & (df["state"] == 4)).sum() * dt)
    fraction = coast_s / moving_s if moving_s > 0.5 else 0.0
    score = float(np.clip(100.0 - fraction * 180.0, 0, 100))
    return {"score": score, "coast_s": coast_s, "fraction": fraction, "events": events}


def _trail(df: pd.DataFrame, dt: float) -> dict:
    brake = (df["brake_f_bar"] >= 8) | (df["brake_r_bar"] >= 8)
    steer_abs = df["steer_deg"].abs()
    earlier = steer_abs.shift(30).fillna(steer_abs)
    unwinding = steer_abs < (earlier - 3.0)
    trail = brake & unwinding & (df["apps_pct"] < 15) & (df["vx_mps"] > 5)
    brake_s = float(brake.sum() * dt)
    trail_s = float(trail.sum() * dt)
    if brake_s < 0.25:
        score = 50.0
    else:
        score = float(np.clip(100.0 * trail_s / brake_s, 0, 100))
    events = []
    if trail_s > 0.15:
        t0 = float(df.loc[trail, "time_s"].iloc[0])
        events.append(
            {
                "t_s": t0,
                "kind": "trail",
                "text": f"Trail-brake örtüşmesi {trail_s:.2f} s (fren varken direksiyon açılıyor, gaz kapalı).",
            }
        )
    elif brake_s > 0.3:
        t0 = float(df.loc[brake, "time_s"].iloc[0])
        events.append(
            {
                "t_s": t0,
                "kind": "trail",
                "text": "Fren var, fakat direksiyon çözülürken fren örtüşmesi kısa. Düzlük freninden önce viraj içinde bırak.",
            }
        )
    return {"score": score, "trail_s": trail_s, "brake_s": brake_s, "events": events}


def _pilot_lines(pedal, trail, coast, steer_score, analysis) -> list[str]:
    lines = [
        f"Pedal yumuşaklığı {pedal['score']:.0f}/100 (gaz değişiminin %90'ı {pedal['rms_pct_s']:.0f} %/s altında). Direksiyon yumuşaklığı {steer_score:.0f}/100.",
        f"Boşta geçen süre {coast['coast_s']:.2f} s, hareketin %{coast['fraction']*100:.0f}'i. Puan {coast['score']:.0f}/100.",
        f"Trail-brake puanı {trail['score']:.0f}/100 ({trail['trail_s']:.2f} s örtüşme / {trail['brake_s']:.2f} s fren).",
    ]
    if pedal["events"]:
        lines.append(f"Viraj çıkışı gaz/kayma olayı: {len(pedal['events'])} adet. En uzunu {pedal['events'][0]['text']}")
    else:
        lines.append("Viraj içinde yüksek gaz + iç arka patinaj eşiği aşılmadı.")
    slip = analysis["slip"]["wheels"]
    worst = max(CORNERS, key=lambda c: slip[c]["peak_slip"])
    lines.append(
        f"En yüksek kayma {worst.upper()} tekerde {slip[worst]['peak_slip']:.2f} "
        f"({slip[worst]['spin_s']:.2f} s patinaj, {slip[worst]['lock_s']:.2f} s kilitlenme yönü)."
    )
    z = analysis["session"]["zero_to_100_s"]
    if z is not None:
        lines.append(f"0–100 km/h: {z:.2f} s.")
    else:
        lines.append("0–100 km/h bu kayıtta tamamlanmadı.")
    return lines


def _vcu_lines(analysis) -> list[str]:
    tv = analysis["tv"]
    hv = analysis["hv"]
    slip = analysis["slip"]
    lines = []
    if tv["samples"] == 0:
        lines.append("Viraj örneği az. TV/denge hükmü için |direksiyon| ve hız eşiği birlikte gelmedi.")
    else:
        if tv["tv_inactive"]:
            lines.append(
                f"Virajda ortalama |sağ−sol| tork {tv['mean_abs_tq_delta_nm']:.2f} Nm. "
                "TV dağılımı fiilen yok; control_hooks.c içindeki tv_yaw_moment nötr ise bu beklenen sonuç."
            )
        else:
            lines.append(
                f"Virajda ortalama |sağ−sol| tork {tv['mean_abs_tq_delta_nm']:.2f} Nm, "
                f"tahmini yaw momenti {tv['mean_abs_yaw_moment_nm']:.1f} Nm."
            )
        if tv["lag_ms"] is not None:
            lines.append(
                f"Tork dağılımı direksiyona göre {tv['lag_ms']:.0f} ms gecikmeli (korelasyon {tv['lag_corr']:.2f}). "
                + ("100 ms üstü; TV döngüsü geç kalıyor." if tv["lag_warn"] else "Gecikme 100 ms altında.")
            )
        bal = tv["balance_median"]
        if bal is not None:
            lines.append(
                f"Yaw / kinematik yaw ortancası {bal:.2f}. "
                f"Understeer payı %{tv['understeer_fraction']*100:.0f}, oversteer payı %{tv['oversteer_fraction']*100:.0f}. "
                f"Hesap volan/yol oranı {tv['steer_ratio_assumed']:.1f} varsayar."
            )
    if slip["tc_miss_corners"]:
        names = ", ".join(c.upper() for c in slip["tc_miss_corners"])
        lines.append(
            f"{names} kayarken tork komutu komşu tekerlere yakın kaldı. tc_wheel_scale bu tekerde kısmamış görünüyor."
        )
    else:
        lines.append("Kayma eşiğini aşan tekerde tork komutu belirgin kesilmemiş bir TC kaçağı yok.")
    if hv["exceed_s"] > 0.02:
        lines.append(
            f"80 kW üstü {hv['exceed_s']*1000:.0f} ms ({len(hv['exceed_events'])} pencere). Limitör geç kesiyor."
        )
    if hv["early_derate_s"] > 0.05:
        lines.append(
            f"Güç {analysis['session']['power_limit_w']/1000:.0f} kW'ın belirgin altındayken pwr_scale 0.90'ın altına indi: "
            f"{hv['early_derate_s']*1000:.0f} ms erken kısma."
        )
    if hv["hold_fraction"] is not None:
        lines.append(
            f"Limitör aktifken örneklerin %{hv['hold_fraction']*100:.0f}'i 78.0–79.9 kW bandında. Hedef bu bandı dalgalanmadan tutmak."
        )
    if hv["logged_power_drops_regen"]:
        lines.append("pack_power_w sütunu regen'de 0. Signed güç pack_v × pack_i ile kuruldu; firmware işareti atıyor.")
    return lines


def _alert_lines(analysis) -> list[str]:
    hv = analysis["hv"]
    lv = analysis["lv"]
    thermal = analysis["thermal"]
    lines = [
        f"Paket {hv['vmin_v']:.1f}–{hv['vmax_v']:.1f} V (dolu bant {hv['full_band'][0]:.0f}–{hv['full_band'][1]:.0f} V). "
        f"Pik güç {hv['peak_w']/1000:.2f} kW, pik deşarj sınırı 80 kW."
    ]
    if hv["rint_ohm"] is not None:
        lines.append(
            f"Deşarj doğrusu: Voc {hv['voc_v']:.1f} V, R_internal {hv['rint_ohm']*1000:.1f} mΩ "
            f"({hv['rint_samples']} örnek, I > eşik)."
        )
    else:
        lines.append("İç direnç için yeterli yüksek akım örneği yok.")
    if hv["cell_vmin_v"] is None:
        lines.append("Hücre minimum voltajı log çerçevesinde yok (BMS_VOLT1/2/3 çözülmüyor).")
    else:
        lines.append(f"Hücre Vmin {hv['cell_vmin_v']:.3f} V.")
    if thermal["dc_voltage_empty"]:
        lines.append("İnverter DC gerilimi kayıtta ~0. inverter_amk.c ACT1/ACT2 içinde DC bara alanı doldurmuyor; HV hüküm paketten.")
    elif thermal["dc_voltage_saturated"]:
        lines.append("dc_v_10mV int16 tavanına (327.67 V) dayanmış. 600 V bara bu alanda taşar.")
    if not thermal["dc_current_logged"]:
        lines.append("İnverter DC akımları ~0. Köşe yükü mekanik tork×hız ile izlendi.")
    if lv["available"]:
        lines.append(
            f"LV {lv['min_v']:.2f}–{lv['max_v']:.2f} V, ortalama {lv['mean_v']:.2f} V. "
            f"21.5 V altı {lv['brownout_s']*1000:.0f} ms."
        )
        if lv["brownout_s"] > 0.02:
            lines.append("LV brownout riski var. 24 V hatta pompa/fan devreye girişi ve VCU beslemesi kontrol edilmeli.")
    else:
        lines.append("LV 24 V bu dosyada yok. CAN 0x131 ve ADC1 INP2 loga yazılmıyor; brownout hükmü verilemez.")
    hot = [
        f"{name.upper()} IGBT {ch['igbt_max_c']:.0f} °C"
        for name, ch in thermal["channels"].items()
        if ch["igbt_max_c"] >= 85
    ]
    if hot:
        lines.append("Sıcak kanal: " + ", ".join(hot) + ".")
    lines.append(
        f"Inverter_1 (FL+FR) IGBT max {thermal['inv1_igbt_max_c']:.1f} °C, "
        f"Inverter_2 (RL+RR) {thermal['inv2_igbt_max_c']:.1f} °C, "
        f"fark tepe {thermal['imbalance_max_c']:.1f} °C."
    )
    if thermal["imbalance_warn"]:
        lines.append("İki inverter arası IGBT farkı 8 °C üstü. Arka/ön soğutma veya tork payı dengesiz.")
    regen = analysis["session"]
    lines.append(
        f"Enerji: deşarj {regen['energy_discharge_wh']:.1f} Wh, geri kazanım {regen['energy_regen_wh']:.1f} Wh "
        f"(oran %{regen['regen_ratio']*100:.1f}). Yazılım regen tavanı 20 kW."
    )
    return lines


def _true_runs(time_s: np.ndarray, mask: np.ndarray):
    runs = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((float(time_s[start]), float(time_s[i - 1])))
            start = None
    if start is not None:
        runs.append((float(time_s[start]), float(time_s[-1])))
    return runs
