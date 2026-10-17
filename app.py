"""IST27 VCU log ve performans ekrani."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from driver_coaching import coach
from log_loader import list_logs, load_log
from ml_engine import run_ml
from schema import LOGS_DIR, load_schema
from vehicle_analytics import analyze
import plots

st.set_page_config(page_title="IST27 VCU Analyzer", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: #151A22;
        border: 1px solid #243041;
        padding: 10px 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def evaluate(path: str):
    frame, meta = load_log(path)
    result = analyze(frame)
    report = coach(result["df"], result)
    model = run_ml(result["df"], result)
    return meta, result, report, model


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    st.title("IST27 VCU Log & Performance Analyzer")
    st.caption("Kaynak çerçeve 100 Hz · CM7 `log_frame_t` 96 bayt · CSV ayırıcı `;` · HV 600 V · LV 24 V · 4 AMK / 2 inverter grubu")

    files = list_logs()
    with st.sidebar:
        st.header("Loglar")
        st.caption(str(LOGS_DIR))
        if not files:
            st.warning("`logs_input` boş. `run_analyzer.bat` örnek CSV üretir.")
            return
        labels = [path.name for path in files]
        picked = st.multiselect("Karşılaştır", labels, default=labels[:1], max_selections=2)
        if not picked:
            st.info("En az bir log seç.")
            return
        schema = load_schema()
        with st.expander("Firmware log notu"):
            st.write(f"Kayıt hızı: **{schema['meta']['log_rate_hz']} Hz**")
            for gap in schema.get("firmware_gaps", []):
                st.write("• " + gap)

    chosen = [next(path for path in files if path.name == name) for name in picked]
    packs = []
    for path in chosen:
        try:
            packs.append((path, *evaluate(str(path))))
        except Exception as exc:
            st.error(f"{path.name} okunamadı: {exc}")
            return

    tabs = st.tabs(
        [
            "Genel Bakış & Koçluk",
            "4 Motor & 2 İnverter",
            "600 V HV & 24 V LV",
            "Pilotaj & G-G",
            "Arıza / Shutdown",
        ]
    )
    with tabs[0]:
        overview(packs)
    with tabs[1]:
        drivetrain(packs[0])
    with tabs[2]:
        power(packs[0])
    with tabs[3]:
        driving(packs[0])
    with tabs[4]:
        blackbox(packs[0])


def overview(packs):
    if len(packs) == 2:
        left, right = st.columns(2)
        with left:
            session_card(packs[0])
        with right:
            session_card(packs[1])
    else:
        session_card(packs[0])

    _, _, result, report, model = packs[0]
    st.subheader("Pilot İçin Bulgular")
    for line in report["pilot"]:
        st.write("• " + line)
    if report["events"]:
        st.dataframe(pd.DataFrame(report["events"])[["t_s", "kind", "text"]], width="stretch", hide_index=True)

    st.subheader("VCU / Kontrol Algoritması Tavsiyeleri")
    for line in report["vcu"]:
        st.write("• " + line)
    st.write("**Darboğaz sırası**")
    st.dataframe(pd.DataFrame(model["bottlenecks"]), width="stretch", hide_index=True)

    st.subheader("600V HV / 24V LV / Termal Uyarılar")
    for line in report["alerts"]:
        st.write("• " + line)

    if model["anomalies"].get("available"):
        st.subheader("IsolationForest")
        anom = model["anomalies"]
        st.write(
            f"{anom['count']} örnek işaretlendi (%{anom['fraction']*100:.1f}). "
            "Sürücüler, standart sapma cinsinden anomali kümesinin uzaklığı."
        )
        if anom["drivers"]:
            st.dataframe(pd.DataFrame(anom["drivers"]), width="stretch", hide_index=True)
        if anom["events"]:
            st.dataframe(pd.DataFrame(anom["events"]), width="stretch", hide_index=True)
    else:
        st.info(model["anomalies"].get("reason", "Anomali modeli çalışmadı."))

    if len(packs) == 2:
        st.subheader("Karşılaştırma")
        st.plotly_chart(plots.time_series(
            packs[0][2]["df"].assign(power_a=packs[0][2]["df"]["power_w"] / 1000.0),
            [("power_a", packs[0][0].name, "#FF4B4B")],
            "Güç karşılaştırması için birinci log",
        ), width="stretch")
        compare_overlay(packs[0], packs[1])


def session_card(pack):
    path, meta, result, report, _model = pack
    session = result["session"]
    scores = report["scores"]
    st.markdown(f"**{path.name}**")
    st.caption(
        f"Ham {meta['rows_raw']} satır · yerel {meta['native_hz']:.1f} Hz · "
        f"ızgara {meta['target_hz']:.0f} Hz · {meta['duration_s']:.2f} s · {meta['origin']}"
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mesafe", f"{session['distance_m']:.0f} m")
    c2.metric("Vmax", f"{session['vmax_kmh']:.1f} km/h")
    c3.metric("Pmax", f"{session['pmax_w']/1000:.1f} kW")
    z = session["zero_to_100_s"]
    c4.metric("0–100", f"{z:.2f} s" if z is not None else "—")
    c5.metric("Koçluk", f"{scores['overall']:.0f}/100")
    if meta["missing_logged"]:
        st.warning("Eksik firmware sütunları: " + ", ".join(meta["missing_logged"]))
    if meta["extensions_present"]:
        st.caption("Ek sütunlar (firmware çerçevesinde yok): " + ", ".join(meta["extensions_present"]))


def compare_overlay(a, b):
    da = a[2]["df"]
    db = b[2]["df"]
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=da["time_s"], y=da["power_w"] / 1000.0, name=a[0].name, line=dict(color="#FF4B4B")))
    fig.add_trace(go.Scatter(x=db["time_s"], y=db["power_w"] / 1000.0, name=b[0].name, line=dict(color="#4CC9F0")))
    fig.add_hline(y=80, line_dash="dash", line_color="#FFD166")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0B0E14", plot_bgcolor="#0B0E14", title="Güç kıyası", yaxis_title="kW", xaxis_title="s")
    st.plotly_chart(fig, width="stretch")


def drivetrain(pack):
    df = pack[2]["df"]
    tv = pack[2]["tv"]
    slip = pack[2]["slip"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("|Tsağ−Tsol|", f"{tv['mean_abs_tq_delta_nm']:.2f} Nm")
    c2.metric("TV gecikme", "—" if tv["lag_ms"] is None else f"{tv['lag_ms']:.0f} ms")
    c3.metric("Understeer", f"%{tv['understeer_fraction']*100:.0f}")
    c4.metric("Oversteer", f"%{tv['oversteer_fraction']*100:.0f}")
    st.plotly_chart(plots.torque_rpm(df), width="stretch")
    st.plotly_chart(plots.slip_fig(df), width="stretch")
    st.plotly_chart(plots.inverter_fig(df), width="stretch")
    rows = []
    for corner, info in slip["wheels"].items():
        rows.append({"teker": corner.upper(), **info})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def power(pack):
    df = pack[2]["df"]
    hv = pack[2]["hv"]
    lv = pack[2]["lv"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("80 kW üstü", f"{hv['exceed_s']*1000:.0f} ms")
    c2.metric("Erken kısma", f"{hv['early_derate_s']*1000:.0f} ms")
    hold = "—" if hv["hold_fraction"] is None else f"%{hv['hold_fraction']*100:.0f}"
    c3.metric("78–79.9 bandı", hold)
    rint = "—" if hv["rint_ohm"] is None else f"{hv['rint_ohm']*1000:.1f} mΩ"
    c4.metric("Rinternal", rint)
    st.plotly_chart(plots.power_fig(df), width="stretch")
    fig = plots.lv_fig(df)
    if fig is None:
        st.info("Bu logda `lv_v_10mV` yok. 24 V brownout ancak sütun eklenince çizilir. Gerçek VCU çerçevesi LV'yi yazmıyor.")
    else:
        st.plotly_chart(fig, width="stretch")
        if lv["available"] and lv["events"]:
            st.dataframe(pd.DataFrame(lv["events"]), width="stretch", hide_index=True)


def driving(pack):
    df = pack[2]["df"]
    st.plotly_chart(plots.driver_fig(df), width="stretch")
    st.plotly_chart(plots.gg_fig(df), width="stretch")
    fft = plots.piezo_fft(df)
    if fft is None:
        st.info("`piezo1` / `piezo2` log çerçevesinde yok. FFT için bu sütunlar gerekir.")
    else:
        st.plotly_chart(fft, width="stretch")


def blackbox(pack):
    faults = pack[2]["faults"]
    if not faults:
        st.success("Bu kayıtta SDC, AMS, IMD, inverter ERROR veya FAULT kenarı yok.")
        return
    labels = [f"{item['t_s']:.3f} s · {item['kind']} · {item['cause']}" for item in faults]
    index = st.selectbox("Olay", range(len(labels)), format_func=lambda i: labels[i])
    event = faults[index]
    st.write(f"**Tetikleyici:** {event['cause']}")
    st.write(f"Durum `{event['state']}`, fault `{event['fault_reason']}`, SDC: {', '.join(event['sdc']) or '—'}")
    for line in event["details"]:
        st.write("• " + line)
    if event["also"]:
        st.caption("Aynı pencerede: " + ", ".join(event["also"]))
    st.plotly_chart(plots.blackbox_fig(event["window"], event["t_s"]), width="stretch")
    show = [
        "time_s", "state", "fault_reason", "sdc_flags", "apps_pct", "apps_status",
        "pack_v", "pack_i", "power_w", "lv_v", "slip_rr", "inv_state_rr",
    ]
    cols = [c for c in show if c in event["window"].columns]
    st.dataframe(event["window"][cols], width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
