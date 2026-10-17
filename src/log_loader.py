"""VCU log dosyalarini muhendislik birimine cevirir ve 100 Hz izgara kurar."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd

from schema import LOGS_DIR, load_schema, signals

FRAME_STRUCT = struct.Struct("<I4B2hHhIH32h8BH")
FRAME_BYTES = FRAME_STRUCT.size
RAW_CORNER_FIELDS = (
    "tq_cmd_cNm",
    "tq_fb_cNm",
    "rpm",
    "slip_permille",
    "dc_v_10mV",
    "dc_i_10mA",
    "igbt_cdeg",
    "motor_cdeg",
)
CORNER_SUFFIX = ("fl", "fr", "rl", "rr")
LOG_EXTENSIONS = {".csv", ".parquet", ".log", ".bin", ".txt"}


def assert_frame_size() -> None:
    if FRAME_BYTES != 96:
        raise RuntimeError(f"log_frame 96 bayt olmali, hesap {FRAME_BYTES}")


def list_logs(folder: Path | None = None) -> list[Path]:
    root = folder or LOGS_DIR
    root.mkdir(parents=True, exist_ok=True)
    files = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in LOG_EXTENSIONS and not path.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def load_log(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Ham logu olcekler, 100 Hz'e hizalar. (df, meta) doner."""
    assert_frame_size()
    path = Path(path)
    raw, origin = _read_any(path)
    scaled = apply_scales(raw)
    meta = _rate_meta(scaled, origin, path)
    aligned = resample_to_hz(scaled, meta["target_hz"])
    meta["rows_aligned"] = int(len(aligned))
    meta["missing_logged"] = _missing_logged(raw)
    meta["extensions_present"] = _extensions_present(raw)
    return aligned, meta


def apply_scales(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    present = set(raw.columns)
    for spec in signals():
        csv_name = spec["csv"]
        eng = spec.get("eng", csv_name)
        if csv_name in present:
            series = pd.to_numeric(raw[csv_name], errors="coerce")
            scale = float(spec.get("scale", 1))
            offset = float(spec.get("offset", 0))
            out[eng] = series * scale + offset
        elif eng in present:
            out[eng] = pd.to_numeric(raw[eng], errors="coerce")
    if "time_ms" not in out.columns:
        raise ValueError("Logda time_ms yok. VCU CSV basligi bekleniyor.")
    out["time_s"] = out["time_ms"] / 1000.0
    return out


def resample_to_hz(df: pd.DataFrame, hz: float) -> pd.DataFrame:
    if len(df) < 3 or hz <= 0:
        out = df.copy()
        if "time_s" not in out.columns and "time_ms" in out.columns:
            out["time_s"] = out["time_ms"] / 1000.0
        return out.reset_index(drop=True)

    work = df.sort_values("time_ms").drop_duplicates("time_ms", keep="last")
    t0 = float(work["time_ms"].iloc[0])
    index = pd.to_timedelta(work["time_ms"] - t0, unit="ms")
    work = work.drop(columns=["time_s"], errors="ignore").set_index(index)
    period_ms = max(1, int(round(1000.0 / hz)))
    grid = pd.timedelta_range(work.index.min(), work.index.max(), freq=f"{period_ms}ms")
    ffill_cols = [
        spec.get("eng", spec["csv"])
        for spec in signals()
        if spec.get("resample") == "ffill" and spec.get("eng", spec["csv"]) in work.columns
    ]
    num_cols = [c for c in work.columns if c not in ffill_cols]
    union = work.index.union(grid)
    numeric = work[num_cols].reindex(union).interpolate(method="time", limit=5).reindex(grid)
    flags = work[ffill_cols].reindex(union).ffill().bfill().reindex(grid)
    merged = pd.concat([numeric, flags], axis=1)
    merged["time_ms"] = t0 + merged.index.total_seconds() * 1000.0
    merged["time_s"] = merged["time_ms"] / 1000.0
    return merged.reset_index(drop=True)


def _rate_meta(df: pd.DataFrame, origin: str, path: Path) -> dict:
    schema = load_schema()
    target = float(schema.get("meta", {}).get("log_rate_hz", 100))
    dt_ms = np.diff(df["time_ms"].to_numpy(dtype=float))
    dt_ms = dt_ms[np.isfinite(dt_ms) & (dt_ms > 0)]
    median = float(np.median(dt_ms)) if len(dt_ms) else 10.0
    return {
        "path": str(path),
        "name": path.name,
        "origin": origin,
        "rows_raw": int(len(df)),
        "median_dt_ms": median,
        "native_hz": (1000.0 / median) if median else target,
        "target_hz": target,
        "duration_s": float(df["time_ms"].iloc[-1] - df["time_ms"].iloc[0]) / 1000.0 if len(df) else 0.0,
    }


def _missing_logged(raw: pd.DataFrame) -> list[str]:
    return [spec["csv"] for spec in signals() if spec.get("logged") and spec["csv"] not in raw.columns]


def _extensions_present(raw: pd.DataFrame) -> list[str]:
    return [spec["csv"] for spec in signals() if not spec.get("logged") and spec["csv"] in raw.columns]


def _read_any(path: Path) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path), "parquet"
    if suffix == ".bin":
        return _read_binary(path), "binary"
    if suffix in {".csv", ".txt", ".log"}:
        text = _read_text_or_binary(path)
        if isinstance(text, pd.DataFrame):
            return text, "binary"
        return _read_text_table(path), "text"
    try:
        return _read_text_table(path), "text"
    except Exception:
        return _read_binary(path), "binary"


def _read_text_or_binary(path: Path) -> pd.DataFrame | str:
    blob = path.read_bytes()
    if not blob:
        raise ValueError(f"Bos dosya: {path.name}")
    sample = blob[:200].lstrip()
    if sample.startswith(b"time_ms") or b"," in sample[:80] or b";" in sample[:80]:
        return "text"
    if len(blob) % FRAME_BYTES == 0 and blob[0:1].isdigit() is False:
        return _read_binary(path)
    return "text"


def _read_text_table(path: Path) -> pd.DataFrame:
    head = path.read_text(encoding="utf-8-sig", errors="replace")[:400]
    sep = ";" if head.count(";") >= head.count(",") else ","
    frame = pd.read_csv(path, sep=sep, encoding="utf-8-sig", engine="python")
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.empty:
        raise ValueError(f"Veri satiri yok: {path.name}")
    if sep == ";" and frame.shape[1] == 1 and "," in str(frame.columns[0]):
        frame = pd.read_csv(path, sep=",", encoding="utf-8-sig")
        frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _read_binary(path: Path) -> pd.DataFrame:
    blob = path.read_bytes()
    if len(blob) < FRAME_BYTES or len(blob) % FRAME_BYTES != 0:
        raise ValueError(f"{path.name} 96 baytlik log cercevesine bolunmuyor ({len(blob)} bayt)")
    rows = []
    for offset in range(0, len(blob), FRAME_BYTES):
        unpacked = FRAME_STRUCT.unpack_from(blob, offset)
        row = {
            "time_ms": unpacked[0],
            "state": unpacked[1],
            "drive_mode": unpacked[2],
            "apps_status": unpacked[3],
            "apps_pct": unpacked[4],
            "steer_cdeg": unpacked[5],
            "yaw_cdps": unpacked[6],
            "pack_v_10mV": unpacked[7],
            "pack_i_10mA": unpacked[8],
            "pack_power_w": unpacked[9],
            "pwr_scale_permille": unpacked[10],
        }
        cursor = 11
        for field in RAW_CORNER_FIELDS:
            for suffix in CORNER_SUFFIX:
                row[f"{field}_{suffix}"] = unpacked[cursor]
                cursor += 1
        for suffix in CORNER_SUFFIX:
            row[f"inv_state_{suffix}"] = unpacked[cursor]
            cursor += 1
        row["brake_f_bar"] = unpacked[cursor]
        row["brake_r_bar"] = unpacked[cursor + 1]
        row["sdc_flags"] = unpacked[cursor + 2]
        row["fault_reason"] = unpacked[cursor + 3]
        rows.append(row)
    return pd.DataFrame(rows)
