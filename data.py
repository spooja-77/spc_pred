"""Data loading, synthetic generator and shared constants."""
import io, os
import numpy as np, pandas as pd

USL, LSL = 35.970, 35.945
TARGET = (USL + LSL) / 2
GUARD = LSL + 0.008        # offset-correction guard line (tune to your plant practice)
CYCLE_S = 35
SHIFT_START_H = (6, 14)   # shift start hours
PARTS_PER_SHIFT = 600     # 600 parts x 35 s = 5.8 h per shift
PROD_END_H = 20           # nothing produced between 20:00 and 06:00; Sundays off
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "spc_data.csv")  # drop your file here


def synth(N=100939, seed=7):
    """Demo data: 35 s cycle, 1200 pcs/day, 2 shifts, no Sundays, wear sawtooth + thermal + chatter + spikes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=140)
    dates = dates[dates.dayofweek != 6][: int(np.ceil(N / 1200))]
    k = np.tile(np.arange(1200), len(dates))[:N]
    day = np.repeat(np.arange(len(dates)), 1200)[:N]
    secs = np.where(k >= 600, 14, 6) * 3600 + (k % 600) * CYCLE_S
    ts = dates.values[day] + secs.astype("timedelta64[s]")
    g, u = rng.standard_normal(N), rng.random((N, 2))
    od, L, ch = np.empty(N), 35.9635, 0
    for i in range(N):
        if i and i % 6000 == 0: L = 35.9635            # insert change
        L -= 7.5e-6                                     # tool wear
        if L < 35.9525: L += 0.010                      # operator offset correction
        if u[i, 0] < 4e-5: ch = 40                      # chatter burst
        sd = 0.0017 * (2.2 if ch > 0 else 1); ch -= 1
        x = L + 0.0015 * np.sin(2 * np.pi * k[i] / 1200) + (-0.003 * (1 - k[i] / 80) if k[i] < 80 else 0) + sd * g[i]
        if u[i, 1] < 4e-4: x += 0.007 * (1 if g[i] > 0 else -1)
        od[i] = x
    return pd.DataFrame({"sno": np.arange(1, N + 1), "od": od, "date": pd.to_datetime(ts).date, "ts": ts,
                         "shift": np.where(k >= 600, "Shift 2", "Shift 1")})


def _read_table(src, name=""):
    """Read csv / tsv / semicolon / xlsx (even an xlsx renamed to .csv)."""
    raw = bytes(src) if isinstance(src, (bytes, bytearray)) else open(src, "rb").read()
    if raw[:2] == b"PK" or str(name).lower().endswith(("xlsx", "xls")):
        return pd.read_excel(io.BytesIO(raw))
    lines = [l for l in raw.decode("utf-8-sig", errors="ignore").splitlines() if l.strip()]
    if not lines: raise ValueError("The file is empty")
    sep = max(["\t", ",", ";", "|"], key=lines[0].count)
    return pd.read_csv(io.StringIO("\n".join(lines)), sep=sep if lines[0].count(sep) else r"\t|,|;|\s{2,}", engine="python")


def load(raw=None, name=""):
    """Columns expected: SNo, SPC Reading, Date, Timestamp, Shift (in this order)."""
    if raw is None and os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0: raw, name = DATA_PATH, DATA_PATH
    if raw is None: df = synth()
    else:
        df = _read_table(raw, name)
        if df.shape[1] < 5: raise ValueError(f"Expected 5 columns, found {df.shape[1]}: {list(df.columns)}")
        df = df.iloc[:, :5]; df.columns = ["sno", "od", "date", "ts", "shift"]
        df["od"] = pd.to_numeric(df["od"], errors="coerce"); df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["od", "ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df["shift_n"] = df["shift"].astype(str).str.extract(r"(\d)")[0].astype(float).fillna(1).astype(int)
    df["idx"] = np.arange(len(df))
    df["ng"] = ((df.od > USL) | (df.od < LSL)).astype(int)
    df["dev"] = (df.od - TARGET) * 1000                 # µm from target
    return df


def future_ts(last_ts, n):
    """Real timestamps of the next n parts, following the plant calendar (2 shifts, 35 s cycle, no Sundays)."""
    last, d, out, cnt = pd.Timestamp(last_ts), pd.Timestamp(last_ts).normalize(), [], 0
    while cnt < n:
        if d.dayofweek != 6:
            for h in SHIFT_START_H:
                t = d + pd.Timedelta(hours=h) + pd.to_timedelta(np.arange(PARTS_PER_SHIFT) * CYCLE_S, unit="s")
                t = t[t > last]; out.append(t.values); cnt += len(t)
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(np.concatenate(out)[:n]) if n > 0 else pd.DatetimeIndex([])
