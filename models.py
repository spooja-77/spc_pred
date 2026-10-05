"""One best-fit algorithm per use case. Every forecast is returned on the real plant calendar (date & time)."""
import warnings
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR, HistGradientBoostingClassifier as HGBC, IsolationForest
from sklearn.metrics import mean_absolute_error, roc_auc_score
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from data import USL, LSL, TARGET, GUARD, SHIFT_START_H, future_ts

warnings.filterwarnings("ignore")
B = 50                                   # block = 50 parts (~29 min of production)
FEAT = ["dev", "sd", "slope", "since", "d1", "hour", "dow", "mins", "shift"]


def cpk(x):
    m, s = np.mean(x), np.std(x, ddof=1)
    return min(USL - m, m - LSL) / (3 * s)


def blocks(df):
    b = df.groupby(df.idx // B).agg(level=("od", "mean"), sd=("od", "std"), ng=("ng", "sum"), ts=("ts", "first"),
                                    te=("ts", "last"), shift=("shift_n", "first")).reset_index(drop=True)
    b["seg"] = (b.level.diff() > 0.005).cumsum()                 # offset / insert-change resets
    b["since"] = b.groupby("seg").cumcount() * B                 # parts since last reset
    b["slope"] = b.level.diff(6) / (6 * B) * 1e5                 # µm / 100 parts
    b["d1"] = b.level.diff().fillna(0) * 1000
    b["dev"] = (b.level - TARGET) * 1000
    hh = b.ts.dt.hour + b.ts.dt.minute / 60
    b["hour"], b["dow"] = b.ts.dt.hour, b.ts.dt.dayofweek       # calendar features (thermal / shift effects)
    b["mins"] = (hh - np.where(hh >= SHIFT_START_H[1], SHIFT_START_H[1], SHIFT_START_H[0])) * 60   # minutes since shift start
    return b


def parts_to_guard(b):
    t = pd.Series(np.nan, index=b.index)
    for _, g in b.groupby("seg"):
        hit = g.index[g.level < GUARD]
        if len(hit): t[g.index] = np.maximum(0, (hit[0] - g.index) * B)   # current (censored) segment stays NaN
    return t


def fit_reg(X, y):
    k = int(len(X) * .8); mk = lambda: HGBR(max_iter=300, learning_rate=.05, random_state=0)
    mae = mean_absolute_error(y.iloc[k:], mk().fit(X.iloc[:k], y.iloc[:k]).predict(X.iloc[k:]))
    return mk().fit(X, y), mae


def fit_q(X, y, q):
    return HGBR(loss="quantile", quantile=q, max_iter=200, learning_rate=.05, random_state=0).fit(X, y)


def fit_clf(X, y):
    k = int(len(X) * .8); mk = lambda: HGBC(max_iter=200, learning_rate=.05, random_state=0); auc = None
    if y.iloc[:k].nunique() > 1 and y.iloc[k:].nunique() > 1:
        auc = roc_auc_score(y.iloc[k:], mk().fit(X.iloc[:k], y.iloc[:k]).predict_proba(X.iloc[k:])[:, 1])
    return (mk().fit(X, y) if y.nunique() > 1 else None), auc


def cusum(x, k=.5, h=5):
    sg = np.mean(np.abs(np.diff(x))) / 1.128
    z = (x - TARGET) / sg; cp, cn, al = np.zeros(len(x)), np.zeros(len(x)), []; a = c = 0.0
    for i, v in enumerate(z):
        a = max(0, a + v - k); c = min(0, c + v + k); cp[i], cn[i] = a, c
        if a > h or c < -h: al.append((i, 1 if a > h else -1)); a = c = 0.0     # alarm -> restart
    return cp, cn, al


def run_all(df):
    R, b = {}, blocks(df); R["b"] = b; X = b[FEAT]; asof = df.ts.iloc[-1]; R["asof"] = asof
    tm = lambda n: future_ts(asof, int(n))[-1] if n >= 1 else asof          # parts ahead -> real timestamp
    # 1 Tool wear – Huber (outlier-robust) slope on last 500 parts
    h = HuberRegressor().fit(np.arange(500).reshape(-1, 1), df.od.values[-500:])
    R["wear"], R["level"] = h.coef_[0] * 1e5, h.predict([[499]])[0]
    st = b[b.seg == b.seg.iloc[-1]].level.iloc[:3].mean()
    R["band"] = float(np.clip((st - R["level"]) / (st - LSL) * 100, 0, 100))
    n_lsl = (R["level"] - LSL) / (-R["wear"] / 1e5) if R["wear"] < -1e-6 else None
    R["lsl_ts"] = tm(n_lsl) if n_lsl else None                               # LSL breach if nobody corrects
    # 2 Remaining useful life – gradient boosting + P10/P90 quantile models -> due time window
    t = parts_to_guard(b); ok = t.notna(); Xo, yo = X[ok].reset_index(drop=True), t[ok].reset_index(drop=True)
    m, R["rul_mae"] = fit_reg(Xo, yo); xl = X.iloc[[-1]]
    R["rul"] = max(0., float(m.predict(xl)[0]))
    lo, hi = sorted(max(0., float(fit_q(Xo, yo, q).predict(xl)[0])) for q in (.1, .9))
    R["rul_lo"], R["rul_hi"] = min(lo, R["rul"]), max(hi, R["rul"])
    R["due"], R["due_lo"], R["due_hi"] = tm(R["rul"]), tm(R["rul_lo"]), tm(R["rul_hi"])
    # 3 Offset adjustment – gradient boosting classifier (breach within 300 parts ≈ 3 h)
    m, R["off_auc"] = fit_clf(Xo, (yo <= 300).astype(int))
    R["off_p"] = float(m.predict_proba(xl)[0, 1]) if m else 0.
    R["off_corr"] = (TARGET - R["level"]) * 1000
    # 4 OD drift – Kalman local-linear-trend on block means, 24 blocks ahead with 80% band
    fc = UnobservedComponents(b.level.tail(400).values, "local linear trend").fit(disp=False).get_forecast(24)
    ci = fc.conf_int(alpha=.2); R["drift"], R["drift_lo"], R["drift_hi"] = fc.predicted_mean, ci[:, 0], ci[:, 1]
    R["fts"] = future_ts(asof, 24 * B)[B - 1::B]                             # timestamp at the end of each forecast block
    # 5 Out-of-spec – gradient boosting classifier (any NG in next 100 parts ≈ 1 h)
    y = pd.Series(((b.ng.shift(-1) + b.ng.shift(-2)) > 0).astype(int)[:-2]).reset_index(drop=True)
    m, R["oos_auc"] = fit_clf(X.iloc[:-2].reset_index(drop=True), y)
    if m: R["oos_p"] = float(m.predict_proba(xl)[0, 1])
    else:
        s = df.od.tail(300).std(); mu = R["drift"][1]
        R["oos_p"] = 1 - (1 - (1 - norm.cdf((USL - mu) / s) + norm.cdf((LSL - mu) / s))) ** 100
    # 6 Process shift – CUSUM with restart after each alarm
    R["cp"], R["cn"], R["al"] = cusum(df.od.values)
    li, ld = R["al"][-1] if R["al"] else (None, 0)
    recent = li is not None and len(df) - 1 - li <= 600
    R["shift"] = ("Upward shift" if ld > 0 else "Downward shift") if recent else "In control"
    R["alarm_ts"] = df.ts.iloc[li] if li is not None else None
    # 7 Variation – damped-trend exponential smoothing on log σ (24 blocks ahead)
    ls = np.log(b.sd.tail(300).values * 1000)
    R["sig_fc"] = np.exp(ExponentialSmoothing(ls, trend="add", damped_trend=True).fit().forecast(24))
    R["sig_now"], R["sig_prev"] = df.od.tail(300).std() * 1000, df.od.iloc[-600:-300].std() * 1000
    # 8 Cpk – daily Cpk by calendar date, damped Holt forecast for the next 7 production days
    d = df.groupby(df.ts.dt.normalize()).od.apply(lambda x: cpk(x) if len(x) > 50 else np.nan).dropna(); R["cpk_d"] = d
    R["cpk_fc"] = ExponentialSmoothing(d.values, trend="add", damped_trend=True).fit().forecast(7) if len(d) > 10 else np.repeat(d.iloc[-1], 7)
    fut = future_ts(asof, 8 * 1200).normalize().unique(); R["cpk_dates"] = fut[fut > asof.normalize()][:7]
    # 9 Anomaly – Isolation Forest on EWMA residual / step / local spread
    res = df.od - df.od.ewm(alpha=.2).mean().shift(1).bfill()
    F = pd.DataFrame({"res": res, "d1": df.od.diff().fillna(0), "sd20": df.od.rolling(20, min_periods=5).std().bfill()})
    R["anom"] = IsolationForest(n_estimators=150, contamination=.003, random_state=0).fit_predict(F) == -1
    return R
