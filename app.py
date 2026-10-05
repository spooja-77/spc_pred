import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from data import *
from models import run_all, cpk, B

st.set_page_config("SPC Predictive Monitor", "📈", layout="wide")
BLUE, GREEN, AMBER, RED, VIO = "#2f6fdc", "#2fb461", "#f59e0b", "#ef4444", "#8b5cf6"
st.markdown("""<style>
header[data-testid=stHeader]{display:none}.block-container{padding:0 1.5rem 3rem;max-width:none}
.top{background:#00008b;color:#fff;margin:0 -1.5rem 8px;padding:14px 24px;display:flex;justify-content:space-between;align-items:center}
.top h1{margin:0;font-size:26px;color:#fff;padding:0}.top small{opacity:.8}
.sec{display:flex;align-items:center;gap:12px;font-size:17px;font-weight:600;margin:26px 0 12px}
.sec:before{content:"";width:3px;height:19px;background:#2f6fdc}.sec:after{content:"";flex:1;height:1px;background:#e6eaf1}
.kp{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));border:1px solid #e6eaf1;border-radius:16px;overflow:hidden;background:#fff}
.kr{display:grid;grid-template-columns:40px 1fr auto 40px;gap:12px;align-items:center;padding:14px 18px;border:1px solid #e6eaf1;margin:0 -1px -1px 0}
.ic{width:34px;height:34px;border-radius:9px;display:grid;place-items:center;font-weight:700;font-size:13px}
.kr em{font-style:normal;color:#6b7488;font-weight:600;font-size:13px}.kr b{font-size:22px}.kr u{text-decoration:none;font:11px monospace;color:#6b7488}
.uc{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.u{background:#fff;border:1px solid #e6eaf1;border-top:3px solid var(--c);border-radius:16px;padding:14px 16px}
.u i{font-style:normal;font-size:11px;color:#6b7488}.u h3{margin:2px 0 6px;font-size:14px;padding:0}.u .v{font-size:21px;font-weight:700;color:var(--c)}
.u p{margin:3px 0 8px;color:#6b7488;font-size:12.5px}.bar{height:6px;border-radius:4px;background:#e6eaf1}.bar div{height:100%;border-radius:4px;background:var(--c)}
</style>""", unsafe_allow_html=True)

up = st.sidebar.file_uploader("Load SPC file (csv / tsv / xlsx)", type=["csv", "tsv", "txt", "xlsx"])

@st.cache_data(show_spinner="Loading data & training models…")
def pipeline(raw, name):
    df = load(raw, name); return df, run_all(df)

df, R = pipeline(up.getvalue() if up else None, up.name if up else None)
b, asof = R["b"], R["asof"]
dmin, dmax = df.ts.min().date(), df.ts.max().date()
rg = st.sidebar.date_input("History range (charts & KPIs)", (dmin, dmax), min_value=dmin, max_value=dmax)
lo, hi = (rg[0], rg[-1]) if isinstance(rg, (list, tuple)) and len(rg) else (dmin, dmax)
dff = df[(df.ts.dt.date >= lo) & (df.ts.dt.date <= hi)]
if dff.empty: dff = df
N = len(dff); m, s = dff.od.mean(), dff.od.std()
cp_, ck, ng = (USL - LSL) / (6 * s), cpk(dff.od), dff.ng.mean() * 100
cpf = float(np.mean(R["cpk_fc"]))
fm = lambda t: pd.Timestamp(t).strftime("%a %d %b, %H:%M")
st.sidebar.markdown("**Model performance (time-split 80/20)**")
st.sidebar.write({"RUL MAE (parts)": round(R["rul_mae"]), "Offset ROC-AUC": R["off_auc"], "Out-of-spec ROC-AUC": R["oos_auc"]})
st.markdown(f'<div class="top"><h1>SPC Predictive Monitor</h1><small>CNC Bearing Bore · Ø{LSL}–{USL} mm · data as of <b>{fm(asof)}</b> · {len(df):,} parts</small></div>', unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)
def show(f, key, rb="hour"):
    f.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=330, plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12), legend=dict(orientation="h", y=-.2), title_font_size=15)
    f.update_xaxes(gridcolor="#e6eaf1"); f.update_yaxes(gridcolor="#e6eaf1")
    if rb:   # hide non-production time (nights / Sundays) so the time axis has no empty gaps
        br = [dict(bounds=["sun", "mon"])] + ([dict(pattern="hour", bounds=[PROD_END_H, SHIFT_START_H[0]])] if rb == "hour" else [])
        f.update_xaxes(rangebreaks=br)
    try: st.plotly_chart(f, width="stretch", key=key)
    except Exception: st.plotly_chart(f, use_container_width=True, key=key)
def lines(f):
    for v, c, d, n in [(USL, RED, "dash", "USL"), (LSL, RED, "dash", "LSL"), (TARGET, GREEN, "solid", "Target")]:
        f.add_hline(y=v, line=dict(color=c, dash=d, width=1.2), annotation_text=n, annotation_position="right")
def now(f): f.add_shape(type="line", x0=asof, x1=asof, y0=0, y1=1, yref="paper", line=dict(color="#6b7488", dash="dot"))
def fig(title, x, y, yr=None):
    f = go.Figure(); f.update_layout(title=title, xaxis_title=x, yaxis_title=y)
    if yr: f.update_yaxes(range=yr)
    return f

kp = [("⚙", "Parts Analysed", f"{N:,}", "pcs", BLUE), ("◎", "Mean OD", f"{m:.4f}", "mm", GREEN), ("σ", "Std Deviation", f"{s*1000:.2f}", "µm", VIO),
      ("Cp", "Process Capability", f"{cp_:.2f}", "Cp", BLUE), ("Ck", "Cpk (Selected Range)", f"{ck:.2f}", "Cpk", GREEN),
      ("⚠", "NG Rate", f"{ng:.2f}", "%", RED), ("↗", "Cpk Forecast (7 prod. days)", f"{cpf:.2f}", "Cpk", AMBER)]
sec("Performance summary")
st.markdown('<div class="kp">' + "".join(f'<div class="kr"><span class="ic" style="background:{c}22;color:{c}">{i}</span><em>{l}</em><b>{v}</b><u>{u}</u></div>' for i, l, v, u, c in kp) + "</div>", unsafe_allow_html=True)

def uc(n, t, v, sub, pct, c): return f'<div class="u" style="--c:{c}"><i>USE CASE {n}</i><h3>{t}</h3><div class="v">{v}</div><p>{sub}</p><div class="bar"><div style="width:{max(2,min(100,pct)):.0f}%"></div></div></div>'
lv, d, fts = R["level"], R["drift"], R["fts"]; vr = (R["sig_now"] / R["sig_prev"] - 1) * 100
na = int(R["anom"][-5000:].sum()); tone = lambda x, a, c: RED if x > c else AMBER if x > a else GREEN
pph = 3600 / CYCLE_S; rh = R["rul"] / pph
sec("Predictive use cases")
st.markdown('<div class="uc">' + "".join([
    uc(1, "Tool wear (Huber regression)", f"{R['wear']*pph/100:+.2f} µm / hour", f"{R['wear']:+.2f} µm per 100 pcs · headroom used {R['band']:.0f}%", R["band"], tone(R["band"], 40, 70)),
    uc(2, "Remaining useful life (Gradient Boosting)", f"{rh:.1f} h · {R['rul']:,.0f} pcs", f"Guard line reached {fm(R['due'])}<br>80% window: {fm(R['due_lo'])} – {fm(R['due_hi'])}", rh * 5, RED if rh < 2 else AMBER if rh < 5 else GREEN),
    uc(3, "Offset adjustment (GB classifier)", f"{R['off_p']*100:.0f}% in next 3 h", f"Correct by {R['off_corr']:+.1f} µm around {fm(R['due'])}", R["off_p"] * 100, tone(R["off_p"] * 100, 30, 70)),
    uc(4, "OD drift (Kalman trend)", f"{d[1]:.4f} mm @ {pd.Timestamp(fts[1]):%H:%M}", f"+4 h ({pd.Timestamp(fts[7]):%a %H:%M}): {d[7]:.4f} · now {lv:.4f} mm", R["band"], VIO),
    uc(5, "Out-of-spec (GB classifier)", f"{R['oos_p']*100:.1f}% in next hour", f"P(any NG before {pd.Timestamp(future_ts(asof, 100)[-1]):%a %H:%M})", R["oos_p"] * 100, tone(R["oos_p"] * 100, 5, 20)),
    uc(6, "Process shift (CUSUM)", R["shift"], f"Last alarm: {fm(R['alarm_ts']) if R['alarm_ts'] is not None else 'none'} · C⁻ {R['cn'][-1]:.1f}σ", max(abs(R["cn"][-1]), R["cp"][-1]) * 20, GREEN if R["shift"] == "In control" else AMBER),
    uc(7, "Variation (damped Holt on log σ)", f"{R['sig_now']:.2f} µm → {R['sig_fc'][7]:.2f}", f"{vr:+.0f}% vs previous 300 pcs · forecast for {pd.Timestamp(fts[7]):%a %H:%M}", abs(vr) * 2 + 10, tone(vr, 8, 25)),
    uc(8, "Cpk (damped Holt, daily)", f"{R['cpk_d'].iloc[-1]:.2f} → {cpf:.2f}", f"{R['cpk_d'].index[-1]:%d %b} → avg of {R['cpk_dates'][0]:%d %b}–{R['cpk_dates'][-1]:%d %b}", cpf / 2 * 100, RED if cpf < 1 else AMBER if cpf < 1.33 else GREEN),
    uc(9, "Anomaly (Isolation Forest)", f"{na} flagged", "in the last 5,000 parts (≈ 2 production days)", na / 5000 * 2000, tone(na, 15, 40))]) + "</div>", unsafe_allow_html=True)

sec("Predicted events (plant calendar)")
c1 = next((f"{x:%a %d %b}" for x, v in zip(R["cpk_dates"], R["cpk_fc"]) if v < 1.33), "not within 7 production days")
ev = pd.DataFrame([("Next offset correction needed", fm(R["due"]), f"{fm(R['due_lo'])} – {fm(R['due_hi'])}"),
                   ("LSL breach if nobody corrects", fm(R["lsl_ts"]) if R["lsl_ts"] is not None else "No downward trend", "linear wear trend"),
                   ("Cpk drops below 1.33", c1, "daily Cpk forecast"),
                   ("Last process-shift alarm", fm(R["alarm_ts"]) if R["alarm_ts"] is not None else "none", R["shift"])], columns=["Event", "Predicted time", "Window / basis"])
st.dataframe(ev, hide_index=True, width="stretch")

sec("Process capability & conformance")
n1, hh = 1200, 12; t = df.tail(n1)
f = fig("OD control chart – last 1,200 parts + 80% forecast band (next ~6 h)", "Date & time", "OD (mm)", [LSL - .004, USL + .004])
f.add_scatter(x=t.ts, y=t.od, name="OD", line=dict(color=BLUE, width=1.3))
fx = [asof] + list(fts[:hh]); f.add_scatter(x=fx, y=[t.od.iloc[-1]] + list(R["drift_hi"][:hh]), line=dict(width=0), showlegend=False, hoverinfo="skip")
f.add_scatter(x=fx, y=[t.od.iloc[-1]] + list(R["drift_lo"][:hh]), line=dict(width=0), fill="tonexty", fillcolor="rgba(245,158,11,.2)", name="80% band", hoverinfo="skip")
f.add_scatter(x=fx, y=[t.od.iloc[-1]] + list(d[:hh]), name="Forecast", line=dict(color=AMBER, dash="dash", width=2.5)); lines(f); now(f); show(f, "c1")
c = st.columns(3)
with c[0]:
    f = go.Figure(go.Pie(labels=["In spec", "> USL", "< LSL"], values=[(dff.ng == 0).sum(), (dff.od > USL).sum(), (dff.od < LSL).sum()], hole=.62, marker_colors=[GREEN, RED, AMBER])); f.update_layout(title="Conformance split"); show(f, "c3", None)
with c[1]:
    g = dff.groupby("shift_n").ng.mean() * 100; f = fig("NG rate by shift", "Shift", "NG (%)"); f.add_bar(x=[f"Shift {i}" for i in g.index], y=g.values, marker_color=[BLUE, VIO]); show(f, "c4", None)
with c[2]:
    f = fig("OD distribution", "OD (mm)", "Part count"); f.add_histogram(x=dff.od, nbinsx=70, marker_color=BLUE)
    for v in (LSL, USL): f.add_vline(x=v, line=dict(color=RED, dash="dash"))
    show(f, "c5", None)
cd = R["cpk_d"]; cd = cd[(cd.index >= pd.Timestamp(lo)) & (cd.index <= pd.Timestamp(hi))].tail(45)
f = fig("Daily Cpk by date & 7-day forecast (next production days)", "Date", "Cpk"); f.add_bar(x=cd.index, y=cd.values, name="Daily", marker_color=[RED if v < 1 else AMBER if v < 1.33 else GREEN for v in cd])
f.add_bar(x=R["cpk_dates"], y=R["cpk_fc"], name="Forecast", marker_color="rgba(139,92,246,.55)"); f.add_hline(y=1.33, line=dict(dash="dot", color="#1b2437")); show(f, "c6", "day")

sec("Tool wear, drift & stability")
t = b.tail(120); r = t[t.level.diff() > .005]
f = fig("Tool-wear sawtooth by date & time (50-part block means)", "Date & time", "OD block mean (mm)", [LSL - .003, USL + .003])
f.add_scatter(x=t.ts, y=t.level, line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(47,111,220,.06)", name="Block mean")
f.add_scatter(x=r.ts, y=r.level, mode="markers", marker=dict(size=9, color=GREEN, symbol="triangle-up"), name="Offset / insert change"); lines(f); show(f, "c2")
c = st.columns(3)
with c[0]:
    t = b.tail(120); f = fig("Rolling σ + forecast", "Date & time", "σ (µm)"); f.add_scatter(x=t.ts, y=t.sd * 1000, name="σ", line=dict(color=VIO))
    f.add_scatter(x=fts, y=R["sig_fc"], name="Forecast", line=dict(color=AMBER, dash="dash")); f.add_hline(y=(USL - LSL) / 2 / (3 * 1.33) * 1000, line=dict(color=RED, dash="dash")); now(f); show(f, "c7")
with c[1]:
    t = df.tail(1500); o = len(df) - 1500; f = fig("CUSUM (σ units) with alarms", "Date & time", "CUSUM")
    f.add_scatter(x=t.ts, y=R["cp"][-1500:], name="C⁺", line=dict(color=AMBER)); f.add_scatter(x=t.ts, y=R["cn"][-1500:], name="C⁻", line=dict(color=BLUE))
    al = [(i, dr) for i, dr in R["al"] if i >= o]; f.add_scatter(x=[df.ts.iloc[i] for i, _ in al], y=[5 * dr for _, dr in al], mode="markers", marker=dict(size=9, color=RED), name="Alarm")
    for v in (5, -5): f.add_hline(y=v, line=dict(color=RED, dash="dash"))
    show(f, "c8")
with c[2]:
    t, a = df.tail(800), R["anom"][-800:]; f = fig("Anomaly detection (Isolation Forest)", "Date & time", "OD (mm)", [LSL - .004, USL + .004])
    f.add_scatter(x=t.ts[~a], y=t.od[~a], mode="markers", marker=dict(size=4, color=BLUE), name="Normal"); f.add_scatter(x=t.ts[a], y=t.od[a], mode="markers", marker=dict(size=9, color=RED), name="Anomaly")
    for v in (USL, LSL): f.add_hline(y=v, line=dict(color=RED, dash="dash"))
    show(f, "c9")
hm = dff.assign(dow=dff.ts.dt.dayofweek, hr=dff.ts.dt.hour).pivot_table(index="dow", columns="hr", values="dev", aggfunc="mean")
f = go.Figure(go.Heatmap(z=hm.values, x=[f"{h}:00" for h in hm.columns], y=[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in hm.index], colorscale="RdBu_r", zmid=0, colorbar_title="µm"))
f.update_layout(title="Thermal heat-map – mean OD deviation (µm) by weekday × hour of day"); show(f, "c10", None)
sec("Latest anomalies (date & time)")
ix = np.where(R["anom"])[0][-10:][::-1]
st.dataframe(df.iloc[ix][["ts", "shift", "od", "dev"]].rename(columns={"ts": "Date & time", "shift": "Shift", "od": "OD (mm)", "dev": "Deviation (µm)"}), hide_index=True, width="stretch")
