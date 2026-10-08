#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usdjpy_barrier.py : USDJPY 1-ghante ka "triple-barrier" model. SIRF PAPER-TRADING / RESEARCH.

Sawaal jo model sikhta hai:
    "Agle 60 minute me bhaav pehle ooper (+K*sigma) chhuega ya neeche (-K*sigma)?"
Dono barrier barabar doori par hain (sigma = pichhle 120 minute ki volatility, ~10 pip). Isliye
jeet/haar barabar hain aur accuracy 50% se upar hone ka matlab seedha munafa hota hai.
60 minute me koi barrier na tute to 60 minute baad ke bhaav par band maana jata hai.

Chalane ka tareeka (usdjpy_247.py ke saath, usi folder me):
    python usdjpy_247.py import HISTDATA_*.zip     # ek baar: purana 1-minute data (kam se kam ~70,000 bars chahiye)
    python usdjpy_barrier.py backtest              # walk-forward test, natija print + barrier_backtest.json
    python usdjpy_barrier.py predict               # abhi ka signal: print + barrier_latest.json + (NTFY_TOPIC ho to alert)

Settings (environment variable): DATA_DIR, NTFY_TOPIC (bot wale hi), BARRIER_K (default 1.0),
BARRIER_THR (default 0.60; 0.60-0.64 rakhein), BARRIER_COST_PIP (default 0.3), BARRIER_RETRAIN_H (default 6),
BARRIER_STALE_MIN (default 20: itne minute se purane data par naya signal nahi, jaise weekend)

Zaroori sach:
  * Yeh model sirf PRICE ke features use karta hai. News / DXY / US10Y waale features isme nahi hain, kyunki
    unka purana data hamare paas nahi tha, to unka test nahi hua. Bot unka data (news_log.csv, ext.csv) jama karta
    rehta hai; kuch hafte baad wahi backtest unke saath chala kar dekha ja sakta hai ki unse fayda hota hai ya nahi.
  * Model ka 'p' asli sambhavna nahi hai, sirf ranking score hai (overconfident hai). 0.60 ka matlab 60% jeet nahi.
  * Backtest me lagat 0.3 pip maani gayi hai. Asli spread + slippage 1 pip ke aaspaas ho to fayda ~0 ho jata hai.
"""
import os, sys, json, math, time
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as sw
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usdjpy_247 as bot                                   # data, features, alert, folder: sab bot se

E = os.environ.get
H = 60; PIP = 0.01; NONE = 9999
K = float(E("BARRIER_K", "1.0")); THR_B = float(E("BARRIER_THR", "0.60")); COST_PIP = float(E("BARRIER_COST_PIP", "0.3"))
RETRAIN_H = float(E("BARRIER_RETRAIN_H", "6")); MAXTR = 60000; STALE_MIN = float(E("BARRIER_STALE_MIN", "20"))
P = bot.P


def log(m): print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)


def sigma_pip(h):
    """Agle 1 ghante ki andazi chaal (1 sigma), pip me, pichhle 120 minute ki volatility se."""
    return np.log(h.c).diff().rolling(120).std() * np.sqrt(60) * h.c / PIP


def features(h):
    F = bot.feats(h)
    cols = [c for c in bot.cols_of(F) if not c.startswith(bot.XN) and F[c].nunique() > 1]
    return F, cols


def outcomes(h):
    """Har bar ke liye (sirf jinke aage 60 minute ka data hai): label y, LONG/SHORT ka net pip, valid mask."""
    n = len(h) - H - 1; e = h.c.values[:n]; bp = (K * sigma_pip(h)).values[:n]; b = bp * PIP
    Whi = sw(h.h.values[1:], H)[:n]; Wlo = sw(h.l.values[1:], H)[:n]; cl = sw(h.c.values[1:], H)[:n][:, -1]
    hu = Whi >= (e + b)[:, None]; hd = Wlo <= (e - b)[:, None]
    ft = np.where(hu.any(1), hu.argmax(1), NONE); fs = np.where(hd.any(1), hd.argmax(1), NONE)
    up = ft < fs; dn = fs < ft; same = (ft == fs) & (ft < NONE)          # ek hi bar me dono => haar maano (conservative)
    y = np.where(up, 1, np.where(dn, 0, (cl > e).astype(int)))
    pl = np.where(up, bp, np.where(dn | same, -bp, (cl - e) / PIP)) - COST_PIP
    ps = np.where(dn, bp, np.where(up | same, -bp, (e - cl) / PIP)) - COST_PIP
    ix = h.index.values; gap = (ix[H + 1:H + 1 + n] - ix[1:1 + n]).astype("timedelta64[m]").astype(int) <= H + 5
    valid = gap & np.isfinite(bp) & (bp > 0) & (cl != e)
    return dict(n=n, y=y, pl=pl, ps=ps, valid=valid, bp=bp)


class Model:
    """HistGradientBoosting + Logistic Regression ka average."""
    def fit(self, X, y, w):
        self.gb = HGB(max_iter=150, learning_rate=.05, max_depth=4, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=w)
        self.sc = StandardScaler().fit(X); self.lr = LogisticRegression(C=.05, max_iter=300).fit(self._f(X), y, sample_weight=w); return self
    def _f(self, A): return np.clip(np.nan_to_num(self.sc.transform(A)), -6, 6)
    def proba(self, A): return (self.gb.predict_proba(A)[:, 1] + self.lr.predict_proba(self._f(A))[:, 1]) / 2


def train_rows(valid, tsx, t_end):
    """Training me sirf wahi bars jinka 60-minute ka nateeja t_end se pehle pata chal chuka tha (koi leakage nahi)."""
    return np.where(valid & (tsx < t_end - pd.Timedelta(minutes=H + 5)))[0][-MAXTR:]


def stats_line(s, base):
    m = s.pnl.mean(); t = m / (s.pnl.std() / math.sqrt(len(s))) if len(s) > 2 else float("nan")
    return dict(trades=int(len(s)), hit=round(float((s.pnl + COST_PIP > 0).mean() * 100), 1), net_pip=round(float(m), 3), t=round(float(t), 2))


def backtest():
    h = bot.load_hist(); F, cols = features(h); o = outcomes(h); n = o["n"]; tsx = h.index[:n]
    X = F[cols].iloc[:n].values.astype(float)
    START = int(E("BARRIER_START", 60000 if n > 110000 else n // 2)); STEP = 10000; out = []
    log(f"backtest: {len(h)} bars {h.index[0]} -> {h.index[-1]} | features={len(cols)} | K={K} | test {tsx[START]} se")
    for k, s in enumerate(range(START, n, STEP)):
        te = np.arange(s, min(s + STEP, n)); te = te[o["valid"][te]]; tr = train_rows(o["valid"], tsx, tsx[s]); w = np.linspace(.3, 1, len(tr))
        if len(tr) < 5000 or len(te) < 50: continue
        m = Model().fit(X[tr], o["y"][tr], w); out.append(pd.DataFrame(dict(p=m.proba(X[te]), y=o["y"][te], pl=o["pl"][te], ps=o["ps"][te]), index=tsx[te]))
        log(f"  fold {k + 1} {tsx[s]} done")
    d = pd.concat(out); base = float(d.pl.mean()); mid = d.index[len(d) // 2]; res = {}
    print(f"\nTEST: {d.index[0]} -> {d.index[-1]} | {len(d)} bar | hamesha-LONG ka avg: {base:+.3f} pip (sirf trend/drift)")
    print(f"barrier ~{np.nanmedian(o['bp'][o['valid']]):.1f} pip | lagat {COST_PIP} pip | har ghante me ek trade (overlap nahi)\n")
    print(f"{'thr':>5} {'trades':>7} {'hit%':>6} {'net pip':>8} {'t':>5} | {'pehli chhamahi':>14} {'doosri chhamahi':>16}")
    for thr in (0.52, 0.55, 0.58, 0.60):
        side = np.where(d.p >= thr, 1, np.where(d.p <= 1 - thr, -1, 0)); s = d.assign(side=side, pnl=np.where(side > 0, d.pl, d.ps))[side != 0]
        s = s[~s.index.floor("60min").duplicated()]
        if len(s) < 30: continue
        a = stats_line(s, base); h1 = s[s.index < mid].pnl.mean(); h2 = s[s.index >= mid].pnl.mean(); res[str(thr)] = a
        print(f"{thr:>5.2f} {a['trades']:>7} {a['hit']:>6} {a['net_pip']:>+8.3f} {a['t']:>5.1f} | {h1:>+14.3f} {h2:>+16.3f}")
    json.dump(res, open(P("barrier_backtest.json"), "w"), indent=1); print("\nsaved:", P("barrier_backtest.json"))


def resolve(h, preds):
    """Purane (60 minute+ puraane) anumaanon ka asli nateeja bars se nikalna."""
    for q in preds:
        if q.get("res") or q["side"] == 0: continue
        t = pd.Timestamp(q["ts"])
        if h.index[-1] < t + pd.Timedelta(minutes=H): continue
        w = h[(h.index > t) & (h.index <= t + pd.Timedelta(minutes=H))]
        if len(w) < H // 2: q["res"] = "data-gap"; continue
        up = q["side"] > 0; tp = q["tp"]; sl = q["sl"]; res = None
        for hi, lo, c in zip(w.h.values, w.l.values, w.c.values):
            hit_tp = hi >= tp if up else lo <= tp; hit_sl = lo <= sl if up else hi >= sl
            if hit_sl: res = ("SL", -q["bp"]); break              # ek hi bar me dono => SL (conservative)
            if hit_tp: res = ("TP", q["bp"]); break
        if res is None: res = ("timeout", q["side"] * (w.c.values[-1] - q["entry"]) / PIP)
        q["res"] = res[0]; q["net_pip"] = round(float(res[1] - COST_PIP), 2)


def live_stats(preds):
    done = [q for q in preds if q.get("net_pip") is not None]; tot = len(done)
    if not tot: return dict(resolved=0, hit_pct=None, net_pip_total=0, net_pip_avg=None)
    hit = sum(q["res"] == "TP" or (q["res"] == "timeout" and q["net_pip"] + COST_PIP > 0) for q in done)
    return dict(resolved=tot, hit_pct=round(100 * hit / tot, 1), net_pip_total=round(sum(q["net_pip"] for q in done), 1), net_pip_avg=round(sum(q["net_pip"] for q in done) / tot, 2))


def predict():
    h = bot.load_hist()
    try:
        new = bot.fetch("1d" if len(h) > 5000 else "7d")
        if new is not None and len(new): h = bot.update_hist(new)
    except Exception as e: log(f"data fetch error (purane bars se chalta hu): {e!r}")
    if len(h) < 70000: log(f"bars kam hain ({len(h)}); pehle `python usdjpy_247.py import HISTDATA_*.zip` chalayen"); return
    age = (pd.Timestamp.now("UTC").tz_localize(None) - h.index[-1]).total_seconds() / 60
    if age > STALE_MIN and not E("BARRIER_ALLOW_STALE"):                    # weekend / data band: purane bhaav par signal nahi
        preds = bot.jload("barrier_preds.json", []); resolve(h, preds); bot.jsave("barrier_preds.json", preds[-5000:])
        prev = bot.jload("barrier_latest.json", {}); prev.update(time=str(h.index[-1]), signal="बाज़ार बंद / डेटा पुराना", live_stats=live_stats(preds),
                                                             note=f"आख़िरी bar {age:.0f} मिनट पुराना है, नया सिग्नल नहीं")
        bot.jsave("barrier_latest.json", prev); print(f"बाज़ार बंद या डेटा {age:.0f} मिनट पुराना: नया सिग्नल नहीं | live (paper):", prev["live_stats"]); return
    F, cols = features(h); o = outcomes(h); n = o["n"]; tsx = h.index; t_now = tsx[-1]
    mp = P("barrier_model.joblib"); mod = None
    if os.path.exists(mp):
        try:
            c = joblib.load(mp)
            if (time.time() - c["t"]) / 3600 < RETRAIN_H and c["cols"] == cols and c["K"] == K: mod = c["m"]
        except Exception: pass
    if mod is None:
        tr = train_rows(o["valid"], tsx[:n], t_now); w = np.linspace(.3, 1, len(tr)); X = F[cols].iloc[:n].values.astype(float)
        mod = Model().fit(X[tr], o["y"][tr], w); joblib.dump(dict(m=mod, cols=cols, K=K, t=time.time()), mp); log(f"model naya train hua ({len(tr)} rows)")
    p = float(mod.proba(F[cols].iloc[[-1]].values.astype(float))[0]); entry = float(h.c.iloc[-1]); bp = float(K * sigma_pip(h).iloc[-1])
    side = 1 if p >= THR_B else (-1 if p <= 1 - THR_B else 0)
    tp = entry + side * bp * PIP; sl = entry - side * bp * PIP
    name = {1: "खरीद (LONG)", -1: "बिक्री (SHORT)", 0: "कोई ट्रेड नहीं"}[side]
    msg = f"USDJPY {entry:.3f} | अगला 1 घंटा: {name} | स्कोर {p:.2f}"
    if side: msg += f" | लक्ष्य {tp:.3f} (+{bp:.1f} pip) | स्टॉप {sl:.3f} (-{bp:.1f} pip) | जोखिम:लाभ 1:1"
    preds = bot.jload("barrier_preds.json", []); resolve(h, preds)
    if not preds or preds[-1]["ts"] != str(t_now):
        preds.append(dict(ts=str(t_now), entry=entry, p=round(p, 3), side=side, tp=round(tp, 3), sl=round(sl, 3), bp=round(bp, 2)))
    last_alert = next((q for q in reversed(preds[:-1]) if q["side"]), None)
    st = live_stats(preds)
    bot.jsave("barrier_preds.json", preds[-5000:]); bot.jsave("barrier_latest.json", dict(time=str(t_now), price=entry, score=round(p, 3), signal=name, barrier_pip=round(bp, 1), tp=tp, sl=sl, horizon_min=H, live_stats=st, note="score asli sambhavna nahi, ranking hai; sirf paper-trading"))
    print(msg); print("live (paper) natija:", st)
    changed = (not last_alert) or last_alert["side"] != side or pd.Timestamp(last_alert["ts"]) < t_now - pd.Timedelta(minutes=55)
    if side and changed: bot.notify(msg, "USDJPY 1-ghanta barrier")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "backtest": backtest()
    elif cmd == "predict": predict()
    else: print(__doc__)
