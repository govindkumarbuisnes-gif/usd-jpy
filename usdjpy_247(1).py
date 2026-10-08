#!/usr/bin/env python3
"""
USDJPY 24x7 intraday self-learning bot v2 (PAPER ONLY, koi trading salah nahi).

Kya karta hai (kabhi nahi rukta, har error par khud sambhal kar chalta rehta hai):
  1. Har minute USDJPY ka live 1m candle laata hai, candle ka naam batata hai (Doji, Hammer, Engulfing...).
  2. Related markets ka 1m data: DXY, US10Y, Nikkei, S&P futures, VIX, Gold, Oil, EURUSD.
  3. News (Google News RSS) + economic calendar (USD/JPY events) ko score karta hai: usd / yen / risk sentiment.
  4. ~200 features: basic + advanced + professional formulas, 5m/15m/1h multi-timeframe, news, cross-market.
  5. 3 horizons par predict karta hai: 1, 5, 15 minute. Har horizon ka alag model.
  6. Har naye bar par online learning + har RETRAIN_MIN minute me retrain (naya data zyada wazan, purana bhi yaad).
  7. Khud naye formulas khojta hai (USDJPY + cross-market), holdout par test karke rakhta hai.
  8. Accuracy lagatar: overall / last100 / last500 / confident-only, p-value, paper PnL, horizon ke hisaab se.
  9. NAYA: sirf up/down nahi, price bhi predict karta hai (jaise 1 ghante baad ~kitna, range ke saath) aur baad me asli price se milata hai.
 10. NAYA: news/event ka nuksan-risk: LONG/SHORT me agle 1 ghante me kitna pip tak ka nuksan ho sakta hai, kaaran aur sambhavit nuksan (Hindi).
 11. NAYA: badi-chaal chetavani (agle 15 min/1 ghante me badi giravat ya uchhal ka khatra, walk-forward AUC 0.73-0.85) + "badhat apramanit" tag
       (agar model 'hamesha ooper/neeche' wale sade anuman se behtar nahi to alert me likhta hai).
 12. NAYA: khabrein ab 7 RSS (Fed, MarketWatch, CNBC, Yahoo, ECB, BOJ...) + 10 Google News search se; har headline ka vishay (Fed/BOJ/intervention/jobs...),
       zaroorat (1-3) aur USDJPY par disha; bot seekhta hai ki kis vishay ki khabar ke baad price kitna hilta hai; zaroori khabar par phone alert.
       Apni RSS jodne ke liye feeds.txt banayein (har line: naam,url ya sirf url).
  *. formulas.txt me aap koi bhi formula likh sakte hain ("naam = expression", o,h,l,c,r,np,pd).

Chalane ka tarika:
  pip install yfinance pandas numpy scikit-learn joblib
  python usdjpy_247.py            (ya ./run_forever.sh jo crash par bhi restart karta hai)
Data source: OANDA_TOKEN (practice token) ho to OANDA, warna yfinance; TWELVE_KEY backup. BACKFILL_DAYS=30 se purani history jud jati hai
  (OANDA ya free Dukascopy se). Sirf history jodni ho: python usdjpy_247.py backfill 30
  HistData ki apni CSV jodni ho: python usdjpy_247.py import DAT_MT_USDJPY_M1_202601.csv DAT_MT_USDJPY_M1_202602.csv ...
Test (bina internet): SYNTH=1 MAX_CYCLES=4 SLEEP=0 RETRAIN_MIN=0 python usdjpy_247.py
Phone alert: NTFY_TOPIC=<lamba-anjaan-naam> python usdjpy_247.py
Data source badalna: DATA_SOURCE=twelvedata TWELVE_KEY=<apni-key> python usdjpy_247.py  (default: yahoo)
Purani history: import_dukascopy.py se Dukascopy ka free 1m data pehle bars.csv me daal sakte hain.
"""
import os, sys, re, json, time, math, zlib, lzma, logging, traceback, datetime as dt, urllib.request, urllib.parse, warnings
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from logging.handlers import RotatingFileHandler
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
E = os.environ.get
SYM = E("SYMBOL", "USDJPY=X")
DIR = E("DATA_DIR", "usdjpy_data")
HORIZONS = [int(x) for x in E("HORIZONS", "1,5,15,60").split(",")]
THR = float(E("THR", "0.55"))
COST = float(E("COST_PCT", "0.002"))
RETRAIN_MIN = float(E("RETRAIN_MIN", "60"))
DISCOVER_EVERY = int(E("DISCOVER_EVERY", "3"))
NEWS_EVERY = float(E("NEWS_EVERY_MIN", "5"))
MAX_TRAIN = int(E("MAX_TRAIN", "60000"))
PIP = 0.01                                     # USDJPY me 1 pip = 0.01 yen
MIN_RISK_N = int(E("MIN_RISK_N", "150"))
MIN_BIG_N = int(E("MIN_BIG_N", "5000"))
BIG_H = [int(x) for x in E("BIG_HORIZONS", "15,60").split(",")]   # badi-chaal chetavani ke horizons
LIVE_WIN = 6000
MAX_DISC = 60
SYNTH = E("SYNTH")
SRC = E("DATA_SOURCE", "yahoo")                    # yahoo | twelvedata
TD_KEY = E("TWELVE_KEY"); TD_EVERY = float(E("TD_EVERY_MIN", "2"))   # free plan: 800 request/din => 2 minute me ek baar
MAX_CYCLES = int(E("MAX_CYCLES", "0"))
MAX_RUN = float(E("MAX_RUNTIME_MIN", "0"))        # GitHub Actions ke liye: itne minute baad saaf tarah band ho
NTFY = E("NTFY_TOPIC")
OANDA_TOKEN = E("OANDA_TOKEN"); OANDA_HOST = E("OANDA_HOST", "api-fxpractice.oanda.com")   # practice (demo) token
TWELVE_KEY = E("TWELVE_KEY"); BACKFILL_DAYS = int(E("BACKFILL_DAYS", "0"))
CAL_URL = E("CAL_URL", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
EXT = {"DXY": "DX-Y.NYB", "US10Y": "^TNX", "N225": "^N225", "ES": "ES=F", "VIX": "^VIX", "GOLD": "GC=F", "OIL": "CL=F", "EURUSD": "EURUSD=X"}
WINS = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
os.makedirs(DIR, exist_ok=True)
P = lambda n: os.path.join(DIR, n)

LOGGER = logging.getLogger("bot"); LOGGER.setLevel(logging.INFO)
for _h in (logging.StreamHandler(sys.stdout), RotatingFileHandler(P("bot.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8")):
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")); LOGGER.addHandler(_h)
log = LOGGER.info


def jload(n, default):
    try: return json.load(open(P(n)))
    except Exception: return default


def jsave(n, o): json.dump(o, open(P(n), "w"), indent=1, default=str)


# ================================================================ GLOBAL STATE (memory me)
HIST = None; EXT_DF = pd.DataFrame(); NEWS_DF = pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]"), "title": pd.Series(dtype=object), "src": pd.Series(dtype=object), "usd": pd.Series(dtype=float), "jpy": pd.Series(dtype=float), "risk": pd.Series(dtype=float), "topics": pd.Series(dtype=object), "imp": pd.Series(dtype=float), "dirsc": pd.Series(dtype=float)}); CAL = []; CH = {}; RISK = {}; BIG = {}


# ================================================================ SYNTHETIC (sirf test ke liye)
_cnt = [0]; _last = [None]
def synth_bars():
    rng = np.random.default_rng(1); n = 12000
    c = 150 * np.exp(np.cumsum(rng.normal(0, 0.00008, n))); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 3e-5, n))); l = np.minimum(o, c) * (1 - abs(rng.normal(0, 3e-5, n)))
    d = pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=pd.date_range("2026-01-05", periods=n, freq="min"))
    k = min(n, 6000 + _cnt[0] * 5); _cnt[0] += 1; _last[0] = d.index[k - 1]
    return d.iloc[:k]


def synth_ext(d):
    r = d.c.pct_change().fillna(0).values; out = {}
    for k in EXT:
        rng = np.random.default_rng(zlib.crc32(k.encode())); nz = rng.normal(0, 1e-4, 12000)[:len(d)]
        out[k] = 100 * np.exp(np.cumsum(0.4 * r + nz))
    return pd.DataFrame(out, index=d.index)


def synth_news():
    rng = np.random.default_rng(7)
    ph = ["Fed Powell hawkish rate hike signals", "Bank of Japan Ueda dovish holds rates", "Japan Ministry of Finance warns yen intervention", "US CPI inflation hot beats forecast",
          "nonfarm payrolls jobs beat expectations", "oil prices surge OPEC cut", "war missile attack raises tensions", "ceasefire talks progress", "stocks rally record high Wall Street",
          "Treasury yields fall", "tariff trade war escalates", "yen weakens dollar gains"]
    t = pd.Timestamp("2026-01-05") + pd.to_timedelta(rng.integers(0, 12000, 700), unit="m")
    df = pd.DataFrame({"ts": t, "title": [f"{ph[i % len(ph)]} #{j}" for j, i in enumerate(rng.integers(0, 999, 700))]})
    return df[df.ts <= _last[0]]


def synth_cal():
    base = pd.Timestamp("2026-01-05"); return [dict(ts=str(base + pd.Timedelta(minutes=700 + 1500 * i)), country="USD", impact="High", title=f"Synth event {i}") for i in range(8)]


# ================================================================ DATA: USDJPY
_td_last = [0.0]
def fetch_twelve():
    """Twelve Data free plan (800 request/din, 8/min): har TD_EVERY minute me aakhri 20 bars, pehli baar 5000 bars."""
    empty = pd.DataFrame({k: pd.Series(dtype=float) for k in "ohlc"}, index=pd.DatetimeIndex([]))
    if not TD_KEY: raise RuntimeError("TWELVE_KEY set nahi hai")
    first = len(HIST) < 5000
    if not first and time.time() - _td_last[0] < TD_EVERY * 60 - 5: return empty
    q = urllib.parse.urlencode(dict(symbol="USD/JPY", interval="1min", outputsize=5000 if first else 20, timezone="UTC", apikey=TD_KEY))
    j = json.loads(http("https://api.twelvedata.com/time_series?" + q, 30)); _td_last[0] = time.time()
    if j.get("status") == "error" or "values" not in j: raise RuntimeError(f"twelvedata: {j.get('message', j)}")
    d = pd.DataFrame(j["values"]); d.index = pd.to_datetime(d["datetime"])
    d = d[["open", "high", "low", "close"]].astype(float).sort_index(); d.columns = list("ohlc")
    return d.iloc[:-1]


def fetch_yf(period):
    if SRC == "twelvedata": return fetch_twelve()
    import yfinance as yf
    d = yf.download(SYM, period=period, interval="1m", auto_adjust=True, progress=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d = d[["Open", "High", "Low", "Close"]].dropna(); d.columns = list("ohlc")
    if d.index.tz is not None: d.index = d.index.tz_convert("UTC").tz_localize(None)
    return d.iloc[:-1]


def empty_bars(): return pd.DataFrame({k: pd.Series(dtype=float) for k in "ohlc"}, index=pd.DatetimeIndex([]))


def oanda_candles(count=300, to=None):
    """OANDA v20 M1 mid candles (sirf poori band hui). Practice token kaafi hai."""
    q = {"granularity": "M1", "price": "M", "count": count}
    if to is not None: q["to"] = str(int(pd.Timestamp(to).tz_localize("UTC").timestamp()))
    req = urllib.request.Request(f"https://{OANDA_HOST}/v3/instruments/USD_JPY/candles?" + urllib.parse.urlencode(q), headers={"Authorization": "Bearer " + OANDA_TOKEN})
    j = json.loads(urllib.request.urlopen(req, timeout=25).read())
    rows = [(pd.to_datetime(c["time"], utc=True).tz_localize(None), float(c["mid"]["o"]), float(c["mid"]["h"]), float(c["mid"]["l"]), float(c["mid"]["c"])) for c in j.get("candles", []) if c.get("complete")]
    return pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]).set_index("t") if rows else empty_bars()


_t12 = [0.0]
def twelve_bars():
    """Twelve Data free plan: 800 calls/din, isliye har 130 sec me ek call (sirf backup)."""
    if time.time() - _t12[0] < 130: return empty_bars()
    _t12[0] = time.time()
    j = json.loads(http("https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(dict(symbol="USD/JPY", interval="1min", outputsize=60, timezone="UTC", apikey=TWELVE_KEY))))
    v = pd.DataFrame(j["values"]); v.index = pd.to_datetime(v.pop("datetime")); d = v[["open", "high", "low", "close"]].astype(float); d.columns = list("ohlc")
    return d.sort_index().iloc[:-1]                                             # aakhri candle abhi ban rahi hai


def fetch(period):
    """Data source ka order: OANDA (agar token ho) -> yfinance -> Twelve Data (agar key ho)."""
    if SYNTH: return synth_bars()
    for name, fn in (("OANDA", lambda: oanda_candles(5000 if period == "7d" else 300) if OANDA_TOKEN else None),
                     ("yfinance", lambda: fetch_yf(period)), ("TwelveData", lambda: twelve_bars() if TWELVE_KEY else None)):
        try:
            d = fn()
            if d is not None and len(d): return d
        except Exception as e: log(f"{name} data error: {e!r}")
    return empty_bars()


def duka_hour(ts):
    """Dukascopy ka ek ghante ka tick file (.bi5) -> 1m OHLC (bid/ask ka mid)."""
    try: raw = http(f"https://datafeed.dukascopy.com/datafeed/USDJPY/{ts.year}/{ts.month - 1:02d}/{ts.day:02d}/{ts.hour:02d}h_ticks.bi5", 30)
    except Exception: return None
    if not raw: return None
    try: data = lzma.decompress(raw)
    except Exception:
        try: data = lzma.LZMADecompressor().decompress(raw)
        except Exception: return None
    n = len(data) // 20
    if n == 0: return None
    a = np.frombuffer(data[:n * 20], dtype=np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("av", ">f4"), ("bv", ">f4")]))
    mid = (a["ask"].astype("float64") + a["bid"].astype("float64")) / 2; mid = mid / (1000.0 if 20 < mid.mean() / 1000 < 400 else 100000.0)
    t = pd.DatetimeIndex(ts + pd.to_timedelta(a["ms"].astype("int64"), unit="ms"))
    g = pd.Series(mid, index=t).resample("1min").ohlc().dropna(); g.columns = list("ohlc"); return g


def backfill_duka(days):
    from concurrent.futures import ThreadPoolExecutor
    end = pd.Timestamp.utcnow().tz_localize(None).floor("h") - pd.Timedelta(hours=2)
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(duka_hour, pd.date_range(end - pd.Timedelta(days=days), end, freq="h")))
    res = [r for r in res if r is not None and len(r)]
    return pd.concat(res).sort_index() if res else empty_bars()


def backfill_oanda(days):
    out = []; to = None; limit = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    for _ in range(80):
        d = oanda_candles(5000, to)
        if not len(d): break
        out.append(d); to = d.index[0]
        if to <= limit: break
    return pd.concat(out).sort_index() if out else empty_bars()


def import_histdata(paths):
    """HistData.com ke CSV (MT ya NT format; unka samay EST bina DST hota hai) ko UTC me badalkar bars.csv me jodta hai."""
    parts = []
    for p in paths:
        first = open(p, encoding="utf-8", errors="ignore").readline()
        if ";" in first:                                                         # NT: 20260101 170400;o;h;l;c;v
            x = pd.read_csv(p, sep=";", header=None, names=["t", "o", "h", "l", "c", "v"]); ts = pd.to_datetime(x.t, format="%Y%m%d %H%M%S")
        else:                                                                    # MT: 2026.01.01,17:04,o,h,l,c,v
            x = pd.read_csv(p, header=None, names=["dt", "tm", "o", "h", "l", "c", "v"]); ts = pd.to_datetime(x.dt + " " + x.tm, format="%Y.%m.%d %H:%M")
        x.index = ts + pd.Timedelta(hours=5); parts.append(x[list("ohlc")]); log(f"{os.path.basename(p)}: {len(x)} bars padhe")
    h = update_hist(pd.concat(parts).sort_index()); log(f"import poora: kul bars={len(h)} ({h.index[0]} se {h.index[-1]} UTC)")


def backfill(days):
    """Purani 1-minute history jodo: pehle OANDA (token ho to), warna Dukascopy (free)."""
    log(f"backfill shuru: pichhle {days} din")
    d = empty_bars()
    if OANDA_TOKEN:
        try: d = backfill_oanda(days)
        except Exception as e: log(f"OANDA backfill error {e!r}")
    if not len(d):
        try: d = backfill_duka(days)
        except Exception as e: log(f"Dukascopy backfill error {e!r}")
    if len(d): h = update_hist(d); log(f"backfill poora: {len(d)} bars joda, kul bars={len(h)}")
    else: log("backfill: koi data nahi mila (internet/source check karein)")


def load_hist():
    try: return pd.read_csv(P("bars.csv"), index_col=0, parse_dates=True)
    except Exception: return pd.DataFrame({k: pd.Series(dtype=float) for k in "ohlc"}, index=pd.DatetimeIndex([]))


def update_hist(new):
    h = pd.concat([load_hist(), new]).astype(float); h = h[~h.index.duplicated(keep="last")].sort_index().iloc[-250000:]
    h.to_csv(P("bars.csv")); return h


def target(d, H):
    """H minute baad ka return %. Weekend/gap wale bar ko ginta nahi."""
    nxt = d.c.shift(-H) / d.c * 100 - 100; ix = d.index.to_series(); gap = ix.shift(-H) - ix
    return nxt.where(gap <= pd.Timedelta(minutes=H + 3))


# ================================================================ DATA: related markets
def fetch_ext(d):
    if SYNTH: return synth_ext(d)
    import yfinance as yf
    x = yf.download(list(EXT.values()), period="7d" if EXT_DF.empty else "1d", interval="1m", auto_adjust=True, progress=False)
    c = x["Close"].rename(columns={v: k for k, v in EXT.items()})
    if c.index.tz is not None: c.index = c.index.tz_convert("UTC").tz_localize(None)
    return c.dropna(how="all").iloc[:-1]


def update_ext(d):
    global EXT_DF
    new = fetch_ext(d)
    if new is None or not len(new): return
    old = EXT_DF if len(EXT_DF) else (pd.read_csv(P("ext.csv"), index_col=0, parse_dates=True) if os.path.exists(P("ext.csv")) else pd.DataFrame())
    EXT_DF = new.combine_first(old).sort_index().iloc[-250000:]; EXT_DF.to_csv(P("ext.csv"))


def aligned_ext(idx):
    if EXT_DF.empty: return pd.DataFrame(index=idx)
    return EXT_DF.reindex(EXT_DF.index.union(idx)).ffill(limit=30).reindex(idx)


# ================================================================ DATA: news + calendar
USD_UP = ["hawkish", "rate hike", "hikes rates", "higher for longer", "strong jobs", "jobs beat", "payrolls beat", "hot inflation", "inflation accelerates", "inflation rises", "yields rise", "yields jump", "yields climb", "dollar rallies", "dollar gains", "dollar surges", "strong dollar", "no rate cut", "delays rate cut"]
USD_DN = ["dovish", "rate cut", "cuts rates", "weak jobs", "jobs miss", "payrolls miss", "inflation cools", "inflation slows", "yields fall", "yields drop", "dollar slips", "dollar falls", "dollar weakens", "dollar slides", "recession", "slowdown", "unemployment rises"]
JPY_UP = ["boj hike", "boj raises", "boj hawkish", "ueda hawkish", "yen rallies", "yen surges", "yen gains", "yen strengthens", "yen jumps", "intervention", "intervene", "safe haven", "rate hike japan"]
JPY_DN = ["yen weakens", "yen slumps", "yen falls", "yen drops", "yen slides", "yen tumbles", "weak yen", "boj dovish", "boj holds", "ultra-loose", "easy policy"]
RISK_ON = ["stocks rally", "stocks rise", "record high", "risk-on", "risk appetite", "equities gain", "nikkei rises", "wall street gains"]
RISK_OFF = ["stocks fall", "stocks slump", "sell-off", "selloff", "risk-off", "plunge", "crash", "geopolitical", "war", "tensions", "panic", "volatility surges", "nikkei falls", "wall street drops"]
NEWS_FEEDS = [("Fed", "https://www.federalreserve.gov/feeds/press_all.xml"), ("Fed-speeches", "https://www.federalreserve.gov/feeds/speeches.xml"),
              ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories"), ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
              ("Yahoo", "http://finance.yahoo.com/rss/topstories"), ("ECB", "https://www.ecb.europa.eu/rss/pub.html"),
              ("BOJ", "https://www.boj.or.jp/en/rss/whatsnew.xml")]          # BOJ ka URL verify nahi hua; na chale to apne aap chhod diya jata hai
NEWS_Q = ['USDJPY OR "USD/JPY" OR yen dollar', "Bank of Japan OR BOJ OR Ueda yen", "Japan Ministry of Finance yen intervention OR currency",
          "Federal Reserve OR Fed OR Powell interest rate", "US CPI OR payrolls OR inflation OR jobs report OR GDP OR PMI", "Treasury yields OR dollar index OR Fed rate expectations",
          "Nikkei OR TOPIX OR Wall Street stocks", "tariffs OR trade war OR sanctions Japan US", "oil prices OR OPEC OR gold prices",
          "geopolitical OR war OR Middle East OR Ukraine OR Taiwan OR North Korea markets"]
TOPICS = {"fed": (["fed", "fomc", "federal reserve", "powell", "dot plot", "fed funds"], 3), "boj": (["bank of japan", "boj", "ueda", "yield curve control", "japanese central bank"], 3),
          "inflation": (["cpi", "inflation", "pce", "ppi", "consumer prices", "producer prices"], 2),
          "jobs": (["nonfarm", "non-farm", "payroll", "payrolls", "jobless claims", "unemployment", "jobs report", "adp", "labor market", "labour market"], 3),
          "growth": (["gdp", "pmi", "ism", "retail sales", "industrial production", "consumer confidence", "recession", "economic growth"], 2),
          "yields": (["treasury yield", "treasury yields", "10-year yield", "bond yield", "bond yields", "jgb", "yields"], 2),
          "intervention": (["intervention", "intervene", "ministry of finance", "mof", "currency check", "rate check", "stealth intervention", "yen buying"], 3),
          "trade": (["tariff", "tariffs", "trade war", "trade deal", "trade talks", "export controls", "sanctions"], 2),
          "geopolitics": (["war", "missile", "attack", "conflict", "ceasefire", "military", "invasion", "north korea", "taiwan", "middle east", "iran", "israel", "russia", "ukraine"], 2),
          "oil": (["oil", "crude", "opec", "brent", "wti"], 1),
          "risk": (["stocks", "wall street", "nikkei", "s&p 500", "dow", "nasdaq", "vix", "sell-off", "selloff", "rally", "risk-off", "risk-on", "equities"], 1),
          "dollar_yen": (["usd/jpy", "usdjpy", "dollar-yen", "yen", "dollar index", "dxy", "greenback"], 2),
          "gov": (["election", "prime minister", "treasury secretary", "bessent", "government shutdown", "debt ceiling"], 1)}
TOPIC_RX = {k: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in kws) + r")\b", re.I) for k, (kws, wt) in TOPICS.items()}
TOPIC_HI = {"fed": "Fed", "boj": "BOJ", "inflation": "महँगाई", "jobs": "रोज़गार", "growth": "आर्थिक वृद्धि", "yields": "यील्ड", "intervention": "हस्तक्षेप (intervention)",
            "trade": "व्यापार/टैरिफ", "geopolitics": "भू-राजनीति", "oil": "तेल", "risk": "बाज़ार-मूड", "dollar_yen": "डॉलर-येन", "gov": "सरकार/राजनीति"}
OFFICIAL = {"Fed", "Fed-speeches", "ECB", "BOJ"}
NEWS_COLS = ["ts", "title", "src", "usd", "jpy", "risk", "topics", "imp", "dirsc"]
NEWS_IMPACT = {}; FEED_FAIL = {}


LEX_RX = {nm: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in L) + r")\b", re.I) for nm, L in
          (("uu", USD_UP), ("ud", USD_DN), ("ju", JPY_UP), ("jd", JPY_DN), ("ron", RISK_ON), ("roff", RISK_OFF))}


def score(title):
    n = lambda k: len({m.lower() for m in LEX_RX[k].findall(title)})            # poore shabd se milan ("award" me "war" nahi)
    return (float(np.clip(n("uu") - n("ud"), -3, 3)), float(np.clip(n("ju") - n("jd"), -3, 3)), float(np.clip(n("ron") - n("roff"), -3, 3)))


def analyze(title, src=""):
    """Ek headline: kaun sa vishay, kitna zaroori (1-3), USDJPY par disha (-3 neeche ... +3 ooper)."""
    u_, j_, r_ = score(title); tps = [k for k, rx in TOPIC_RX.items() if rx.search(title)]
    imp = max([TOPICS[k][1] for k in tps], default=0)
    if imp and src in OFFICIAL: imp = min(3, imp + 1)
    t = title.lower(); dr = u_ - j_ + 0.5 * r_
    if "intervention" in tps and not any(w in t for w in ("no intervention", "denies", "rules out")): dr -= 2
    if "geopolitics" in tps: dr += 1 if any(w in t for w in ("ceasefire", "peace", "truce", "de-escalat")) else -1
    return dict(usd=u_, jpy=j_, risk=r_, topics="|".join(tps), imp=float(imp), dirsc=float(np.clip(dr, -3, 3)))


def http(url, t=20):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=t).read()


def parse_feed(raw, name):
    """RSS 2.0 / RSS 1.0 (RDF) / Atom, teeno chalte hain."""
    root = ET.fromstring(raw); rows = []; now = pd.Timestamp.utcnow().tz_localize(None)
    for it in root.iter():
        if it.tag.split("}")[-1].lower() not in ("item", "entry"): continue
        g = lambda *ns: next((c.text for c in it if c.tag.split("}")[-1].lower() in ns and c.text), None)
        title = (g("title") or "").strip()
        if not title: continue
        try: ts = pd.to_datetime(g("pubdate", "date", "published", "updated"), utc=True).tz_localize(None)
        except Exception: ts = now
        if pd.isna(ts): ts = now
        if ts > now + pd.Timedelta(hours=1): continue
        rows.append(dict(ts=ts, title=title, src=name))
    return rows


def all_feeds():
    feeds = list(NEWS_FEEDS) + [("GoogleNews", "https://news.google.com/rss/search?q=" + urllib.parse.quote(q + " when:1d") + "&hl=en-US&gl=US&ceid=US:en") for q in NEWS_Q]
    for fp in ("feeds.txt", P("feeds.txt")):                                     # aap koi bhi RSS jod sakte hain: har line me "naam,url" ya sirf url
        try:
            for ln in open(fp, encoding="utf-8").read().splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("#"): continue
                head, sep, tail = ln.partition(",")
                feeds.append((head.strip(), tail.strip()) if sep and "/" not in head else ("custom", ln))
        except Exception: pass
    return feeds


def fetch_one(spec):
    name, url = spec
    if FEED_FAIL.get(url, (0, 0))[0] >= 5 and time.time() - FEED_FAIL[url][1] < 3600: return []
    try:
        rows = parse_feed(http(url, 15), name); FEED_FAIL[url] = (0, 0); return rows
    except Exception as e:
        n = FEED_FAIL.get(url, (0, 0))[0] + 1; FEED_FAIL[url] = (n, time.time())
        if n in (1, 5): log(f"news feed error ({name}, {n} baar): {e!r}"[:220])
        return []


def fetch_news():
    if SYNTH: return synth_news()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(fetch_one, all_feeds()))
    return pd.DataFrame([r for rr in res for r in rr])


def load_news():
    df = pd.read_csv(P("news_log.csv"), parse_dates=["ts"])
    if "imp" not in df:                                                           # purani file: naye tareeke se dobara analyze
        df = pd.concat([df[["ts", "title"]], pd.DataFrame([analyze(t, "") for t in df.title])], axis=1); df["src"] = "old"
    return clean_news(df)


def clean_news(df):
    for c in NEWS_COLS:
        if c not in df: df[c] = "" if c in ("src", "topics", "title") else 0.0
    for c in ("usd", "jpy", "risk", "imp", "dirsc"): df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["topics"] = df["topics"].fillna(""); df["src"] = df["src"].fillna(""); df["ts"] = pd.to_datetime(df["ts"])
    return df[NEWS_COLS].reset_index(drop=True)


def update_news():
    global NEWS_DF
    new = fetch_news()
    if new is None or not len(new): return
    if "src" not in new: new["src"] = "synth"
    new = pd.concat([new.reset_index(drop=True), pd.DataFrame([analyze(t, s_) for t, s_ in zip(new.title, new.src)])], axis=1)
    new = new[(new.imp > 0) | (new.usd != 0) | (new.jpy != 0) | (new.risk != 0)]   # sirf matlab ki khabrein
    if not len(NEWS_DF) and os.path.exists(P("news_log.csv")): NEWS_DF = load_news()
    NEWS_DF = clean_news(pd.concat([NEWS_DF, new])).drop_duplicates(subset=["title"]).sort_values("ts").iloc[-40000:].reset_index(drop=True)
    NEWS_DF.to_csv(P("news_log.csv"), index=False)


def learn_news_impact(h):
    """Pichhli khabron ke baad 15 min me price kitna hilta hai (vishay ke hisaab se), apne data se seekhta hai."""
    nd = NEWS_DF[NEWS_DF.imp > 0] if len(NEWS_DF) else NEWS_DF
    if len(nd) < 40 or len(h) < 2000: return
    hidx = h.index; c = h.c.values; n = len(h); hm = nd.ts.dt.floor("min").values
    pos = hidx.searchsorted(hm); p15 = np.minimum(pos + 15, n - 1); p0 = np.minimum(pos, n - 1)
    ok = (pos + 15 < n) & ((hidx[p0].values - hm) <= np.timedelta64(3, "m")) & ((hidx[p15].values - hidx[p0].values) <= np.timedelta64(18, "m"))
    nd = nd.assign(mv=(c[p15] - c[p0]) / PIP)[ok]
    base = float((h.c.diff(15).abs() / PIP).dropna().mean()); out = {}
    sets = [("all", pd.Series(True, index=nd.index))] + [(t, nd.topics.str.contains(rf"(?:^|\|){t}(?:\||$)", regex=True)) for t in TOPICS]
    for nm, m in sets:
        x = nd[m]
        if len(x) < 20: continue
        mv = x.mv.values; sd = mv.std() or 1.0; ds = x[x.dirsc != 0]
        out[nm] = dict(n=int(len(x)), mean=float(mv.mean()), t=float(mv.mean() / (sd / math.sqrt(len(x)))), abs=float(np.abs(mv).mean()), base_abs=base,
                       agree=float((np.sign(ds.dirsc.values) == np.sign(ds.mv.values)).mean()) if len(ds) >= 20 else None)
    NEWS_IMPACT.clear(); NEWS_IMPACT.update(out); jsave("news_impact.json", NEWS_IMPACT)
    log("[news-impact] seekha: " + ", ".join(f"{k}(n={v['n']}, |chaal|={v['abs']:.1f}pip)" for k, v in list(out.items())[:6]))


def top_news(now_ts, minutes=60, k=3):
    """Pichhle 1 ghante ki sabse zaroori khabrein, USDJPY par asar ke saath."""
    if not len(NEWS_DF): return [], []
    x = NEWS_DF[(NEWS_DF.ts > now_ts - pd.Timedelta(minutes=minutes)) & (NEWS_DF.ts <= now_ts) & (NEWS_DF.imp > 0)].sort_values(["imp", "ts"], ascending=False).head(k)
    lines, data = [], []
    for _, r in x.iterrows():
        tps = [t for t in r.topics.split("|") if t]; hi = "/".join(TOPIC_HI.get(t, t) for t in tps)
        dn = "ऊपर ↑" if r.dirsc > 0 else "नीचे ↓" if r.dirsc < 0 else "अनिश्चित"; st = NEWS_IMPACT.get(tps[0]) if tps else None
        hist = f" | इतिहास: ऐसी खबर के बाद 15 मिनट में औसत {st['abs']:.1f} pip चाल (सामान्य {st['base_abs']:.1f}, n={st['n']})" if st and st["n"] >= 30 else ""
        lines.append(f"   📰 [{hi}] {r.title[:110]} → USDJPY: {dn}{hist}")
        data.append(dict(ts=str(r.ts), title=r.title, topics=tps, importance=r.imp, direction=dn, src=r.src))
    return lines, data


def news_alerts(state, now_ts, now):
    """Bahut zaroori (imp=3) nayi khabar par phone alert (kam se kam 5 minute ke fasle par)."""
    if not len(NEWS_DF): return
    x = NEWS_DF[(NEWS_DF.ts > now_ts - pd.Timedelta(minutes=30)) & (NEWS_DF.imp >= 3)].sort_values("ts"); done = set(state.get("news_notified", []))
    for _, r in x.iterrows():
        key = r.title[:80]
        if key in done: continue
        done.add(key)
        if now - state.get("t_newsnotify", 0) < 300: continue
        tps = "/".join(TOPIC_HI.get(t, t) for t in r.topics.split("|") if t); dn = "ऊपर" if r.dirsc > 0 else "नीचे" if r.dirsc < 0 else "अनिश्चित"
        notify(f"[{tps}] {r.title[:140]} | USDJPY पर संभावित असर: {dn}", "ताज़ा खबर"); state["t_newsnotify"] = now
    state["news_notified"] = list(done)[-300:]


def update_cal():
    global CAL
    try:
        if SYNTH: new = synth_cal()
        else:
            new = [dict(ts=str(pd.Timestamp(e["date"]).tz_convert("UTC").tz_localize(None)), country=e["country"], impact=e["impact"], title=e["title"])
                   for e in json.loads(http(CAL_URL)) if e.get("country") in ("USD", "JPY") and e.get("impact") in ("High", "Medium")]
    except Exception as e: log(f"calendar error: {e!r}"); return
    seen = {(c["ts"], c["title"]) for c in CAL}; CAL.extend(c for c in new if (c["ts"], c["title"]) not in seen)
    CAL = sorted(CAL, key=lambda c: c["ts"])[-2000:]; jsave("calendar.json", CAL)


# ================================================================ CANDLE KA NAAM
def candle_info(w):
    o, h, l, c = [float(w[k].iloc[-1]) for k in "ohlc"]; rg = h - l
    if rg <= 0: return f"Flat candle | O{o:.3f} H{h:.3f} L{l:.3f} C{c:.3f}"
    body = abs(c - o) / rg; up = (h - max(o, c)) / rg; lo = (min(o, c) - l) / rg; bull = c > o
    if body < 0.1: nm = "Dragonfly Doji" if lo > 0.6 else "Gravestone Doji" if up > 0.6 else "Doji"
    elif body > 0.9: nm = ("Bullish" if bull else "Bearish") + " Marubozu"
    elif lo > 0.6 and body < 0.35: nm = "Hammer"
    elif up > 0.6 and body < 0.35: nm = "Shooting Star"
    elif body < 0.3: nm = "Spinning Top"
    else: nm = "Bullish candle" if bull else "Bearish candle"
    extra = ""
    if len(w) > 1:
        po, ph, pl, pc = [float(w[k].iloc[-2]) for k in "ohlc"]
        if bull and pc < po and c > po and o < pc: extra = " + Bullish Engulfing"
        elif (not bull) and pc > po and c < po and o > pc: extra = " + Bearish Engulfing"
        elif h < ph and l > pl: extra = " + Inside bar"
        elif h > ph and l < pl: extra = " + Outside bar"
    return f"{nm}{extra} | body {body*100:.0f}% up-wick {up*100:.0f}% low-wick {lo*100:.0f}% | O{o:.3f} H{h:.3f} L{l:.3f} C{c:.3f}"


# ================================================================ FORMULAS
def rsi(c, n):
    dl = c.diff(); u = dl.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); w = (-dl.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + u / w.replace(0, np.nan))


def base(d):
    """Basic + advanced + professional formulas. Sirf bar band hone tak ki jaankari."""
    o, h, l, c = d.o, d.h, d.l, d.c; f = {}; r = c.pct_change() * 100
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a14 = tr.ewm(alpha=1 / 14, adjust=False).mean(); atr = tr.rolling(14).mean() / c * 100
    rg = (h - l).replace(0, np.nan); tp = (h + l + c) / 3
    for n in (1, 2, 3, 5, 8, 13, 21, 34, 55): f[f"ret{n}"] = c.pct_change(n) * 100
    f["gap"] = (o - c.shift()) / c * 100; f["body"] = (c - o) / rg
    f["uwick"] = (h - np.maximum(o, c)) / rg; f["lwick"] = (np.minimum(o, c) - l) / rg; f["rng"] = (h - l) / c * 100
    f["ret1_z"] = r / atr
    for j in (1, 2, 3, 4, 5): f[f"ret1_z_l{j}"] = f["ret1_z"].shift(j)
    f["doji"] = ((c - o).abs() / rg < 0.1).astype(float); f["marubozu"] = ((c - o).abs() / rg > 0.9).astype(float)
    f["hammer"] = ((np.minimum(o, c) - l) / rg > 0.6).astype(float); f["shooting"] = ((h - np.maximum(o, c)) / rg > 0.6).astype(float)
    f["inside"] = ((h < h.shift()) & (l > l.shift())).astype(float); f["outside"] = ((h > h.shift()) & (l < l.shift())).astype(float)
    f["bull_engulf"] = ((c > o) & (o.shift() > c.shift()) & (c > o.shift()) & (o < c.shift())).astype(float)
    f["bear_engulf"] = ((c < o) & (o.shift() < c.shift()) & (c < o.shift()) & (o > c.shift())).astype(float)
    for n in (5, 10, 20, 50, 100, 200): f[f"sma{n}"] = (c / c.rolling(n).mean() - 1) * 100
    for a, b in ((5, 13), (9, 21), (21, 50), (50, 200)): f[f"ema{a}_{b}"] = (c.ewm(span=a).mean() / c.ewm(span=b).mean() - 1) * 100
    t1 = c.ewm(span=15).mean().ewm(span=15).mean().ewm(span=15).mean(); f["trix"] = t1.pct_change() * 1e4
    f["kama_er"] = (c - c.shift(10)).abs() / c.diff().abs().rolling(10).sum().replace(0, np.nan)
    for a, b, s in ((12, 26, 9), (5, 35, 5)):
        m = c.ewm(span=a).mean() - c.ewm(span=b).mean(); sg = m.ewm(span=s).mean(); f[f"macd{a}"] = m / c * 100; f[f"macdh{a}"] = (m - sg) / c * 100
    for n in (7, 14, 21): f[f"rsi{n}"] = rsi(c, n)
    rs = f["rsi14"]; f["stochrsi"] = (rs - rs.rolling(14).min()) / (rs.rolling(14).max() - rs.rolling(14).min()).replace(0, np.nan)
    ll, hh = l.rolling(14).min(), h.rolling(14).max(); k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    f["stoch_k"] = k; f["stoch_d"] = k.rolling(3).mean(); f["willr"] = k - 100
    f["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean() + 1e-12)
    f["roc10"] = c.pct_change(10) * 100; f["roc30"] = c.pct_change(30) * 100
    up, dn = h.diff(), -l.diff()
    pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / a14
    mdi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / a14
    f["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=1 / 14, adjust=False).mean(); f["di_diff"] = pdi - mdi
    f["aroon"] = h.rolling(26).apply(lambda x: x.argmax(), raw=True) * 4 - l.rolling(26).apply(lambda x: x.argmin(), raw=True) * 4
    f["vortex"] = ((h - l.shift()).abs().rolling(14).sum() - (l - h.shift()).abs().rolling(14).sum()) / tr.rolling(14).sum()
    f["chop"] = 100 * np.log10(tr.rolling(14).sum() / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)) / np.log10(14)
    f["atr"] = atr; f["atr_ratio"] = tr.rolling(5).mean() / tr.rolling(50).mean()
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pct"] = (c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan); f["bb_w"] = 4 * s20 / m20 * 100; f["z20"] = (c - m20) / s20.replace(0, np.nan)
    f["kelt"] = (c - c.ewm(span=20).mean()) / (2 * a14).replace(0, np.nan)
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min(); f["donch"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan) - 0.5
    for n in (10, 30): f[f"rv{n}"] = r.rolling(n).std()
    f["vol_ratio"] = f["rv10"] / f["rv30"]
    f["parkinson"] = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * math.log(2))) * 100
    gk = 0.5 * np.log(h / l) ** 2 - (2 * math.log(2) - 1) * np.log(c / o) ** 2; f["garman"] = np.sqrt(gk.rolling(20).mean().clip(lower=0)) * 100
    rsv = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o); f["rogers"] = np.sqrt(rsv.rolling(20).mean().clip(lower=0)) * 100
    f["skew30"] = r.rolling(30).skew(); f["kurt30"] = r.rolling(30).kurt(); f["ac1"] = r.rolling(30).corr(r.shift()); f["ac2"] = r.rolling(30).corr(r.shift(2))
    x = ((c - c.rolling(10).min()) / (c.rolling(10).max() - c.rolling(10).min()).replace(0, np.nan) - 0.5).clip(-0.499, 0.499) * 2
    f["fisher"] = 0.5 * np.log((1 + 0.999 * x) / (1 - 0.999 * x))
    e13 = c.ewm(span=13).mean(); f["elder_bull"] = (h - e13) / c * 100; f["elder_bear"] = (l - e13) / c * 100
    ts = (h.rolling(9).max() + l.rolling(9).min()) / 2; kj = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spa = ((ts + kj) / 2).shift(26); spb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    f["ichi_tk"] = (ts - kj) / c * 100; f["ichi_cloud"] = (c - (spa + spb) / 2) / c * 100
    f["piv"] = (c - tp) / tp * 100
    for n in (55, 233, 1440): f[f"fib{n}"] = (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min()).replace(0, np.nan)
    f["round50"] = (c / 0.5) % 1; f["round100"] = c % 1                                    # psychological levels (.00 / .50)
    hr, mn, dw = d.index.hour, d.index.minute, d.index.dayofweek
    f["h_sin"] = np.sin(2 * np.pi * hr / 24); f["h_cos"] = np.cos(2 * np.pi * hr / 24)
    f["m_sin"] = np.sin(2 * np.pi * mn / 60); f["m_cos"] = np.cos(2 * np.pi * mn / 60)
    f["d_sin"] = np.sin(2 * np.pi * dw / 5); f["d_cos"] = np.cos(2 * np.pi * dw / 5)
    f["tokyo"] = ((hr >= 0) & (hr < 9)).astype(float); f["london"] = ((hr >= 7) & (hr < 16)).astype(float); f["newyork"] = ((hr >= 12) & (hr < 21)).astype(float)
    return pd.DataFrame(f, index=d.index)


def mtf(d):
    """5m/15m/1h candles ke features. Sirf poori band hui higher-TF candle ka istemal (leak nahi)."""
    out = []
    for m in (5, 15, 60):
        g = d.resample(f"{m}min").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
        if len(g) < 30: continue
        c = g.c; r = c.pct_change() * 100; rg = (g.h - g.l).replace(0, np.nan)
        trr = pd.concat([g.h - g.l, (g.h - c.shift()).abs(), (g.l - c.shift()).abs()], axis=1).max(axis=1)
        m20, s20 = c.rolling(20).mean(), c.rolling(20).std(); mc = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        ft = pd.DataFrame({f"m{m}_ret": r, f"m{m}_rsi": rsi(c, 14), f"m{m}_ema": (c.ewm(span=9).mean() / c.ewm(span=21).mean() - 1) * 100,
                           f"m{m}_atr": trr.rolling(14).mean() / c * 100, f"m{m}_bb": (c - m20) / (2 * s20).replace(0, np.nan),
                           f"m{m}_macdh": (mc - mc.ewm(span=9).mean()) / c * 100, f"m{m}_body": (c - g.o) / rg,
                           f"m{m}_stoch": (c - g.l.rolling(14).min()) / (g.h.rolling(14).max() - g.l.rolling(14).min()).replace(0, np.nan)})
        ft.index = ft.index + pd.Timedelta(minutes=m - 1); ft = ft[~ft.index.duplicated()]
        out.append(ft.reindex(ft.index.union(d.index)).ffill(limit=m * 3).reindex(d.index))
    return pd.concat(out, axis=1) if out else pd.DataFrame(index=d.index)


def extf(d, e):
    """Related markets ke features: x_<naam>_..."""
    f = {}; r = d.c.pct_change() * 100
    for col in e.columns:
        s = e[col]; r1 = s.pct_change()
        for n in (1, 5, 15, 60): f[f"x_{col}_r{n}"] = s.pct_change(n) * 100
        f[f"x_{col}_corr"] = r.rolling(60).corr(r1); f[f"x_{col}_rel5"] = d.c.pct_change(5) * 100 - f[f"x_{col}_r5"]
    return pd.DataFrame(f, index=d.index)


def newsf(idx):
    """News sentiment + vishay + zaroorat + economic calendar ke features: n_..."""
    f = {}
    if len(idx):
        full = pd.date_range(idx[0].floor("min") - pd.Timedelta(hours=4), idx[-1].floor("min"), freq="min")
        nd = NEWS_DF[(NEWS_DF.ts >= full[0]) & (NEWS_DF.ts <= full[-1])].reset_index(drop=True) if len(NEWS_DF) else NEWS_DF
        mn = nd.ts.dt.floor("min") if len(nd) else None
        def ms(v, mask=None):
            if mn is None: return pd.Series(0.0, index=full)
            m = np.ones(len(nd), bool) if mask is None else np.asarray(mask, bool)
            return pd.Series(np.asarray(v, float)[m], index=mn[m]).groupby(level=0).sum().reindex(full, fill_value=0.0)
        ser = {}
        for k in ("usd", "jpy", "risk"):
            s = ms(nd[k]) if mn is not None else ms(None); ser[k] = s
            for W in (15, 60, 240): f[f"n_{k}{W}"] = s.rolling(W, min_periods=1).sum().reindex(idx)
        ones = np.ones(len(nd)); f["n_cnt60"] = ms(ones).rolling(60, min_periods=1).sum().reindex(idx)
        imp_s = ms(nd.imp) if mn is not None else ms(None); dir_s = ms(nd.imp * nd.dirsc) if mn is not None else ms(None)
        for W in (15, 60, 240): f[f"n_imp{W}"] = imp_s.rolling(W, min_periods=1).sum().reindex(idx); f[f"n_dir{W}"] = dir_s.rolling(W, min_periods=1).sum().reindex(idx)
        for tpc in TOPICS:
            m = nd.topics.str.contains(rf"(?:^|\|){tpc}(?:\||$)", regex=True) if mn is not None else None
            f[f"n_t_{tpc}60"] = ms(ones, m).rolling(60, min_periods=1).sum().reindex(idx)
        f["n_bias"] = (ser["usd"] - ser["jpy"] + 0.5 * ser["risk"]).ewm(halflife=60).mean().reindex(idx)
        ev = [c for c in CAL]; t = idx.values.astype("datetime64[ns]")
        for tag, evs in (("hi", [c for c in ev if c["impact"] == "High"]), ("any", ev)):
            arr = np.array(sorted(pd.Timestamp(c["ts"]).to_datetime64() for c in evs), dtype="datetime64[ns]")
            if len(arr):
                pos = np.searchsorted(arr, t, side="right")
                nx = np.where(pos < len(arr), (arr[np.minimum(pos, len(arr) - 1)] - t) / np.timedelta64(1, "m"), 1440.0)
                pv = np.where(pos > 0, (t - arr[np.maximum(pos - 1, 0)]) / np.timedelta64(1, "m"), 1440.0)
            else: nx = pv = np.full(len(t), 1440.0)
            nx, pv = np.minimum(nx, 1440.0), np.minimum(pv, 1440.0)
            f[f"n_ev_to_{tag}"] = nx; f[f"n_ev_since_{tag}"] = pv; f[f"n_ev_win_{tag}"] = ((nx <= 15) | (pv <= 15)).astype(float)
    return pd.DataFrame(f, index=idx)


def spec_feat(d, s, e):
    o, h, l, c = d.o, d.h, d.l, d.c; r = c.pct_change() * 100; t, a = s[0], s[1:]
    if t == "ret": return c.pct_change(a[0]) * 100
    if t == "smar": return (c / c.rolling(a[0]).mean() - 1) * 100
    if t == "emar": return (c.ewm(span=a[0]).mean() / c.ewm(span=a[1]).mean() - 1) * 100
    if t == "rsi": return rsi(c, a[0])
    if t == "z": return (c - c.rolling(a[0]).mean()) / c.rolling(a[0]).std().replace(0, np.nan)
    if t == "vol": return r.rolling(a[0]).std()
    if t == "volr": return r.rolling(a[0]).std() / r.rolling(a[1]).std().replace(0, np.nan)
    if t == "stoch": return (c - l.rolling(a[0]).min()) / (h.rolling(a[0]).max() - l.rolling(a[0]).min()).replace(0, np.nan)
    if t == "rank": return c.rolling(a[0]).rank(pct=True)
    if t == "skew": return r.rolling(a[0]).skew()
    if t == "lagret": return (c.pct_change(a[1]) * 100).shift(a[0])
    if t == "slope": return c.ewm(span=a[0]).mean().pct_change(3) * 100
    if t == "hlr": return ((h - l) / c * 100).rolling(a[0]).mean()
    if t == "ac": return r.rolling(a[0]).corr(r.shift())
    if t == "xret": return e[a[0]].pct_change(a[1]) * 100
    if t == "xcorr": return r.rolling(a[1]).corr(e[a[0]].pct_change())
    if t == "xz": return (e[a[0]] - e[a[0]].rolling(a[1]).mean()) / e[a[0]].rolling(a[1]).std().replace(0, np.nan)
    raise ValueError(t)


def rand_spec(rng):
    t = rng.choice(["ret", "smar", "emar", "rsi", "z", "vol", "volr", "stoch", "rank", "skew", "lagret", "slope", "hlr", "ac", "xret", "xcorr", "xz", "xret", "xcorr"])
    w = lambda lo=0: int(rng.choice([x for x in WINS if x >= lo]))
    if t in ("emar", "volr"): a, b = sorted(rng.choice(WINS, 2, replace=False)); return [t, int(a), int(b)]
    if t == "lagret": return [t, int(rng.integers(1, 9)), w()]
    if t in ("skew", "ac"): return [t, w(8)]
    if t.startswith("x"): return [t, str(rng.choice(list(EXT))), w(5)]
    return [t, w()]


def name_of(s): return "d_" + "_".join(map(str, s))


USER_BAD = set()
def user_formulas(d):
    out = {}
    try: lines = open(P("formulas.txt")).read().splitlines()
    except Exception: return out
    env = {"o": d.o, "h": d.h, "l": d.l, "c": d.c, "r": d.c.pct_change() * 100, "np": np, "pd": pd}
    for ln in lines:
        if "=" not in ln or ln.strip().startswith("#"): continue
        n, ex = ln.split("=", 1)
        try:
            v = eval(ex.strip(), {"__builtins__": {}}, env)
            if isinstance(v, pd.Series): out["u_" + n.strip()] = v.replace([np.inf, -np.inf], np.nan)
        except Exception as ee:
            if n not in USER_BAD: log(f"formulas.txt galat line: {n.strip()} {ee!r}"); USER_BAD.add(n)
    return out


def feats(d):
    e = aligned_ext(d.index)
    parts = [base(d), mtf(d), extf(d, e), newsf(d.index)]
    extra = {}
    for s in jload("discovered.json", []):
        try: extra[name_of(s)] = spec_feat(d, s, e)
        except Exception: pass
    extra.update(user_formulas(d))
    if extra: parts.append(pd.DataFrame(extra, index=d.index))
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


XN = ("x_", "n_")
def cols_of(F): return [c for c in F.columns if c.startswith(XN) or F[c].notna().mean() > 0.5]


def table(F, d, H, cols):
    A = F[cols].copy(); xn = [c for c in cols if c.startswith(XN)]; A[xn] = A[xn].fillna(0.0)
    A["nxt"] = target(d, H); A = A.dropna(); return A[A.nxt != 0]


# ================================================================ MODELS
def fit_models(X, y, w, yr):
    """Direction (up/down) ke 3 model + kitna upar/neeche (price-return %) ke 3 model."""
    sc = StandardScaler().fit(X); Z = np.clip(sc.transform(X), -8, 8)
    lr = LogisticRegression(C=0.05, max_iter=300).fit(Z, y, sample_weight=w)
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=w)
    sgd = SGDClassifier(loss="log_loss", alpha=1e-3, random_state=0).fit(Z, y, sample_weight=w)
    ysd = float(np.std(yr)) or 1e-6; t = yr / ysd
    rr = Ridge(alpha=50.0).fit(Z, t, sample_weight=w)
    rh = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(X, t, sample_weight=w)
    rs = SGDRegressor(alpha=1e-3, random_state=0).fit(Z, t, sample_weight=w)
    return dict(scaler=sc, lr=lr, hgb=hgb, sgd=sgd, rr=rr, rh=rh, rs=rs, ysd=ysd)


def proba(m, X):
    Z = np.clip(m["scaler"].transform(X), -8, 8)
    return np.mean([m["lr"].predict_proba(Z)[:, 1], m["hgb"].predict_proba(X)[:, 1], m["sgd"].predict_proba(Z)[:, 1]], axis=0)


def pred_ret(m, X):
    """H minute baad ka expected return (%)."""
    Z = np.clip(m["scaler"].transform(X), -8, 8)
    return np.mean([m["rr"].predict(Z), m["rh"].predict(X), m["rs"].predict(Z)], axis=0) * m["ysd"]


def live_acc(preds, H, n=300):
    done = [p for p in preds if p["H"] == H and p.get("hit") is not None][-n:]
    return (100 * np.mean([p["hit"] for p in done]), len(done)) if done else (None, 0)


def retrain_all(h, state, preds):
    d = h.iloc[-MAX_TRAIN:]; F = feats(d); cols = cols_of(F)
    try: build_risk(d, F)
    except Exception as e: log(f"risk table error {e!r}")
    try: learn_news_impact(h)
    except Exception as e: log(f"news-impact error {e!r}")
    ai = cols.index("atr") if "atr" in cols else None
    for H in HORIZONS:
        A = table(F, d, H, cols)
        if len(A) < 1000: log(f"[retrain H{H}] data kam: {len(A)}"); continue
        y = (A.nxt > 0).astype(int).values; X = A[cols].values; n = len(A); w = np.linspace(0.3, 1.0, n)   # naya zyada wazan, purana bhi yaad
        q1, q99 = A.nxt.quantile([.01, .99]); yr = A.nxt.clip(q1, q99).values
        cut = int(n * 0.8); mm = fit_models(X[:cut], y[:cut], w[:cut], yr[:cut]); hold = float(((proba(mm, X[cut:]) >= .5).astype(int) == y[cut:]).mean() * 100)
        maj = max(y[cut:].mean(), 1 - y[cut:].mean()) * 100; la, ln = live_acc(preds, H)
        pr = pred_ret(mm, X[cut:]); at = X[cut:, ai] if ai is not None else np.ones(len(pr)); at = np.where(at > 0, at, np.nan)
        res = (yr[cut:] - pr) / at; q10, q90 = [float(v) for v in np.nanquantile(res, [.1, .9])]
        mae = float(np.mean(np.abs(yr[cut:] - pr))); mae0 = float(np.mean(np.abs(yr[cut:]))); skill = 1 - mae / mae0
        better = H not in CH or "rr" not in CH[H] or ln < 100 or (hold >= la and hold > 50)
        log(f"[retrain H{H}] rows={n} features={len(cols)} holdout={hold:.2f}% (majority {maj:.2f}%) price-skill={skill*100:+.1f}% (0 se upar = 'koi badlav nahi' se behtar) live={'-' if la is None else round(la, 2)}% -> {'NAYA MODEL' if better else 'purana rakha'}")
        if better:
            m = fit_models(X, y, w, yr); atm = float(np.nanmedian(X[:, ai])) if ai is not None else 1.0
            m.update(cols=cols, hold_acc=hold, res_q=[q10, q90], hold_skill=skill, atr_med=atm, trained_at=str(dt.datetime.utcnow()), last_learn=str(A.index[-1]))
            CH[H] = m; joblib.dump(m, P(f"champion_{H}.joblib"))
    try: train_bigmove(d, F, cols)
    except Exception as e: log(f"bigmove error {e!r}")
    state["retrains"] = state.get("retrains", 0) + 1; state["last_retrain"] = time.time()


def discover(h, state, tries=30):
    """Khud naye formulas khojo (USDJPY + cross-market). Train/holdout dono me same disha + z>3 ho tabhi rakho."""
    d = h.iloc[-40000:]
    if len(d) < 5000: return
    e = aligned_ext(d.index); rng = np.random.default_rng(int(time.time())); disc = jload("discovered.json", []); have = {name_of(s) for s in disc}
    sg = {H: np.sign(target(d, H)) for H in HORIZONS}; cut = int(len(d) * 0.6); added = []
    for _ in range(tries):
        s = rand_spec(rng)
        if name_of(s) in have: continue
        try: v = spec_feat(d, s, e).replace([np.inf, -np.inf], np.nan)
        except Exception: continue
        if v.notna().sum() < 4000: continue
        v = v.clip(v.quantile(.01), v.quantile(.99))
        for H, sgn in sg.items():
            ok = v.notna() & sgn.notna() & (sgn != 0); a, b = ok.copy(), ok.copy(); a.iloc[cut:] = False; b.iloc[:cut] = False
            if a.sum() < 2000 or b.sum() < 2000: continue
            ct = np.corrcoef(v[a], sgn[a])[0, 1]; ch = np.corrcoef(v[b], sgn[b])[0, 1]
            if np.sign(ct) == np.sign(ch) and abs(ch) * math.sqrt(b.sum() / H) > 3.5 and abs(ct) > 0.005:
                disc.append(s); have.add(name_of(s)); added.append((name_of(s), f"H{H}", round(float(ch), 4))); break
    disc = disc[-MAX_DISC:]; jsave("discovered.json", disc); state["discovered_total"] = len(disc)
    log(f"[discover] {len(added)} naye formula mile: {added if added else '-'} | kul khoje hue: {len(disc)}")


# ================================================================ LIVE
def notify(msg, title):
    if not NTFY: return
    try: urllib.request.urlopen(urllib.request.Request("https://ntfy.sh/" + NTFY, data=msg.encode("utf-8"), headers={"Title": title}), timeout=15)
    except Exception as e: log(f"notify error {e!r}")


def resolve(h, preds):
    for p in preds[-1500:]:
        if p.get("hit") is not None or p.get("skip"): continue
        H = p["H"]; ts = pd.Timestamp(p["ts"]); i = h.index.searchsorted(ts)
        if i + H < len(h) and h.index[i] == ts:
            if h.index[i + H] - ts > pd.Timedelta(minutes=H + 3): p["skip"] = True; continue
            act = float(h.c.iloc[i + H]); p0 = p.get("p0", float(h.c.iloc[i])); mv = (act / p0 - 1) * 100
            if mv == 0: p["skip"] = True; continue
            p["hit"] = int((mv > 0) == (p["p"] >= .5)); side = 1 if p["p"] >= THR else -1 if p["p"] <= 1 - THR else 0
            p["pnl"] = round(side * mv - COST * (side != 0), 5)
            if "r" in p:                                                         # asli price vs predicted price
                tp = p0 * (1 + p["r"] / 100); p["act"] = round(act, 3); p["err"] = round(abs(tp - act) / PIP, 2); p["nerr"] = round(abs(p0 - act) / PIP, 2)
                p["in"] = int(p0 * (1 + p["lo"] / 100) <= act <= p0 * (1 + p["hi"] / 100))


def stats(preds, H):
    done = [p for p in preds if p["H"] == H and p.get("hit") is not None]; s = {"n": len(done)}
    if not done: return s
    hit = np.array([p["hit"] for p in done]); s["acc"] = round(100 * hit.mean(), 2)
    for n in (100, 500): s[f"acc{n}"] = round(100 * hit[-n:].mean(), 2)
    cf = [p for p in done if p["p"] >= THR or p["p"] <= 1 - THR]; s["conf_n"] = len(cf)
    if cf:
        k = np.array([p["hit"] for p in cf]); s["conf_acc"] = round(100 * k.mean(), 2)
        z = (k.mean() - .5) / math.sqrt(.25 / len(k)); s["conf_p"] = round(math.erfc(abs(z) / math.sqrt(2)), 4); s["pnl%"] = round(float(sum(p["pnl"] for p in cf)), 3)
    last = done[-500:]; ups = np.array([(p["p"] >= .5) == bool(p["hit"]) for p in last])      # asal me 'ooper' gaya kya
    maj = max(ups.mean(), 1 - ups.mean()) * 100; s["edge_pt"] = round(float(100 * np.mean([p["hit"] for p in last]) - maj), 2)   # model - sada anuman (ank)
    pe = [p for p in done if "err" in p][-500:]
    if pe:                                                                       # price prediction ki asli parakh
        s["mae_pip"] = round(float(np.mean([p["err"] for p in pe])), 2); s["naive_pip"] = round(float(np.mean([p["nerr"] for p in pe])), 2)
        s["range_cover%"] = round(100 * float(np.mean([p["in"] for p in pe])), 1)
    return s


def lab(H): return "1 घंटा" if H == 60 else f"{H} मिनट"
HSIG = {"LONG": "खरीद (LONG)", "SHORT": "बिक्री (SHORT)", "NO TRADE": "कोई ट्रेड नहीं"}


def build_risk(d, F, H=60):
    """News/event ke samay pichhle data me agle 60 min me sabse bada nuksan kitna hua (95% maamlon me)."""
    c = d.c; fl = d.l[::-1].rolling(H, min_periods=H).min()[::-1].shift(-1); fh = d.h[::-1].rolling(H, min_periods=H).max()[::-1].shift(-1)
    long_loss = (1 - fl / c) * 100; short_loss = (fh / c - 1) * 100             # % me, positive = nuksan
    ok = target(d, H).notna() & long_loss.notna()
    z = lambda k: F[k].fillna(0) if k in F else pd.Series(0.0, index=d.index)
    ev, b, cnt = z("n_ev_win_hi"), z("n_bias"), z("n_cnt60"); b9, b1, c9 = float(b.quantile(.9)), float(b.quantile(.1)), float(cnt.quantile(.9))
    masks = {"normal": ok & (ev == 0), "event": ok & (ev == 1), "usd_up_news": ok & (b >= b9) & (b > 0),
             "usd_down_news": ok & (b <= b1) & (b < 0), "high_news_flow": ok & (cnt >= c9) & (cnt > 0)}
    tab = {}
    for k, m in masks.items():
        if int(m.sum()) >= MIN_RISK_N:
            tab[k] = dict(n=int(m.sum()), long_q95=float(long_loss[m].quantile(.95)), short_q95=float(short_loss[m].quantile(.95)),
                          long_med=float(long_loss[m].median()), short_med=float(short_loss[m].median()))
    RISK.clear(); RISK.update(tab); RISK["_th"] = dict(b9=b9, b1=b1, c9=c9); jsave("risk.json", RISK)
    log("[risk] news-regime table: " + ", ".join(f"{k}(n={v['n']})" for k, v in tab.items()) if tab else "[risk] abhi news history kam hai, ATR se andaza lagega")


def risk_report(last, price, ne, mins):
    """News se hone wale nuksan ka hisaab (LONG aur SHORT dono ke liye) + kaaran."""
    g = lambda k: float(last.get(k, 0) or 0); th = RISK.get("_th", {}); b = g("n_bias")
    ev_now = g("n_ev_win_hi") == 1 or (ne is not None and ne["impact"] == "High" and mins is not None and mins <= 15)
    active = ["normal"] + (["event"] if ev_now else []) + (["usd_up_news"] if th and b >= th["b9"] and b > 0 else []) \
        + (["usd_down_news"] if th and b <= th["b1"] and b < 0 else []) + (["high_news_flow"] if th and g("n_cnt60") >= th["c9"] and g("n_cnt60") > 0 else [])
    pips = lambda pct: pct / 100 * price / PIP
    hist = [a for a in active if a in RISK]; note = ""
    if hist:
        lq = max(RISK[a]["long_q95"] for a in hist); sq = max(RISK[a]["short_q95"] for a in hist)
        base = RISK.get("normal"); mult = max(lq / base["long_q95"], sq / base["short_q95"]) if base else 1.0
    else:
        atr = g("atr") or 0.01; lq = sq = 1.8 * atr * math.sqrt(60) * (2.0 if ev_now else 1.0); mult = 2.0 if ev_now else 1.0; note = " (अंदाज़ा: news history अभी कम है)"
    causes = []
    if ne is not None and mins is not None and mins <= 120: causes.append(f"{ne['country']} {ne['title']} {mins} मिनट में ({ne['impact']} impact)")
    if th and b >= th["b9"] and b > 0: causes.append("ताज़ा news USD/USDJPY के ऊपर जाने के पक्ष में (ऊपर की ओर दबाव)")
    if th and b <= th["b1"] and b < 0: causes.append("ताज़ा news USD कमज़ोर / येन मज़बूत होने के पक्ष में (नीचे की ओर दबाव)")
    if g("x_VIX_r15") > 1.5: causes.append("VIX तेज़ी से बढ़ रहा है, risk-off में येन मज़बूत होने से USDJPY गिर सकता है")
    if g("x_US10Y_r15") < -0.8: causes.append("US 10-साल yield गिर रही है, USDJPY पर नीचे का दबाव")
    if g("x_N225_r15") < -0.3: causes.append("निक्केई कमज़ोर है, risk-off का संकेत")
    if g("x_DXY_r15") < -0.1: causes.append("डॉलर इंडेक्स गिर रहा है")
    if g("n_t_intervention60") > 0: causes.append("पिछले 1 घंटे में intervention / वित्त मंत्रालय (MOF) से जुड़ी खबर आई है")
    if g("n_t_fed60") > 0 or g("n_t_boj60") > 0: causes.append("पिछले 1 घंटे में Fed/BOJ से जुड़ी खबरें आई हैं")
    if abs(g("n_dir60")) >= 4: causes.append("खबरों का कुल झुकाव USDJPY के " + ("ऊपर" if g("n_dir60") > 0 else "नीचे") + " की ओर है")
    losses = []
    if ev_now or causes: losses += ["अचानक तेज़ उतार-चढ़ाव (spike) और उल्टी-सीधी चाल (whipsaw), stop-loss जल्दी लग सकता है",
                                    "spread चौड़ा हो सकता है और slippage (मनचाहे भाव पर सौदा न मिलना) हो सकता है"]
    if th and b <= th["b1"] and b < 0: losses.append("LONG पोज़िशन पर news से सीधा नुकसान का जोखिम ज़्यादा")
    if th and b >= th["b9"] and b > 0: losses.append("SHORT पोज़िशन पर news से सीधा नुकसान का जोखिम ज़्यादा")
    return dict(long_pips=round(pips(lq), 1), short_pips=round(pips(sq), 1), mult=round(mult, 2), note=note, causes=causes, losses=losses, bias=round(b, 2), active=active)


def fwd_extremes(d, H):
    """Agle H minute me sabse badi giravat / uchhal (pip), entry = abhi ka close."""
    fl = d.l[::-1].rolling(H, min_periods=H).min()[::-1].shift(-1); fh = d.h[::-1].rolling(H, min_periods=H).max()[::-1].shift(-1)
    ok = target(d, H).notna()
    return ((d.c - fl) / PIP).where(ok & fl.notna()), ((fh - d.c) / PIP).where(ok & fh.notna())


def train_bigmove(d, F, cols):
    """BADI CHAAL KI CHETAVANI: agle 15/60 min me train-data ki top-10% jitni badi giravat/uchhal hogi kya? (walk-forward test me AUC ~0.73-0.85)"""
    X = F[cols].copy(); xn = [c for c in cols if c.startswith(XN)]; X[xn] = X[xn].fillna(0.0)
    for H in BIG_H:
        dd, uu = fwd_extremes(d, H); ok = (X.notna().all(axis=1) & dd.notna() & uu.notna()).values; idx = np.flatnonzero(ok); n = len(idx)
        if n < MIN_BIG_N: log(f"[bigmove H{H}] data kam: {n}"); continue
        cut = int(n * 0.85); tr = idx[:cut:2]; ca = idx[cut + H:]                       # beech me H bar ka fasla (leak nahi)
        td, tu = float(np.percentile(dd.values[tr], 90)), float(np.percentile(uu.values[tr], 90))
        labs = {"down": (dd >= td).values, "up": (uu >= tu).values}; labs["any"] = labs["down"] | labs["up"]
        mods, q, base = {}, {}, {}
        for k, y in labs.items():
            mk = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
            m = mk.fit(X.values[tr], y[tr].astype(int)); pc = m.predict_proba(X.values[ca])[:, 1]
            mods[k] = m; q[k] = [float(np.quantile(pc, .75)), float(np.quantile(pc, .90))]; base[k] = float(y[tr].mean())
            if k == "any": auc = float(roc_auc_score(y[ca].astype(int), pc)) if 0 < y[ca].sum() < len(ca) else float("nan")
        BIG[H] = dict(m=mods, td=td, tu=tu, cols=cols, q=q, base=base, auc=auc); joblib.dump(BIG[H], P(f"bigmove_{H}.joblib"))
        log(f"[bigmove H{H}] badi chaal: giravat>={td:.1f} pip ya uchhal>={tu:.1f} pip | AUC(anadekha hissa)={auc:.3f} (0.5=bekaar, 0.7+ achha)")


def big_predict(F):
    out = {}
    for H, b in BIG.items():
        X = F.reindex(columns=b["cols"]); xn = [c for c in b["cols"] if c.startswith(XN)]; X[xn] = X[xn].fillna(0.0); row = X.iloc[[-1]].fillna(0.0).values
        pa, pdn, pup = (float(b["m"][k].predict_proba(row)[0, 1]) for k in ("any", "down", "up"))
        lvl = "ऊँचा" if pa >= b["q"]["any"][1] else "मध्यम" if pa >= b["q"]["any"][0] else "सामान्य"
        out[H] = dict(level=lvl, p_any=round(pa, 3), p_down=round(pdn, 3), p_up=round(pup, 3), base_any=round(b["base"]["any"], 3), down_pips=round(b["td"], 1), up_pips=round(b["tu"], 1), auc=round(b["auc"], 3))
    return out


def next_event():
    now = pd.Timestamp.utcnow().tz_localize(None) if not SYNTH else _last[0]
    for c in CAL:
        t = pd.Timestamp(c["ts"])
        if t > now: return c, int((t - now).total_seconds() // 60)
    return None, None


def cycle(state, preds):
    global HIST
    HIST = h = update_hist(fetch("7d" if len(HIST) < 5000 else "1d"))
    if len(h) < 1500: log(f"data jama ho raha hai: {len(h)}"); return
    now = time.time()
    try:
        if now - state.get("t_ext", 0) >= 55: update_ext(h); state["t_ext"] = now
    except Exception as e: log(f"ext error {e!r}")
    try:
        if now - state.get("t_news", 0) >= NEWS_EVERY * 60: update_news(); state["t_news"] = now
        if now - state.get("t_cal", 0) >= 3600: update_cal(); state["t_cal"] = now
    except Exception as e: log(f"news error {e!r}")
    resolve(h, preds)
    due = now - state.get("last_retrain", 0) >= RETRAIN_MIN * 60
    if not CH or due:
        if CH and state.get("retrains", 0) % DISCOVER_EVERY == 0: discover(h, state)
        retrain_all(h, state, preds)
    last_bar = str(h.index[-1])
    if last_bar == state.get("last_bar") and not due:
        if (pd.Timestamp.utcnow().tz_localize(None) - h.index[-1]) > pd.Timedelta(minutes=15) and not SYNTH: state["market"] = "band/data nahi aa raha"
        return
    state["market"] = "chalu"
    if not CH: return
    w = h.iloc[-LIVE_WIN:]; F = feats(w); ts = str(F.index[-1]); price = float(w.c.iloc[-1]); parts = []; sigs = {}; rec = {}
    for H in HORIZONS:
        m = CH.get(H)
        if not m or "rr" not in m: continue
        cols = m["cols"]; X = F.reindex(columns=cols); xn = [c for c in cols if c.startswith(XN)]; X[xn] = X[xn].fillna(0.0)
        tg = target(w, H); ok = X.notna().all(axis=1) & tg.notna() & (tg != 0) & (X.index > pd.Timestamp(m["last_learn"]))
        if ok.any():                                                             # online learning (direction + price dono)
            Z = np.clip(m["scaler"].transform(X[ok].values), -8, 8); m["sgd"].partial_fit(Z, (tg[ok] > 0).astype(int).values)
            m["rs"].partial_fit(Z, np.clip(tg[ok].values / m["ysd"], -5, 5))
            m["last_learn"] = str(X.index[ok][-1]); state["online_updates"] = state.get("online_updates", 0) + int(ok.sum())
            if state["online_updates"] % 20 < int(ok.sum()): joblib.dump(m, P(f"champion_{H}.joblib"))
        if not any(p["ts"] == ts and p["H"] == H for p in preds[-80:]):
            xr = X.iloc[[-1]].fillna(0.0); p = float(proba(m, xr.values)[0]); r = float(pred_ret(m, xr.values)[0])
            at = float(xr["atr"].iloc[0]) if "atr" in xr else m["atr_med"]; at = at if at > 0 else m["atr_med"]
            lo_r, hi_r = r + m["res_q"][0] * at, r + m["res_q"][1] * at
            preds.append(dict(ts=ts, H=H, p=round(p, 4), hit=None, p0=round(price, 3), r=round(r, 5), lo=round(lo_r, 5), hi=round(hi_r, 5)))
            sig = "LONG" if p >= THR else "SHORT" if p <= 1 - THR else "NO TRADE"; tp, lo, hi = price * (1 + r / 100), price * (1 + lo_r / 100), price * (1 + hi_r / 100)
            sigs[H] = (sig, p, tp, lo, hi); st = stats(preds, H); rec[f"H{H}"] = dict(p_up=round(p, 3), p_down=round(1 - p, 3), signal=sig, target=round(tp, 3), low=round(lo, 3), high=round(hi, 3), stats=st)
            parts.append(f"{lab(H)}: ऊपर {p*100:.0f}% | नीचे {(1 - p) * 100:.0f}% → {HSIG[sig]} | कीमत {price:.3f} → ~{tp:.3f} (रेंज {lo:.3f}–{hi:.3f}, {(tp - price) / PIP:+.1f} pip) | "
                         f"सटीकता(100)={st.get('acc100', '-')}% (सादे अनुमान से {st.get('edge_pt', '-')} अंक{' ⚠ बढ़त अप्रमाणित' if st.get('n', 0) >= 200 and st.get('edge_pt', 0) < 1 else ''}) | कीमत-त्रुटि {st.get('mae_pip', '-')} pip (बिना-बदलाव {st.get('naive_pip', '-')}) | रेंज-कवर {st.get('range_cover%', '-')}%")
    if not parts: return
    last = F.iloc[-1]; ne, mins = next_event(); rk = risk_report(last, price, ne, mins)
    log(f"{ts} | {price:.3f} | कैंडल: {candle_info(w)}")
    for x in parts: log("   " + x)
    log(f"   ⚠ नुकसान का जोखिम (अगले 1 घंटे में ~95% मामलों में): LONG में अधिकतम गिरावट ~{rk['long_pips']} pip | SHORT में अधिकतम उछाल ~{rk['short_pips']} pip | सामान्य से {rk['mult']}x{rk['note']}")
    if rk["causes"]: log("   कारण: " + " | ".join(rk["causes"]))
    if rk["losses"]: log("   संभावित नुकसान: " + " | ".join(rk["losses"]))
    bg = big_predict(F) if BIG else {}
    for H, b in bg.items():
        log(f"   बड़ी चाल की चेतावनी ({lab(H)} में ≥{b['down_pips']:.0f} pip गिरावट या ≥{b['up_pips']:.0f} pip उछाल): खतरा {b['level']} | कोई भी {b['p_any']*100:.0f}% (सामान्य {b['base_any']*100:.0f}%) | गिरावट {b['p_down']*100:.0f}% | उछाल {b['p_up']*100:.0f}% | दिशा अनिश्चित")
        if b["level"] == "ऊँचा" and state.get(f"bigl{H}") != "ऊँचा" and now - state.get("t_bignotify", 0) > 600:
            notify(f"बड़ी हलचल का खतरा ऊँचा: अगले {lab(H)} में ≥{b['down_pips']:.0f} pip की चाल की संभावना {b['p_any']*100:.0f}% (सामान्य {b['base_any']*100:.0f}%)। दिशा अनिश्चित।", "USDJPY खतरा"); state["t_bignotify"] = now
        state[f"bigl{H}"] = b["level"]
    nlines, ndata = top_news(F.index[-1])
    for x in nlines: log(x)
    news_alerts(state, F.index[-1], now)
    log(f"   news: usd{last.get('n_usd60', 0):+.0f} jpy{last.get('n_jpy60', 0):+.0f} risk{last.get('n_risk60', 0):+.0f} | अगला event: {ne['title'] + ' ' + str(mins) + ' मिनट में' if ne else '-'}")
    for H, (sig, p, tp, lo, hi) in sigs.items():                                 # phone alert
        if sig != "NO TRADE" and sig != state.get(f"sig{H}") and H >= 5 and now - state.get("t_notify", 0) > 120:
            risky = rk["long_pips"] if sig == "LONG" else rk["short_pips"]
            conf_p = p if sig == "LONG" else 1 - p
            sh = state.get("stats", {}).get(f"H{H}", {}); noedge = " | बढ़त अप्रमाणित" if (sh.get("n", 0) >= 200 and sh.get("edge_pt", 0) < 1) else ""
            notify(f"USDJPY {price:.3f} | {lab(H)}: {HSIG[sig]} {conf_p*100:.0f}%{noedge} | लक्ष्य ~{tp:.3f} ({lo:.3f}–{hi:.3f}) | news-जोखिम ~{risky} pip ({rk['mult']}x)", "USDJPY signal"); state["t_notify"] = now
        state[f"sig{H}"] = sig
    if ne and mins is not None and mins <= 15 and ne["impact"] == "High" and (ne["ts"], ne["title"]) != tuple(state.get("ev_notified", [])):
        notify(f"{ne['country']} {ne['title']} {mins} मिनट में। अचानक उतार-चढ़ाव का जोखिम: LONG ~{rk['long_pips']} pip, SHORT ~{rk['short_pips']} pip।", "High-impact news"); state["ev_notified"] = [ne["ts"], ne["title"]]
    day = str(dt.datetime.utcnow().date())
    if state.get("day") not in (None, day): notify(" | ".join(f"{lab(H)} {json.dumps(stats(preds, H), ensure_ascii=False)}" for H in HORIZONS), "USDJPY daily accuracy")
    state["day"] = day; state["stats"] = {f"H{H}": stats(preds, H) for H in HORIZONS}; state["features"] = len(CH[HORIZONS[0]]["cols"]) if HORIZONS[0] in CH else 0
    jsave("latest.json", dict(ts=ts, price=round(price, 3), candle=candle_info(w), predictions=rec, risk=rk, big_move=bg, news=ndata, next_event=(ne, mins)))


def main():
    global HIST, NEWS_DF
    if len(sys.argv) > 2 and sys.argv[1] == "import":
        import_histdata(sys.argv[2:]); return
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        backfill(int(sys.argv[2]) if len(sys.argv) > 2 else 30); return
    state = jload("state.json", {}); preds = jload("preds.json", []); HIST = load_hist(); n = 0; t0 = time.time()
    if BACKFILL_DAYS and state.get("backfilled", 0) < BACKFILL_DAYS:
        backfill(BACKFILL_DAYS); state["backfilled"] = BACKFILL_DAYS; HIST = load_hist()
    for H in HORIZONS:
        try: CH[H] = joblib.load(P(f"champion_{H}.joblib"))
        except Exception: pass
    CAL.extend(jload("calendar.json", [])); RISK.update(jload("risk.json", {})); NEWS_IMPACT.update(jload("news_impact.json", {}))
    if os.path.exists(P("news_log.csv")):
        try: NEWS_DF = load_news()
        except Exception as e: log(f"news_log padhne me error {e!r}")
    for H in BIG_H:
        try: BIG[H] = joblib.load(P(f"bigmove_{H}.joblib"))
        except Exception: pass
    log(f"USDJPY 24x7 bot शुरू | horizons={HORIZONS} | purane bars={len(HIST)} | models={list(CH)}")
    while True:
        try: cycle(state, preds)
        except Exception as e:
            log(f"cycle error (bot chalta rahega): {e!r}"); log(traceback.format_exc()[-600:]); time.sleep(20 if not SYNTH else 0)
        preds[:] = preds[-30000:]
        try: jsave("preds.json", preds); jsave("state.json", state)
        except Exception as e: log(f"save error {e!r}")
        n += 1
        if MAX_CYCLES and n >= MAX_CYCLES: break
        if MAX_RUN and time.time() - t0 > MAX_RUN * 60: break
        sl = E("SLEEP"); time.sleep(float(sl) if sl is not None else 62 - time.time() % 60)
    for H, m in CH.items(): joblib.dump(m, P(f"champion_{H}.joblib"))
    log("run poora hua, models/data save ho gaye. Agla run yahin se aage chalega.")


if __name__ == "__main__":
    main()
