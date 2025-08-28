#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Ingest & Merge (prezzi + news) — ONLINE-FIRST per la Merged Trading Pipeline Full Fixed

Obiettivo: prendere più informazioni possibili dai siti online, unirle e produrre un dataset
massimo e coerente per il training (inclusi input opzionali) per massimizzare le performance.

- Online-first: usa più fonti web; fallback sintetico solo se richiesto.
- Prezzi multi-sorgente: yfinance (primario) + ccxt (Binance, Kraken, Coinbase, Bitfinex) + Stooq/Investpy.
- Derivatives (Binance Futures): funding rate & open interest per BTC/USDT e ETH/USDT.
- Macro FRED (se FRED_API_KEY): DGS10, FEDFUNDS, CPIAUCSL, UNRATE, M2SL.
- Google Trends (pytrends): 'bitcoin', 'ethereum'.
- News NLP: RSS + CryptoPanic → sentiment giornaliero e volume news.
- Indice globale massimo + tutte le colonne stessa lunghezza (reindex + ffill + bfill).
- Feature derivate base: ritorni/log-ritorni, volatilità rolling, range normalizzati.
- Manifest/Report con coperture e correlazioni top± vs BTC.
"""
from __future__ import annotations
import argparse, os, json, time, re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# --- opzionali, gestiti con try/except ---
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None
try:
    from pandas_datareader import data as pdr  # type: ignore
except Exception:
    pdr = None
try:
    import investpy  # type: ignore
except Exception:
    investpy = None
try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
except Exception:
    SentimentIntensityAnalyzer = None
try:
    import requests
except Exception:
    requests = None
try:
    from pytrends.request import TrendReq  # type: ignore
except Exception:
    TrendReq = None

DEFAULT_BASE = 'BTC-USD'
DEFAULT_CRYPTO = ['ETH-USD','BNB-USD','SOL-USD','XRP-USD','ADA-USD','LTC-USD','DOGE-USD','MATIC-USD','AVAX-USD','BCH-USD']
DEFAULT_EQUITY = ['IBIT','FBTC','BITB','ARKB','COIN','MSTR','RIOT','MARA','GBTC']
DEFAULT_FOREX = ['EURUSD=X','GBPUSD=X','USDJPY=X','USDCAD=X','AUDUSD=X']
DEFAULT_COMMODITIES = ['GC=F','CL=F','XAUUSD=X','XAGUSD=X','DX=F']
DEFAULT_INDEXES = ['^GSPC','^NDX','^VIX']
DEFAULT_RSS = [
    'https://finance.yahoo.com/rss/topstories',
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://cointelegraph.com/rss',
    'https://www.investing.com/rss/news_25.rss',
    'https://www.investing.com/rss/commodities.rss',
]
BASE_ALIASES: Dict[str, List[str]] = {'BTC-USD': ['XBT-USD', 'BTCUSD=X', 'BTC-EUR']}
FRED_SERIES = ['DGS10','FEDFUNDS','CPIAUCSL','UNRATE','M2SL']

def slugify(sym: str) -> str:
    return sym.replace('^','SPX').replace('=','').replace('-','_').replace('/', '_')

def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df

def to_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    rename = {'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'}
    df = df.rename(columns=rename)
    for c in ['open','high','low','close','volume']:
        if c not in df.columns:
            df[c] = np.nan
    return df[['open','high','low','close','volume']]

def combine_prefer_left(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left = to_ohlcv_df(ensure_dt_index(left)); right = to_ohlcv_df(ensure_dt_index(right))
    idx = left.index.union(right.index).unique()
    out = left.reindex(idx).combine_first(right.reindex(idx))
    return ensure_dt_index(out)

def _sleep_backoff(t: float) -> None:
    time.sleep(min(5.0, t))

def _req_json(url: str, params: Optional[dict]=None, tries: int=3) -> Optional[dict]:
    if requests is None:
        return None
    back = 0.5
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        _sleep_backoff(back); back *= 2
    return None

# --------- download prezzi ----------
def fetch_yf(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(symbol, start=start, end=end, interval='1d', auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return to_ohlcv_df(ensure_dt_index(df))
    except Exception:
        return pd.DataFrame()

CCXT_MAP = {
    'BTC-USD': ['BTC/USDT','BTC/USD'],
    'ETH-USD': ['ETH/USDT','ETH/USD'],
    'BNB-USD': ['BNB/USDT'],
    'SOL-USD': ['SOL/USDT','SOL/USD'],
    'XRP-USD': ['XRP/USDT','XRP/USD'],
    'ADA-USD': ['ADA/USDT','ADA/USD'],
    'LTC-USD': ['LTC/USDT','LTC/USD'],
    'DOGE-USD': ['DOGE/USDT','DOGE/USD'],
    'MATIC-USD': ['MATIC/USDT','MATIC/USD'],
    'AVAX-USD': ['AVAX/USDT','AVAX/USD'],
    'BCH-USD': ['BCH/USDT','BCH/USD'],
}
CCXT_EXCHANGES = ['binance','kraken','coinbase','bitfinex']

def fetch_ccxt_multi(symbol_yf: str, start: str, end: Optional[str]) -> pd.DataFrame:
    if ccxt is None: return pd.DataFrame()
    pairs = CCXT_MAP.get(symbol_yf) or []
    if not pairs: return pd.DataFrame()
    dfs: List[pd.DataFrame] = []
    since_ms = int(pd.Timestamp(start).timestamp()*1000)
    end_ms = int(pd.Timestamp(end).timestamp()*1000) if end else None
    for ex_name in CCXT_EXCHANGES:
        try:
            ex = getattr(ccxt, ex_name)()
            ex.load_markets()
            for pair in pairs:
                if pair not in ex.markets: continue
                cursor = since_ms; local=[]
                while True:
                    batch = ex.fetch_ohlcv(pair, timeframe='1d', since=cursor, limit=1000)
                    if not batch: break
                    cursor = batch[-1][0] + 24*3600*1000
                    local += batch
                    if end_ms and cursor >= end_ms: pass
                    if len(batch) < 1000: break
                    _sleep_backoff(0.2)
                if local:
                    arr = np.array(local)
                    df = pd.DataFrame({'open':arr[:,1],'high':arr[:,2],'low':arr[:,3],'close':arr[:,4],'volume':arr[:,5]},
                                      index=pd.to_datetime(arr[:,0], unit='ms'))
                    dfs.append(to_ohlcv_df(ensure_dt_index(df)))
        except Exception:
            continue
    if not dfs: return pd.DataFrame()
    out = dfs[0]
    for extra in dfs[1:]:
        out = combine_prefer_left(out, extra)
    return ensure_dt_index(out)

def fetch_stooq(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    if pdr is None: return pd.DataFrame()
    try:
        st = pd.Timestamp(start); en = pd.Timestamp(end) if end else pd.Timestamp.today()
        df = pdr.DataReader(symbol, 'stooq', st, en).sort_index()
        return to_ohlcv_df(ensure_dt_index(df))
    except Exception:
        return pd.DataFrame()

def fetch_investpy_equity(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    if investpy is None: return pd.DataFrame()
    try:
        st = pd.Timestamp(start); en = pd.Timestamp(end) if end else pd.Timestamp.today()
        df = investpy.get_stock_historical_data(stock=symbol, country='united states',
                                                from_date=st.strftime('%d/%m/%Y'),
                                                to_date=en.strftime('%d/%m/%Y'))
        df.index.name = None
        return to_ohlcv_df(ensure_dt_index(df))
    except Exception:
        return pd.DataFrame()

def best_ohlcv(symbol: str, start: str, end: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats: Dict[str, int] = {}
    dfs: List[pd.DataFrame] = []
    yf_df = fetch_yf(symbol, start, end); stats['yfinance'] = int(len(yf_df))
    if not yf_df.empty: dfs.append(yf_df)
    ccxt_df = fetch_ccxt_multi(symbol, start, end); stats['ccxt_multi'] = int(len(ccxt_df))
    if not ccxt_df.empty: dfs[0] = combine_prefer_left(dfs[0], ccxt_df) if dfs else dfs.append(ccxt_df) or ccxt_df
    stq_df = fetch_stooq(symbol, start, end); stats['stooq'] = int(len(stq_df))
    if not stq_df.empty: dfs[0] = combine_prefer_left(dfs[0], stq_df) if dfs else dfs.append(stq_df) or stq_df
    inv_df = fetch_investpy_equity(symbol, start, end); stats['investpy'] = int(len(inv_df))
    if not inv_df.empty: dfs[0] = combine_prefer_left(dfs[0], inv_df) if dfs else dfs.append(inv_df) or inv_df
    if not dfs: return pd.DataFrame(), stats
    return ensure_dt_index(dfs[0]), stats

# --------- Derivatives (Binance Futures) ----------
def fetch_binance_funding(symbol_ccxt: str, start: str, end: Optional[str]) -> pd.Series:
    if requests is None: return pd.Series(dtype=float)
    base = 'https://fapi.binance.com/fapi/v1/fundingRate'
    sym = symbol_ccxt.replace('/','')
    start_ms = int(pd.Timestamp(start).timestamp()*1000)
    end_ms = int(pd.Timestamp(end).timestamp()*1000) if end else None
    out = []; cursor = start_ms
    while True:
        params = {'symbol': sym, 'limit': 1000, 'startTime': cursor}
        if end_ms: params['endTime'] = end_ms
        data = _req_json(base, params)
        if not data: break
        out += data
        if len(data) < 1000: break
        cursor = int(data[-1]['fundingTime']) + 1
        _sleep_backoff(0.25)
    if not out: return pd.Series(dtype=float)
    df = pd.DataFrame(out)
    s = pd.Series(pd.to_numeric(df['fundingRate'], errors='coerce').astype(float).values,
                  index=pd.to_datetime(df['fundingTime'], unit='ms')).resample('1D').mean()
    s.name = f'funding_rate_{sym}'
    return s

def fetch_binance_oi(symbol_ccxt: str, start: str, end: Optional[str]) -> pd.Series:
    if requests is None: return pd.Series(dtype=float)
    base = 'https://fapi.binance.com/futures/data/openInterestHist'
    sym = symbol_ccxt.replace('/','')
    start_ms = int(pd.Timestamp(start).timestamp()*1000)
    end_ms = int(pd.Timestamp(end).timestamp()*1000) if end else None
    out = []; cursor = start_ms
    while True:
        params = {'symbol': sym, 'period': '1d', 'limit': 500, 'startTime': cursor}
        if end_ms: params['endTime'] = end_ms
        data = _req_json(base, params)
        if not data: break
        out += data
        if len(data) < 500: break
        cursor = int(pd.to_datetime(data[-1]['timestamp']).timestamp()*1000) + 1
        _sleep_backoff(0.25)
    if not out: return pd.Series(dtype=float)
    df = pd.DataFrame(out)
    s = pd.Series(pd.to_numeric(df['sumOpenInterestValue'], errors='coerce').astype(float).values,
                  index=pd.to_datetime(df['timestamp'])).resample('1D').mean()
    s.name = f'open_interest_notional_{sym}'
    return s

# --------- Macro (FRED) ----------
def fetch_fred_series(series_id: str, start: str, end: Optional[str]) -> pd.Series:
    api_key = os.environ.get('FRED_API_KEY')
    if requests is None or not api_key: return pd.Series(dtype=float)
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {'series_id': series_id, 'api_key': api_key, 'file_type': 'json', 'observation_start': start}
    if end: params['observation_end'] = end
    js = _req_json(url, params)
    if not js or 'observations' not in js: return pd.Series(dtype=float)
    obs = js['observations']
    idx = pd.to_datetime([o['date'] for o in obs])
    vals = pd.to_numeric([o['value'] for o in obs], errors='coerce')
    s = pd.Series(vals, index=idx).resample('1D').ffill()
    s.name = f'fred_{series_id}'
    return s

# --------- Google Trends ----------
def fetch_trends_daily(keyword: str, start: str, end: Optional[str]) -> pd.Series:
    if TrendReq is None: return pd.Series(dtype=float)
    try:
        tr = TrendReq(hl='en-US', tz=0)
        timefr = f"{start} {end or datetime.utcnow().strftime('%Y-%m-%d')}"
        tr.build_payload([keyword], timeframe=timefr, geo='')
        df = tr.interest_over_time()
        if df is None or df.empty: return pd.Series(dtype=float)
        s = df[keyword].resample('1D').interpolate().ffill()
        s.name = f'gtrend_{keyword.lower()}'
        return s
    except Exception:
        return pd.Series(dtype=float)

# --------- News & Sentiment ----------
def simple_sent(text: str) -> float:
    txt = (text or '').lower()
    pos = len(re.findall(r" (up|surge|bull|gain|beat|rally|approve|etf|record|all-time high) ", txt))
    neg = len(re.findall(r" (down|drop|bear|loss|miss|ban|hack|lawsuit|reject|selloff|crash) ", txt))
    return float((pos - neg) / max(1, pos + neg))

def fetch_rss(url: str) -> List[dict]:
    items: List[dict] = []
    if feedparser is None: return items
    try:
        feed = feedparser.parse(url)
        for ent in getattr(feed, 'entries', []) or []:
            title = (getattr(ent, 'title', '') or '').strip()
            summary = (getattr(ent, 'summary', '') or '').strip()
            link = (getattr(ent, 'link', '') or '').strip()
            ts = getattr(ent, 'published', None) or getattr(ent, 'updated', None) or None
            try:
                timestamp = pd.to_datetime(ts)
            except Exception:
                timestamp = pd.Timestamp.utcnow()
            # FIX: rimpiazza newline correttamente
            text = f"{title} {summary}".replace('\n',' ').replace('\r',' ').strip()
            items.append({'timestamp': timestamp, 'title': title, 'text': text, 'source': url, 'url': link})
    except Exception:
        return items
    return items

def fetch_cryptopanic(api_token: Optional[str], max_pages: int = 2) -> List[dict]:
    if not api_token or requests is None: return []
    base = 'https://cryptopanic.com/api/v1/posts/'
    params = {'auth': api_token, 'currencies': 'BTC,ETH', 'regions': 'en', 'kind': 'news', 'public': 'true'}
    items: List[dict] = []; url = base
    for _ in range(max_pages):
        data = _req_json(url, params)
        if not data: break
        for p in data.get('results', []):
            ts = pd.to_datetime(p.get('published_at', pd.Timestamp.utcnow()))
            title = p.get('title','')
            items.append({'timestamp': ts, 'title': title, 'text': title, 'source': 'cryptopanic', 'url': p.get('url','')})
        url = data.get('next'); params = None
        if not url: break
        _sleep_backoff(0.25)
    return items

def collect_news(rss_list: List[str], cryptopanic_token: Optional[str]) -> pd.DataFrame:
    all_items: List[dict] = []
    for u in rss_list: all_items += fetch_rss(u)
    all_items += fetch_cryptopanic(cryptopanic_token)
    if not all_items:
        return pd.DataFrame(columns=['timestamp','title','text','source','url','sentiment'])
    df = pd.DataFrame(all_items)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    for c in ['title','text','source','url']:
        if c not in df.columns: df[c] = ''
        df[c] = df[c].astype(str).fillna('')
    df['hash'] = (df['title'].str.strip() + '|' + df['url'].str.strip()).apply(hash)
    df = df.drop_duplicates(subset=['hash']).drop(columns=['hash'])
    patt = re.compile(r" (bitcoin|btc|ethereum|eth|crypto|etf|coinbase|blockchain|sec) ", re.I)
    df = df[df['text'].apply(lambda s: bool(patt.search(str(s))))]
    if df.empty:
        return pd.DataFrame(columns=['timestamp','title','text','source','url','sentiment'])
    if SentimentIntensityAnalyzer is not None:
        an = SentimentIntensityAnalyzer()
        df['sentiment'] = df['text'].apply(lambda t: an.polarity_scores(str(t))['compound'])
    else:
        df['sentiment'] = df['text'].apply(simple_sent)
    return df.sort_values('timestamp').reset_index(drop=True)

def daily_news_features(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df is None or news_df.empty or 'timestamp' not in news_df.columns:
        return pd.DataFrame(index=pd.DatetimeIndex([], name='date'))
    tmp = news_df[['timestamp','sentiment']].copy()
    tmp['date'] = pd.to_datetime(tmp['timestamp']).dt.floor('D')
    g = tmp.groupby('date')
    df = pd.DataFrame({
        'news_sentiment_daily': g['sentiment'].mean(),
        'news_count_daily': g['sentiment'].size()
    })
    df['news_sentiment_7d'] = df['news_sentiment_daily'].rolling(7, min_periods=1).mean()
    return df

# --------- Feature derivate base ----------
def add_derived_features(merged: pd.DataFrame) -> pd.DataFrame:
    need = {'open','high','low','close','volume'}
    if not need.issubset(merged.columns): return merged
    out = merged.copy()
    out['feat_ret_1d'] = out['close'].pct_change()
    out['feat_logret_1d'] = np.log(out['close']).diff()
    out['feat_vol_7d'] = out['feat_logret_1d'].rolling(7).std()
    out['feat_vol_30d'] = out['feat_logret_1d'].rolling(30).std()
    out['feat_hl_range'] = (out['high'] - out['low']) / out['close']
    out['feat_oc_range'] = (out['open'] - out['close']) / out['close']
    return out.ffill().bfill()

# --------- Merge ----------
def build_global_index(series_map: Dict[str, pd.DataFrame], end: Optional[str]) -> pd.DatetimeIndex:
    mins = [df.index.min() for df in series_map.values() if not df.empty]
    if not mins: raise RuntimeError('Nessuna serie valida scaricata')
    global_start = min(mins)
    global_end = pd.to_datetime(end) if end else pd.Timestamp.today().normalize()
    return pd.date_range(global_start, global_end, freq='D')

def align_and_extend(series_map: Dict[str, pd.DataFrame], base: str, end: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    idx = build_global_index(series_map, end)
    if base not in series_map or series_map[base].empty:
        raise RuntimeError(f'Serie base {base} mancante')
    btc = series_map[base].reindex(idx)

    stats: Dict[str, dict] = {}
    def fill_and_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        before = int(df.notna().all(axis=1).sum())
        df_ff = df.ffill().bfill()
        after = int(df_ff.notna().all(axis=1).sum())
        return df_ff, {'rows_original_complete': before, 'rows_after_fill_complete': after,
                       'coverage_ratio': float(after/len(df_ff)) if len(df_ff) > 0 else 0.0}

    btc_filled, st_btc = fill_and_stats(btc)
    stats[base] = st_btc | {'start': str(idx.min()), 'end': str(idx.max()), 'rows': int(len(idx))}
    merged = btc_filled[['open','high','low','close','volume']].copy()

    for sym, df in series_map.items():
        if sym == base or df.empty: continue
        df2 = df.reindex(idx)
        cols = ['open','high','low','close','volume']
        df2_filled, st = fill_and_stats(df2[cols])
        stats[sym] = st | {'start': str(idx.min()), 'end': str(idx.max()), 'rows': int(len(idx))}
        s = slugify(sym)
        for col in cols:
            merged[f'{col}_{s}'] = df2_filled[col]

    merged.index.name = 'date'
    return merged, stats

# --------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=str, default='./raw_data')
    ap.add_argument('--start', type=str, default='2010-01-01')
    ap.add_argument('--end', type=str, default=None)
    ap.add_argument('--base', type=str, default=DEFAULT_BASE)
    ap.add_argument('--crypto', type=str, default=','.join(DEFAULT_CRYPTO))
    ap.add_argument('--equity', type=str, default=','.join(DEFAULT_EQUITY))
    ap.add_argument('--forex', type=str, default=','.join(DEFAULT_FOREX))
    ap.add_argument('--commodities', type=str, default=','.join(DEFAULT_COMMODITIES))
    ap.add_argument('--indexes', type=str, default=','.join(DEFAULT_INDEXES))
    ap.add_argument('--rss', type=str, default=','.join(DEFAULT_RSS))
    ap.add_argument('--cryptopanic-token', type=str, default=os.environ.get('CRYPTOPANIC_TOKEN'))
    ap.add_argument('--no-news', action='store_true')
    ap.add_argument('--write-pipeline-config', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--strict-base', action='store_true')
    ap.add_argument('--offline-demo', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        # smoke mini test su stringhe & allineamento base
        title, summary = 'Hello', 'World\nNew Line\rCarriage'
        text = f"{title} {summary}".replace('\n',' ').replace('\r',' ').strip()
        assert '\n' not in text and '\r' not in text
        idx_a = pd.date_range('2020-01-01', periods=10, freq='D')
        a = pd.DataFrame({'open':1,'high':1,'low':1,'close':1,'volume':1}, index=idx_a)
        merged, _ = align_and_extend({'BTC-USD': a}, 'BTC-USD', None)
        assert len(merged) == len(idx_a)
        print('[SELF-TEST] OK'); return

    out_dir = args.out_dir
    prices_dir = os.path.join(out_dir, 'prices', 'INDIVIDUAL'); os.makedirs(prices_dir, exist_ok=True)
    news_dir = os.path.join(out_dir, 'news'); os.makedirs(news_dir, exist_ok=True)

    # universo
    universe: List[str] = [args.base]
    for grp in [args.crypto, args.equity, args.forex, args.commodities, args.indexes]:
        if grp: universe += [s.strip() for s in grp.split(',') if s.strip()]
    seen = set(); universe = [x for x in universe if not (x in seen or seen.add(x))]

    # prezzi
    series_map: Dict[str, pd.DataFrame] = {}
    src_stats: Dict[str, Dict[str, int]] = {}
    print('[1/7] Scarico prezzi (online, multi-sorgente)...')
    for sym in universe:
        df_best, stats = best_ohlcv(sym, args.start, args.end)
        if not df_best.empty:
            series_map[sym] = df_best; src_stats[sym] = stats
            try: df_best.to_parquet(os.path.join(prices_dir, f'{sym}.parquet'))
            except Exception: pass
        else:
            print(f'[WARN] Nessun dato per {sym} (tutte le fonti)')

    base_used = args.base; alias_used = None
    if (args.base not in series_map or series_map[args.base].empty):
        if not args.strict_base:
            for alias in BASE_ALIASES.get(args.base, []):
                df_a, stats_a = best_ohlcv(alias, args.start, args.end)
                if not df_a.empty:
                    series_map[args.base] = df_a; src_stats[args.base] = stats_a; alias_used = alias; break
        if (args.base not in series_map or series_map[args.base].empty):
            if args.offline_demo:
                print(f"[INFO] Base '{args.base}' mancante: genero serie sintetica (offline demo).")
                idx = pd.date_range(args.start, args.end or datetime.utcnow().strftime('%Y-%m-%d'), freq='D')
                rng = np.random.RandomState(42)
                r = rng.normal(0,0.01,len(idx)); close = 20000*np.exp(np.cumsum(r))
                high = close*(1+np.abs(rng.normal(0,0.002,len(idx))))
                low  = close*(1-np.abs(rng.normal(0,0.002,len(idx))))
                open_ = close*(1+rng.normal(0,0.001,len(idx)))
                vol = (np.abs(r)*1e6).astype(int)
                series_map[args.base] = pd.DataFrame({'open':open_, 'high':high, 'low':low, 'close':close, 'volume':vol}, index=idx)
                src_stats[args.base] = {'synthetic': len(idx)}
            elif args.strict_base:
                raise SystemExit("Base %s non disponibile: impossibile proseguire (usa fonti online, rimuovi --strict-base o abilita --offline-demo)" % args.base)
            else:
                for cand in [s for s in series_map.keys() if s.endswith('-USD') or s.endswith('=X')] + list(series_map.keys()):
                    if not series_map[cand].empty:
                        series_map[args.base] = series_map[cand]
                        src_stats[args.base] = src_stats.get(cand, {}) | {'_substituted_from': cand}
                        base_used = args.base
                        break

    base_used = args.base

    print('[2/7] Derivatives (funding & open interest)...')
    fr_btc = fetch_binance_funding('BTC/USDT', args.start, args.end)
    oi_btc = fetch_binance_oi('BTC/USDT', args.start, args.end)
    fr_eth = fetch_binance_funding('ETH/USDT', args.start, args.end)
    oi_eth = fetch_binance_oi('ETH/USDT', args.start, args.end)

    print('[3/7] Macro FRED (se API key presente)...')
    fred = {}
    for sid in FRED_SERIES:
        s = fetch_fred_series(sid, args.start, args.end)
        if not s.empty:
            fred[s.name] = s

    print('[4/7] Google Trends (se disponibile)...')
    gtrend_btc = fetch_trends_daily('bitcoin', args.start, args.end)
    gtrend_eth = fetch_trends_daily('ethereum', args.start, args.end)

    print('[5/7] Allineo e unisco...')
    merged, align_stats = align_and_extend(series_map, base_used, args.end)

    # join extra serie
    for s in [fr_btc, oi_btc, fr_eth, oi_eth]:
        if s is not None and not getattr(s, 'empty', True):
            merged = merged.join(s, how='left')
    for name, s in fred.items():
        merged = merged.join(s, how='left')
    for s in [gtrend_btc, gtrend_eth]:
        if s is not None and not getattr(s, 'empty', True):
            merged = merged.join(s, how='left')

    # [6/7] News ingest
    news_csv_path = None; news_parquet_path = None
    if not args.no_news:
        print('[6/7] News & sentiment...')
        rss_list = [s.strip() for s in (args.rss or '').split(',') if s.strip()]
        news_df = collect_news(rss_list, args.cryptopanic_token)
        if not news_df.empty:
            news_parquet_path = os.path.join(news_dir, 'news.parquet')
            news_csv_path = os.path.join(news_dir, 'news.csv')
            keep = ['timestamp','text','title','source','url','sentiment']
            for c in keep:
                if c not in news_df.columns:
                    news_df[c] = '' if c != 'timestamp' else pd.NaT
            news_df = news_df[keep]
            try: news_df.to_parquet(news_parquet_path, index=False)
            except Exception: pass
            news_df.to_csv(news_csv_path, index=False)
            news_daily_feat = daily_news_features(news_df)
            if not news_daily_feat.empty:
                merged = merged.join(news_daily_feat, how='left')
        else:
            print('[WARN] Nessuna news raccolta (continua solo prezzi)')
    else:
        print('[INFO] Ingest news disabilitato (--no-news)')

    # feature derivate e fill finale
    merged = add_derived_features(merged).ffill().bfill()

    # check colonne base
    for c in ['open','high','low','close','volume']:
        if c not in merged.columns:
            raise SystemExit(f'Colonna richiesta mancante: {c}')

    # suggerimento seq_cols
    suggested_seq_cols = (
        ['open','high','low','close','volume'] +
        [c for c in merged.columns if c.startswith(('open_','high_','low_','close_','volume_'))] +
        [c for c in merged.columns if c.startswith('funding_rate_') or c.startswith('open_interest_notional_')] +
        [c for c in merged.columns if c.startswith('fred_') or c.startswith('gtrend_')] +
        [c for c in ['news_sentiment_daily','news_sentiment_7d','news_count_daily'] if c in merged.columns] +
        [c for c in merged.columns if c.startswith('feat_')]
    )

    # salvataggi
    os.makedirs(out_dir, exist_ok=True)
    try: merged.to_parquet(os.path.join(out_dir, 'prices_merged.parquet'))
    except Exception: pass
    merged.to_csv(os.path.join(out_dir, 'prices_merged.csv'))

    # correlazioni vs base
    corrs = {}
    try:
        base_close = merged['close']
        for c in merged.columns:
            if c.startswith('close_') and merged[c].notna().sum() > 10:
                corrs[c] = float(pd.concat([base_close, merged[c]], axis=1).dropna().corr().iloc[0,1])
    except Exception:
        pass
    top_pos = sorted([(k,v) for k,v in corrs.items() if not np.isnan(v)], key=lambda x: x[1], reverse=True)[:10]
    top_neg = sorted([(k,v) for k,v in corrs.items() if not np.isnan(v)], key=lambda x: x[1])[:10]

    manifest = {
        'created_at': datetime.utcnow().isoformat(timespec='seconds')+'Z',
        'feature_inclusion_policy': 'max_all_available',
        'base_asset_requested': args.base,
        'base_asset_used': base_used,
        'base_alias_used': alias_used,
        'universe': universe,
        'out_paths': {
            'prices_merged_csv': os.path.join(out_dir, 'prices_merged.csv'),
            'prices_merged_parquet': os.path.join(out_dir, 'prices_merged.parquet'),
            'news_csv': news_csv_path,
            'news_parquet': news_parquet_path,
            'individual_dir': os.path.join(out_dir, 'prices', 'INDIVIDUAL'),
        },
        'sources_rows': src_stats,
        'align_stats': align_stats,
        'suggested_seq_cols': suggested_seq_cols,
        'notes': {
            'all_columns_same_length': True,
            'index_frequency': '1D',
            'fill_method': 'ffill_then_bfill',
            'pipeline_required_columns': ['open','high','low','close','volume']
        }
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    pipeline_cfg = {
        'prices_csv': manifest['out_paths']['prices_merged_csv'],
        'news_csv': manifest['out_paths']['news_csv'],
        'seq_cols': suggested_seq_cols,
        'defaults': {'seq_len': 64, 'horizon': 30}
    }
    with open(os.path.join(out_dir, 'pipeline_config.json'), 'w') as pf:
        json.dump(pipeline_cfg, pf, indent=2)

    rep = {
        'rows_total': int(len(merged)),
        'date_start': str(merged.index.min()),
        'date_end': str(merged.index.max()),
        'columns': list(merged.columns),
        'non_null_counts': {c: int(merged[c].notna().sum()) for c in merged.columns},
        'symbol_coverage': align_stats,
        'corr_top_pos': top_pos,
        'corr_top_neg': top_neg
    }
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump(rep, f, indent=2)

    print('\n[DONE] Dataset creato (online-first):')
    print(' -', manifest['out_paths']['prices_merged_csv'])
    if manifest['out_paths']['news_csv']:
        print(' -', manifest['out_paths']['news_csv'])
    print("Feature totali:", len(merged.columns))
    print('Suggerimento seq_cols -> pipeline_config.json')

if __name__ == '__main__':
    main()
