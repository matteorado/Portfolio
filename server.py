"""
Portfolio Optimizer — Server locale
Prima installazione (una volta sola):
  pip3 install yfinance scipy numpy

Poi avvia con:
  python3 server.py
"""

import http.server
import urllib.parse
import urllib.request
import json
import os
import sys
import math
import numpy as np

PORT = 7532

try:
    import yfinance as yf
except ImportError:
    print("\n  ERRORE: pip3 install yfinance\n")
    sys.exit(1)

try:
    from scipy.optimize import minimize
except ImportError:
    print("\n  ERRORE: pip3 install scipy\n")
    sys.exit(1)
print("  [OK] yfinance + scipy pronti")


# ─── Yahoo Finance ────────────────────────────────────────────────────────────

def search_tickers(query):
    try:
        results = yf.Search(query, max_results=12, news_count=0)
        quotes = []
        for q in (results.quotes or []):
            if q.get('quoteType') in ('CURRENCY', 'OPTION'):
                continue
            quotes.append({
                'symbol':    q.get('symbol', ''),
                'longname':  q.get('longname') or q.get('shortname', ''),
                'shortname': q.get('shortname', ''),
                'exchDisp':  q.get('exchDisp') or q.get('exchange', ''),
                'quoteType': q.get('quoteType', ''),
            })
        return {'quotes': quotes}
    except Exception as e:
        return {'error': str(e), 'quotes': []}


def get_quote(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period='5d')
        closes = hist['Close'].dropna()  # scarta bar incompleti (es. giornata odierna/weekend)
        if closes.empty:
            return {'error': f'Nessun dato per {symbol}'}
        price = float(closes.iloc[-1])
        prev  = float(closes.iloc[-2]) if len(closes) > 1 else price
        chg   = (price - prev) / prev * 100 if prev else 0.0
        info  = t.fast_info
        return {
            'quoteSummary': {'result': [{'price': {
                'symbol':                     symbol,
                'longName':                   getattr(info, 'longName', symbol),
                'shortName':                  symbol,
                'currency':                   getattr(info, 'currency', 'USD'),
                'regularMarketPrice':         {'raw': price},
                'regularMarketChangePercent': {'raw': chg / 100},
                'regularMarketPreviousClose': {'raw': prev},
            }}]}
        }
    except Exception as e:
        return {'error': str(e)}


def get_chart(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period='1y')
        if hist.empty:
            return {'error': f'Nessun dato storico per {symbol}'}
        s      = hist['Close'].dropna()
        closes = [float(x) for x in s.tolist()]
        # Date delle stesse sedute, normalizzate a 'YYYY-MM-DD' e senza fuso orario.
        # Servono al server per allineare i titoli per data (non per posizione)
        # quando si mettono insieme borse con calendari diversi.
        dates  = [d.strftime('%Y-%m-%d') for d in s.index]
        info   = t.fast_info
        price  = closes[-1]
        return {
            'chart': {'result': [{'meta': {
                'symbol':             symbol,
                'currency':           getattr(info, 'currency', 'USD'),
                'exchangeName':       getattr(info, 'exchange', ''),
                'regularMarketPrice': price,
                'previousClose':      closes[-2] if len(closes) > 1 else price,
            }, 'dates': dates, 'indicators': {'quote': [{'close': closes}]}}]}
        }
    except Exception as e:
        return {'error': str(e)}


# ─── Markowitz esatto (scipy) ─────────────────────────────────────────────────

def calc_daily_returns(closes):
    c = np.array(closes, dtype=float)
    c = c[np.isfinite(c)]  # scarta NaN/None (es. bar odierna incompleta)
    return (c[1:] - c[:-1]) / c[:-1]

def align_returns(symbols_data):
    """
    Rendimenti giornalieri allineati fra titoli. Ritorna (matrice (n, T), modo).

    Se ogni titolo porta con sé le proprie date, l'allineamento avviene PER DATA:
    si tengono solo le sedute in cui tutti i titoli hanno quotato e i rendimenti
    si calcolano sui prezzi già allineati. È l'unico modo corretto quando si
    mescolano borse con calendari diversi (le festività nazionali sono sfalsate:
    il 4 luglio a New York, il 2 giugno a Milano).

    Se le date mancano — frontend vecchio, dati mock, download parziale — si
    ripiega sull'allineamento PER POSIZIONE (taglio alla serie più corta), che è
    corretto solo se tutti i titoli quotano sulla stessa borsa.
    """
    have_dates = bool(symbols_data) and all(
        d.get('dates') and len(d['dates']) == len(d.get('closes') or [])
        for d in symbols_data
    )

    if have_dates:
        import pandas as pd
        series = {}
        for i, d in enumerate(symbols_data):
            # chiave = indice, non ticker: due voci sullo stesso titolo non si sovrascrivono
            s = pd.Series([float(x) for x in d['closes']],
                          index=pd.to_datetime(d['dates']))
            s = s[~s.index.duplicated(keep='last')]
            series[i] = s
        df = pd.DataFrame(series).dropna().sort_index()
        if len(df) >= 30:
            P = df.values                       # (T, n) prezzi sulle sedute comuni
            rets = (P[1:] - P[:-1]) / P[:-1]    # (T-1, n)
            return np.asarray(rets.T, dtype=float), 'date'
        # troppe poche sedute in comune: meglio il fallback che statistiche su 5 punti

    daily = [calc_daily_returns(d['closes']) for d in symbols_data]
    min_len = min(len(r) for r in daily)
    return np.array([r[-min_len:] for r in daily]), 'position'


def portfolio_stats(w, mu, cov):
    r = float(w @ mu)
    v = float(np.sqrt(w @ cov @ w))
    return r, v

def optimize_markowitz(symbols_data, rf):
    """
    symbols_data: lista di dict con 'symbol' e 'closes'
    rf: tasso risk-free annualizzato (es. 0.0365)
    Ritorna frontiera efficiente + portafoglio tangenza long-only e con short
    """
    n = len(symbols_data)

    # Allineamento per data se il frontend le manda, per posizione altrimenti
    aligned, align_mode = align_returns(symbols_data)   # (n, T)

    # Rendimenti e covarianza annualizzati
    mu  = aligned.mean(axis=1) * 252
    cov = np.cov(aligned) * 252

    w0 = np.ones(n) / n

    def neg_sharpe(w):
        r, v = portfolio_stats(w, mu, cov)
        return -(r - rf) / v if v > 1e-8 else 0

    def port_variance(w):
        return float(w @ cov @ w)

    eq_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # ── Portafoglio di Tangenza LONG ONLY ──
    bounds_long = [(0.0, 1.0)] * n
    res_long = minimize(neg_sharpe, w0, method='SLSQP',
                        bounds=bounds_long, constraints=[eq_constraint],
                        options={'ftol': 1e-12, 'maxiter': 1000})
    w_tangency_long = res_long.x
    r_tl, v_tl = portfolio_stats(w_tangency_long, mu, cov)

    # ── Portafoglio di Tangenza CON SHORT SELLING ──
    bounds_short = [(-1.0, 2.0)] * n
    res_short = minimize(neg_sharpe, w0, method='SLSQP',
                         bounds=bounds_short, constraints=[eq_constraint],
                         options={'ftol': 1e-12, 'maxiter': 1000})
    w_tangency_short = res_short.x
    r_ts, v_ts = portfolio_stats(w_tangency_short, mu, cov)

    # ── Minima Varianza (long only) ──
    res_minvar = minimize(port_variance, w0, method='SLSQP',
                          bounds=bounds_long, constraints=[eq_constraint],
                          options={'ftol': 1e-12, 'maxiter': 1000})
    w_minvar = res_minvar.x
    r_mv, v_mv = portfolio_stats(w_minvar, mu, cov)

    # ── Frontiera efficiente LONG ONLY ──
    # Varia il rendimento target tra minvar e max singolo titolo
    ret_min = r_mv
    ret_max = float(mu.max())
    frontier = []
    for target in np.linspace(ret_min, ret_max, 60):
        cons = [
            eq_constraint,
            {'type': 'eq', 'fun': lambda w, t=target: float(w @ mu) - t}
        ]
        res = minimize(port_variance, w0, method='SLSQP',
                       bounds=bounds_long, constraints=cons,
                       options={'ftol': 1e-10, 'maxiter': 500})
        if res.success:
            vol = float(np.sqrt(res.fun))
            frontier.append({'ret': round(target * 100, 3), 'vol': round(vol * 100, 3)})

    # ── Frontiera efficiente CON SHORT ──
    frontier_short = []
    for target in np.linspace(ret_min * 0.8, ret_max * 1.2, 60):
        cons = [
            eq_constraint,
            {'type': 'eq', 'fun': lambda w, t=target: float(w @ mu) - t}
        ]
        res = minimize(port_variance, w0, method='SLSQP',
                       bounds=bounds_short, constraints=cons,
                       options={'ftol': 1e-10, 'maxiter': 500})
        if res.success:
            vol = float(np.sqrt(res.fun))
            frontier_short.append({'ret': round(target * 100, 3), 'vol': round(vol * 100, 3)})

    return {
        'symbols': [d['symbol'] for d in symbols_data],
        'asset_rets': [round(float(x) * 100, 3) for x in mu],
        'asset_vols': [round(float(np.sqrt(cov[i,i])) * 100, 3) for i in range(n)],
        'rf': round(rf * 100, 3),
        'align_mode': align_mode,
        'hist_days': int(aligned.shape[1]),
        'tangency_long': {
            'weights': [round(float(x) * 100, 2) for x in w_tangency_long],
            'ret': round(r_tl * 100, 3),
            'vol': round(v_tl * 100, 3),
            'sharpe': round((r_tl - rf) / v_tl, 4) if v_tl > 0 else 0,
        },
        'tangency_short': {
            'weights': [round(float(x) * 100, 2) for x in w_tangency_short],
            'ret': round(r_ts * 100, 3),
            'vol': round(v_ts * 100, 3),
            'sharpe': round((r_ts - rf) / v_ts, 4) if v_ts > 0 else 0,
        },
        'minvar': {
            'weights': [round(float(x) * 100, 2) for x in w_minvar],
            'ret': round(r_mv * 100, 3),
            'vol': round(v_mv * 100, 3),
            'sharpe': round((r_mv - rf) / v_mv, 4) if v_mv > 0 else 0,
        },
        'frontier_long':  frontier,
        'frontier_short': frontier_short,
    }


# ─── Analisi di diversificazione / contributo marginale ───────────────────────

def _tangency_long(mu, cov, rf):
    """Portafoglio di tangenza long-only su (mu, cov). Ritorna (w, ret, vol, sharpe)."""
    n = len(mu)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        return -(r - rf) / v if v > 1e-8 else 0

    eq = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    res = minimize(neg_sharpe, w0, method='SLSQP',
                   bounds=[(0.0, 1.0)] * n, constraints=[eq],
                   options={'ftol': 1e-12, 'maxiter': 1000})
    w = res.x
    r = float(w @ mu)
    v = float(np.sqrt(w @ cov @ w))
    sharpe = (r - rf) / v if v > 0 else 0
    return w, r, v, sharpe


def calc_diversification(symbols_data, rf):
    """
    Analisi di diversificazione e contributo marginale di ogni titolo.
    Per ciascun titolo: peso nella tangenza, contributo marginale al rischio (MCTR),
    correlazione media con gli altri, e ΔSharpe se rimosso dal portafoglio.
    """
    n = len(symbols_data)
    aligned, align_mode = align_returns(symbols_data)   # (n, T)

    mu   = aligned.mean(axis=1) * 252
    cov  = np.cov(aligned) * 252
    corr = np.corrcoef(aligned)
    vols = np.sqrt(np.diag(cov))

    # Portafoglio di tangenza completo
    w, r_p, v_p, sharpe_full = _tangency_long(mu, cov, rf)

    # Contributo marginale al rischio (% della varianza di portafoglio)
    Sw     = cov @ w
    var_p  = float(w @ cov @ w)
    mctr   = [float(w[i] * Sw[i] / var_p * 100) if var_p > 0 else 0 for i in range(n)]

    # Diversification ratio (con i pesi di tangenza): media pesata delle vol / vol portafoglio
    dr = float((w @ vols) / v_p) if v_p > 0 else 0

    per_asset = []
    for i in range(n):
        # Sharpe del titolo da solo
        sharpe_solo = float((mu[i] - rf) / vols[i]) if vols[i] > 0 else 0
        # correlazione media con gli altri
        avg_corr = float((corr[i].sum() - 1) / (n - 1)) if n > 1 else 0
        # ΔSharpe se rimosso: ri-ottimizza il portafoglio senza il titolo i
        if n > 2:
            idx = [j for j in range(n) if j != i]
            _, _, _, sharpe_wo = _tangency_long(mu[idx], cov[np.ix_(idx, idx)], rf)
        else:
            # resta un solo titolo → la sua "tangenza" è il titolo stesso
            j = 1 - i
            sharpe_wo = float((mu[j] - rf) / vols[j]) if vols[j] > 0 else 0
        per_asset.append({
            'symbol':        symbols_data[i]['symbol'],
            'weight':        round(float(w[i]) * 100, 2),
            'mctr':          round(mctr[i], 2),
            'avg_corr':      round(avg_corr, 3),
            'asset_ret':     round(float(mu[i]) * 100, 2),
            'asset_vol':     round(float(vols[i]) * 100, 2),
            'sharpe_solo':   round(sharpe_solo, 4),
            'sharpe_without': round(float(sharpe_wo), 4),
            'delta_sharpe':  round(float(sharpe_full - sharpe_wo), 4),
        })

    return {
        'symbols':       [d['symbol'] for d in symbols_data],
        'corr':          [[round(float(corr[i][j]), 3) for j in range(n)] for i in range(n)],
        'rf':            round(rf * 100, 3),
        'align_mode':    align_mode,
        'hist_days':     int(aligned.shape[1]),
        'portfolio': {
            'ret':                  round(r_p * 100, 3),
            'vol':                  round(v_p * 100, 3),
            'sharpe':               round(sharpe_full, 4),
            'diversification_ratio': round(dr, 3),
        },
        'assets': per_asset,
    }


# ─── Monte Carlo di portafoglio ───────────────────────────────────────────────

def fetch_closes_for_mc(symbols_data, hist_period):
    """
    Riscarica uno storico lungo per i titoli del paniere: il basket del frontend
    ha solo 1 anno di closes, troppo poco (e troppo regime-dipendente) per
    simulare orizzonti pluriennali. Fallback sui closes forniti se il download fallisce.
    """
    out = []
    for d in symbols_data:
        closes = dates = None
        try:
            h = yf.Ticker(d['symbol']).history(period=hist_period)['Close'].dropna()
            if len(h) >= 120:
                closes = [float(x) for x in h.tolist()]
                # date della stessa serie: permettono ad align_returns di allineare
                # per data anche in Monte Carlo, non solo per posizione
                dates  = [ts.strftime('%Y-%m-%d') for ts in h.index.tz_localize(None).normalize()]
        except Exception:
            pass
        if closes:
            out.append({'symbol': d['symbol'], 'closes': closes, 'dates': dates})
        else:
            # fallback: si riusano i dati del frontend, date comprese se c'erano
            out.append({'symbol': d['symbol'],
                        'closes': d.get('closes') or [],
                        'dates':  d.get('dates')})
    return out


def calc_montecarlo(symbols_data, rf, years, capital, target, method, wmode, n_sims=3000):
    """
    Simula n_sims traiettorie del portafoglio su `years` anni.
    method: 'boot' = bootstrap dei rendimenti giornalieri storici (conserva le code
            grasse), 'normal' = GBM gaussiano con stessi mu/sigma (per confronto).
    wmode:  'tangency' | 'minvar' | 'equal' — pesi con cui simulare.
    """
    n = len(symbols_data)
    aligned, align_mode = align_returns(symbols_data)
    min_len = aligned.shape[1]
    mu  = aligned.mean(axis=1) * 252
    cov = np.cov(aligned) * 252

    if wmode == 'equal':
        w = np.ones(n) / n
    elif wmode == 'minvar':
        eq = {'type': 'eq', 'fun': lambda w_: np.sum(w_) - 1}
        res = minimize(lambda w_: float(w_ @ cov @ w_), np.ones(n) / n, method='SLSQP',
                       bounds=[(0.0, 1.0)] * n, constraints=[eq],
                       options={'ftol': 1e-12, 'maxiter': 1000})
        w = res.x
    else:
        w, _, _, _ = _tangency_long(mu, cov, rf)

    # Serie storica dei rendimenti giornalieri del portafoglio (pesi fissi)
    port_daily = w @ aligned
    ann_ret = float(port_daily.mean() * 252)
    ann_vol = float(port_daily.std(ddof=1) * np.sqrt(252))

    T = int(252 * years)
    rng = np.random.default_rng()
    if method == 'normal':
        sims = rng.normal(port_daily.mean(), port_daily.std(ddof=1), size=(n_sims, T))
    else:
        sims = port_daily[rng.integers(0, len(port_daily), size=(n_sims, T))]

    paths = capital * np.cumprod(1.0 + sims, axis=1)   # (n_sims, T)
    del sims
    finals = paths[:, -1].copy()

    # Max drawdown per traiettoria
    cummax = np.maximum.accumulate(paths, axis=1)
    maxdd  = (paths / cummax - 1.0).min(axis=1)
    del cummax

    # Bande percentili su ~80 punti temporali (per il fan chart)
    steps = np.unique(np.linspace(0, T - 1, 80).astype(int))
    sub = paths[:, steps]
    bands = {str(p): [round(float(x)) for x in np.percentile(sub, p, axis=0)]
             for p in (5, 25, 50, 75, 95)}
    sample_idx = rng.choice(n_sims, size=min(24, n_sims), replace=False)
    sample_paths = [[round(float(x)) for x in sub[i]] for i in sample_idx]

    k = max(1, int(0.05 * n_sims))
    med = float(np.median(finals))
    return {
        'symbols':  [d['symbol'] for d in symbols_data],
        'weights':  [round(float(x) * 100, 1) for x in w],
        'hist_days': int(min_len),
        'align_mode': align_mode,
        'wmode': wmode, 'method': method, 'years': years,
        'n_sims': n_sims, 'capital': capital, 'target': target,
        'ann_ret': round(ann_ret * 100, 2),
        'ann_vol': round(ann_vol * 100, 2),
        'median':  round(med),
        'mean':    round(float(finals.mean())),
        'p5':      round(float(np.percentile(finals, 5))),
        'p95':     round(float(np.percentile(finals, 95))),
        'cvar5':   round(float(np.sort(finals)[:k].mean())),
        'cagr_median': round(((med / capital) ** (1 / years) - 1) * 100, 2),
        'prob_loss':   round(float((finals < capital).mean()) * 100, 1),
        'prob_target': round(float((finals >= target).mean()) * 100, 1) if target else None,
        'maxdd_median': round(float(np.median(maxdd)) * 100, 1),
        'maxdd_worst5': round(float(np.percentile(maxdd, 5)) * 100, 1),
        'time_axis':    [round(float((s + 1) / 252), 2) for s in steps],
        'bands':        bands,
        'sample_paths': sample_paths,
        'finals':       [round(float(x)) for x in finals],
    }


# ─── NAV del portafoglio (buy & hold) ─────────────────────────────────────────

def calc_portfolio_nav(items, rf, period, benchmark):
    """
    Tratta il portafoglio come un titolo unico: compra le quote al primo giorno
    comune secondo i pesi indicati e le tiene (buy & hold). Ritorna il NAV
    normalizzato a 100, le metriche e la deriva dei pesi nel tempo.
    items: [{'symbol', 'weight'}] con pesi in % (normalizzati qui).
    """
    try:
        import pandas as pd
        series = {}
        for it in items:
            sym = it['symbol']
            h = yf.Ticker(sym).history(period=period)['Close'].dropna()
            if len(h) < 60:
                return {'error': f'Storico insufficiente per {sym} ({len(h)} sedute)'}
            h.index = h.index.tz_localize(None).normalize()
            series[sym] = h

        bench_ok = False
        if benchmark:
            try:
                hb = yf.Ticker(benchmark).history(period=period)['Close'].dropna()
                hb.index = hb.index.tz_localize(None).normalize()
                if len(hb) >= 60:
                    series['__BENCH__'] = hb
                    bench_ok = True
            except Exception:
                pass

        df = pd.DataFrame(series).dropna()
        if len(df) < 60:
            return {'error': f'Solo {len(df)} sedute comuni tra i titoli: prova un periodo più lungo o togli il benchmark.'}

        syms = [it['symbol'] for it in items]
        w = np.array([max(0.0, float(it.get('weight', 0) or 0)) for it in items])
        if w.sum() <= 0:
            w = np.ones(len(items))
        w = w / w.sum()

        P = df[syms].values                    # (T, n) prezzi total-return
        shares = w / P[0]                      # quote comprate il giorno 1 con capitale 1
        nav = P @ shares
        nav = nav / nav[0] * 100

        # Deriva dei pesi: da iniziali a attuali
        end_vals = shares * P[-1]
        w_end = end_vals / end_vals.sum()

        r = nav[1:] / nav[:-1] - 1
        n_years = len(r) / 252
        total = float(nav[-1] / 100 - 1)
        cagr  = float((nav[-1] / 100) ** (1 / n_years) - 1) if n_years > 0 else 0
        vol   = float(np.std(r, ddof=1) * np.sqrt(252))
        sharpe = (float(np.mean(r) * 252) - rf) / vol if vol > 0 else 0
        peak  = np.maximum.accumulate(nav)
        dd    = nav / peak - 1
        best_i, worst_i = int(np.argmax(r)), int(np.argmin(r))
        dates_all = [d.strftime('%Y-%m-%d') for d in df.index]

        # Finestra troncata: il periodo effettivo è limitato dal titolo con lo
        # storico più corto (o dal benchmark). Serve ad avvisare quando "10 anni"
        # in realtà restituisce molto meno perché un titolo è troppo giovane.
        starts = {sym: s.index[0] for sym, s in series.items()}
        lim = max(starts, key=lambda k: starts[k])
        limiting_symbol = benchmark if lim == '__BENCH__' else lim
        period_years = {'1y': 1, '3y': 3, '5y': 5, '10y': 10}.get(period, 3)
        truncated = n_years < period_years * 0.9

        bench_nav = None
        if bench_ok:
            b = df['__BENCH__'].values
            bench_nav = b / b[0] * 100

        idx = np.unique(np.linspace(0, len(nav) - 1, min(300, len(nav))).astype(int))
        return {
            'symbols': syms, 'period': period,
            'benchmark': benchmark if bench_ok else None,
            'n_days': int(len(nav)),
            'from': dates_all[0], 'to': dates_all[-1],
            'requested_years': period_years,
            'actual_years': round(n_years, 1),
            'truncated': bool(truncated),
            'limiting_symbol': limiting_symbol,
            'total_ret':  round(total * 100, 2),
            'cagr':       round(cagr * 100, 2),
            'vol':        round(vol * 100, 2),
            'sharpe':     round(sharpe, 3),
            'max_dd':     round(float(dd.min()) * 100, 1),
            'current_dd': round(float(dd[-1]) * 100, 1),
            'best_day':   {'date': dates_all[best_i + 1],  'ret': round(float(r[best_i]) * 100, 2)},
            'worst_day':  {'date': dates_all[worst_i + 1], 'ret': round(float(r[worst_i]) * 100, 2)},
            'bench_total': round(float(bench_nav[-1] / 100 - 1) * 100, 2) if bench_ok else None,
            'weights': [{'symbol': syms[i],
                         'initial': round(float(w[i]) * 100, 1),
                         'current': round(float(w_end[i]) * 100, 1)} for i in range(len(syms))],
            'dates': [dates_all[i] for i in idx],
            'nav':   [round(float(nav[i]), 2) for i in idx],
            'bench': [round(float(bench_nav[i]), 2) for i in idx] if bench_ok else None,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {'error': str(e)}


# ─── Dividendi / Total return ─────────────────────────────────────────────────

def calc_dividends(symbol, period):
    """
    Scomposizione del rendimento: prezzo puro (Close) vs total return con cedole
    reinvestite (Adj Close), più storico dividendi, yield TTM, crescita e tagli.
    """
    try:
        import pandas as pd
        t = yf.Ticker(symbol)
        df = t.history(period=period, auto_adjust=False)
        if df.empty:
            return {'error': f'Nessun dato per {symbol}'}

        close = df['Close'].dropna()
        adj   = df['Adj Close'].dropna() if 'Adj Close' in df.columns else close
        common = close.index.intersection(adj.index)
        if len(common) < 60:
            return {'error': f'Dati insufficienti per {symbol} ({len(common)} sedute)'}
        close, adj = close.loc[common], adj.loc[common]

        divs = df['Dividends'] if 'Dividends' in df.columns else pd.Series(dtype=float)
        divs = divs[divs > 0]

        price_growth = close / close.iloc[0]
        total_growth = adj / adj.iloc[0]
        n_years = len(close) / 252

        price_ret = float(price_growth.iloc[-1] - 1)
        total_ret = float(total_growth.iloc[-1] - 1)
        div_contrib_pp = (total_ret - price_ret) * 100  # punti percentuali dalle cedole
        div_share = (div_contrib_pp / (total_ret * 100) * 100) if abs(total_ret) > 1e-9 else 0

        last_price = float(close.iloc[-1])
        cutoff = close.index[-1] - pd.Timedelta(days=365)
        ttm_sum = float(divs[divs.index >= cutoff].sum())
        yield_ttm = ttm_sum / last_price if last_price > 0 else 0
        n_payments_ttm = int((divs.index >= cutoff).sum())

        # Dividendi per anno solare. Parziali: l'anno corrente (non finito) e il
        # primo anno della finestra se questa non parte da inizio anno.
        per_year = divs.groupby(divs.index.year).sum()
        current_year = close.index[-1].year
        first_year = close.index[0].year
        first_is_partial = close.index[0].dayofyear > 31
        years = [{'year': int(y), 'amount': round(float(a), 4),
                  'partial': bool(y == current_year or (y == first_year and first_is_partial))}
                 for y, a in per_year.items()]

        # CAGR e tagli calcolati solo sugli anni pieni
        full = [y for y in years if not y['partial']]
        div_cagr = None
        cuts = 0
        if len(full) >= 2 and full[0]['amount'] > 0:
            span = full[-1]['year'] - full[0]['year']
            if span > 0:
                div_cagr = round(((full[-1]['amount'] / full[0]['amount']) ** (1 / span) - 1) * 100, 2)
            cuts = sum(1 for i in range(1, len(full)) if full[i]['amount'] < full[i-1]['amount'] * 0.999)

        last_div = None
        if len(divs) > 0:
            last_div = {'amount': round(float(divs.iloc[-1]), 4),
                        'date': divs.index[-1].strftime('%Y-%m-%d')}

        info = t.fast_info
        currency = getattr(info, 'currency', '') or ''

        # Serie per il grafico (downsample ≤ 300 punti)
        idx = np.unique(np.linspace(0, len(close) - 1, min(300, len(close))).astype(int))
        return {
            'symbol': symbol, 'period': period, 'currency': currency,
            'n_days': int(len(close)), 'n_years': round(n_years, 1),
            'price_ret':      round(price_ret * 100, 2),
            'total_ret':      round(total_ret * 100, 2),
            'div_contrib_pp': round(div_contrib_pp, 2),
            'div_share':      round(float(div_share), 1),
            'yield_ttm':      round(yield_ttm * 100, 2),
            'ttm_sum':        round(ttm_sum, 4),
            'n_payments_ttm': n_payments_ttm,
            'div_cagr':       div_cagr,
            'cuts':           cuts,
            'n_full_years':   len(full),
            'last_div':       last_div,
            'has_dividends':  bool(len(divs) > 0),
            'dates':        [close.index[i].strftime('%Y-%m-%d') for i in idx],
            'price_growth': [round(float(price_growth.iloc[i]), 4) for i in idx],
            'total_growth': [round(float(total_growth.iloc[i]), 4) for i in idx],
            'per_year':     years,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {'error': str(e)}


# ─── Backtest out-of-sample ───────────────────────────────────────────────────

def calc_backtest(symbols, rf, hist, lookback_days, rebal_days, benchmark):
    """
    Walk-forward: a ogni data di ribilanciamento stima i pesi (tangenza long-only,
    minima varianza) usando SOLO la finestra passata, li applica al periodo
    successivo, e confronta con equal-weight e un benchmark. Risponde alla
    domanda: i pesi di Markowitz reggono fuori campione?
    """
    import pandas as pd

    series = {}
    for sym in symbols:
        h = yf.Ticker(sym).history(period=hist)['Close'].dropna()
        if len(h) < 120:
            return {'error': f'Storico insufficiente per {sym} ({len(h)} sedute)'}
        h.index = h.index.tz_localize(None).normalize()
        series[sym] = h

    bench_ok = False
    if benchmark:
        try:
            hb = yf.Ticker(benchmark).history(period=hist)['Close'].dropna()
            hb.index = hb.index.tz_localize(None).normalize()
            if len(hb) >= 120:
                series['__BENCH__'] = hb
                bench_ok = True
        except Exception:
            pass

    # Allinea sulle date di borsa comuni (gestisce calendari misti USA/Europa)
    df = pd.DataFrame(series).dropna()
    rets = df.pct_change().dropna()
    n = len(symbols)
    R = rets[symbols].values                    # (T, n)
    bench_r = rets['__BENCH__'].values if bench_ok else None
    dates = [d.strftime('%Y-%m-%d') for d in rets.index]
    T = len(R)

    min_required = lookback_days + rebal_days + 20
    if T < min_required:
        return {'error': f'Servono almeno {min_required} sedute comuni tra i titoli; disponibili {T}. Allunga lo storico totale.'}

    strat = {'tangency': [], 'minvar': [], 'equal': []}
    w_prev = {'tangency': None, 'minvar': None}
    turnover = {'tangency': [], 'minvar': []}
    eq_constraint = {'type': 'eq', 'fun': lambda w_: np.sum(w_) - 1}
    w_eq = np.ones(n) / n
    n_rebal = 0

    t = lookback_days
    while t < T:
        win = R[t - lookback_days:t]
        mu = win.mean(axis=0) * 252
        cov = np.cov(win.T) * 252
        w_tan, _, _, _ = _tangency_long(mu, cov, rf)
        res = minimize(lambda w_: float(w_ @ cov @ w_), w_eq, method='SLSQP',
                       bounds=[(0.0, 1.0)] * n, constraints=[eq_constraint],
                       options={'ftol': 1e-12, 'maxiter': 1000})
        w_mv = res.x

        end = min(t + rebal_days, T)
        seg = R[t:end]
        for name, w in (('tangency', w_tan), ('minvar', w_mv), ('equal', w_eq)):
            strat[name].extend((seg @ w).tolist())
        for name, w in (('tangency', w_tan), ('minvar', w_mv)):
            if w_prev[name] is not None:
                turnover[name].append(float(np.abs(w - w_prev[name]).sum() / 2))
            w_prev[name] = w
        n_rebal += 1
        t = end

    eval_dates = dates[lookback_days:]
    metrics = {}

    def build_curve(name, r_list):
        r = np.array(r_list)
        eqty = np.cumprod(1 + r)
        yrs = len(r) / 252
        cagr = float(eqty[-1] ** (1 / yrs) - 1) if yrs > 0 else 0
        vol = float(r.std(ddof=1) * np.sqrt(252))
        sharpe = (float(r.mean() * 252) - rf) / vol if vol > 0 else 0
        peak = np.maximum.accumulate(eqty)
        mdd = float((eqty / peak - 1).min())
        metrics[name] = {
            'total':  round((float(eqty[-1]) - 1) * 100, 1),
            'cagr':   round(cagr * 100, 2),
            'vol':    round(vol * 100, 2),
            'sharpe': round(sharpe, 3),
            'maxdd':  round(mdd * 100, 1),
        }
        return eqty

    curves = {name: build_curve(name, r) for name, r in strat.items()}
    if bench_ok:
        curves['bench'] = build_curve('bench', bench_r[lookback_days:].tolist())

    # Downsample per il grafico
    L = len(eval_dates)
    idx = np.unique(np.linspace(0, L - 1, min(400, L)).astype(int))
    return {
        'symbols': symbols,
        'benchmark': benchmark if bench_ok else None,
        'lookback_days': lookback_days,
        'rebal_days': rebal_days,
        'n_rebalances': n_rebal,
        'n_days': L,
        'period': {'from': eval_dates[0], 'to': eval_dates[-1]},
        'turnover': {k: round(float(np.mean(v)) * 100, 1) if v else 0.0 for k, v in turnover.items()},
        'dates':  [eval_dates[i] for i in idx],
        'equity': {name: [round(float(c[i]), 4) for i in idx] for name, c in curves.items()},
        'metrics': metrics,
    }


# ─── Scheda-titolo di rischio ─────────────────────────────────────────────────

def calc_risk(symbol, period, rf):
    """
    Profilo di rischio del singolo titolo su rendimenti giornalieri:
    Sortino, max drawdown + serie underwater, VaR/CVaR storici e parametrico,
    skewness/curtosi + test di normalità Jarque-Bera, volatilità rolling.
    """
    try:
        from scipy import stats
        t = yf.Ticker(symbol)
        hist = t.history(period=period)['Close'].dropna()
        if len(hist) < 60:
            return {'error': f'Dati insufficienti per {symbol} ({len(hist)} giorni). Prova un periodo più lungo.'}
        closes = hist.values.astype(float)
        dates  = [d.strftime('%Y-%m-%d') for d in hist.index]
        dr = (closes[1:] - closes[:-1]) / closes[:-1]
        n  = len(dr)

        ann_ret = float(np.mean(dr) * 252)
        ann_vol = float(np.std(dr, ddof=1) * np.sqrt(252))
        sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0

        # Sortino: penalizza solo la volatilità sotto il risk-free
        rf_d     = rf / 252
        downside = np.minimum(dr - rf_d, 0)
        dd_dev   = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252))
        sortino  = (ann_ret - rf) / dd_dev if dd_dev > 0 else 0

        # Drawdown su equity normalizzata
        equity = closes / closes[0]
        peak   = np.maximum.accumulate(equity)
        dd     = equity / peak - 1
        i_trough = int(np.argmin(dd))
        max_dd   = float(dd[i_trough])
        i_peak   = int(np.argmax(equity[:i_trough + 1])) if i_trough > 0 else 0
        calmar   = ann_ret / abs(max_dd) if max_dd < 0 else 0

        # VaR / CVaR storici (giornalieri) + parametrico normale per confronto
        var95  = float(np.percentile(dr, 5))
        var99  = float(np.percentile(dr, 1))
        cvar95 = float(dr[dr <= var95].mean()) if (dr <= var95).any() else var95
        cvar99 = float(dr[dr <= var99].mean()) if (dr <= var99).any() else var99
        var95_param = float(np.mean(dr) - 1.645 * np.std(dr, ddof=1))

        # Momenti superiori + normalità
        skew = float(stats.skew(dr))
        kurt = float(stats.kurtosis(dr))  # curtosi in eccesso (normale = 0)
        jb   = stats.jarque_bera(dr)
        jb_p = float(jb.pvalue if hasattr(jb, 'pvalue') else jb[1])

        # Volatilità rolling 21 giorni, annualizzata
        win  = 21
        roll = [float(np.std(dr[i - win:i], ddof=1) * np.sqrt(252)) for i in range(win, n + 1)]

        return {
            'symbol': symbol, 'period': period, 'n_days': n,
            'ann_ret':      round(ann_ret * 100, 2),
            'ann_vol':      round(ann_vol * 100, 2),
            'sharpe':       round(sharpe, 3),
            'sortino':      round(sortino, 3),
            'downside_dev': round(dd_dev * 100, 2),
            'calmar':       round(calmar, 3),
            'max_dd':       round(max_dd * 100, 2),
            'max_dd_days':  int(i_trough - i_peak),
            'current_dd':   round(float(dd[-1]) * 100, 2),
            'var95':        round(var95 * 100, 2),
            'var99':        round(var99 * 100, 2),
            'cvar95':       round(cvar95 * 100, 2),
            'cvar99':       round(cvar99 * 100, 2),
            'var95_param':  round(var95_param * 100, 2),
            'skew':         round(skew, 3),
            'kurtosis':     round(kurt, 3),
            'jb_pvalue':    round(jb_p, 5),
            'normal':       bool(jb_p >= 0.05),
            'pos_days':     round(float((dr > 0).mean()) * 100, 1),
            'dates':          dates[1:], 
            'drawdown':       [round(float(x) * 100, 2) for x in dd[1:]], 
            'daily_returns':  [round(float(x) * 100, 3) for x in dr], 
            'rolling_vol':    [round(v * 100, 2) for v in roll],
            'rolling_dates':  dates[win:],
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {'error': str(e)}


# ─── CAPM: beta, alpha, R², SML e SCL ─────────────────────────────────────────

def calc_capm(symbol, benchmark, period, rf):
    """CAPM analysis: beta, alpha, R², SML, SCL"""
    try:
        import numpy as np
        t1 = yf.Ticker(symbol)
        t2 = yf.Ticker(benchmark)
        h1 = t1.history(period=period)['Close']
        h2 = t2.history(period=period)['Close']

        # Resample to monthly returns
        r1 = h1.resample('ME').last().pct_change().dropna()
        r2 = h2.resample('ME').last().pct_change().dropna()

        # Align
        common = r1.index.intersection(r2.index)
        if len(common) < 6:
            return {'error': f'Dati insufficienti ({len(common)} mesi comuni). Prova un periodo più lungo.'}
        r1 = r1.loc[common].values
        r2 = r2.loc[common].values
        n = len(r1)

        # Il CAPM si stima sui rendimenti IN ECCESSO sul risk-free:
        #   (r_titolo - rf) = alpha + beta * (r_mercato - rf) + eps
        # Senza questa sottrazione l'intercetta non e' l'alpha di Jensen:
        # differirebbe da esso di rf*(1 - beta).
        rf_m = rf / 12
        e1 = r1 - rf_m
        e2 = r2 - rf_m

        # Regressione OLS con standard error robusti HAC (Newey-West).
        # I residui delle serie finanziarie sono eteroschedastici e autocorrelati:
        # gli errori standard OLS classici li sottostimano, gonfiando le t-stat e
        # facendo sembrare significativi alpha che non lo sono.
        import statsmodels.api as sm
        X = sm.add_constant(e2)
        nw_lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))   # Newey-West (1994)
        fit = sm.OLS(e1, X).fit(cov_type='HAC',
                                cov_kwds={'maxlags': nw_lags, 'use_correction': True})

        alpha_monthly, beta = float(fit.params[0]),  float(fit.params[1])
        se_alpha,  se_beta  = float(fit.bse[0]),     float(fit.bse[1])
        t_alpha,   t_beta   = float(fit.tvalues[0]), float(fit.tvalues[1])
        p_alpha,   p_beta   = float(fit.pvalues[0]), float(fit.pvalues[1])
        r_squared           = float(fit.rsquared)
        ci = fit.conf_int()                       # intervalli di confidenza al 95%
        beta_ci = [float(ci[1][0]), float(ci[1][1])]

        # Annualize
        asset_ret  = float((1 + np.mean(r1)) ** 12 - 1)
        market_ret = float((1 + np.mean(r2)) ** 12 - 1)
        alpha_annual = float((1 + alpha_monthly) ** 12 - 1)
        asset_vol  = float(np.std(r1, ddof=1) * np.sqrt(12))
        market_vol = float(np.std(r2, ddof=1) * np.sqrt(12))

        # CAPM expected return
        capm_expected = rf + beta * (market_ret - rf)

        # Ratios
        sharpe_asset = (asset_ret - rf) / asset_vol if asset_vol > 0 else 0
        treynor = (asset_ret - rf) / beta if abs(beta) > 0.001 else 0

        return {
            'beta':           round(float(beta), 4),
            'alpha_monthly':  round(float(alpha_monthly), 6),
            'alpha_annual':   round(alpha_annual, 4),
            'r_squared':      round(float(r_squared), 4),
            'asset_ret':      round(asset_ret, 4),
            'market_ret':     round(market_ret, 4),
            'asset_vol':      round(asset_vol, 4),
            'market_vol':     round(market_vol, 4),
            'capm_expected':  round(capm_expected, 4),
            'sharpe_asset':   round(sharpe_asset, 4),
            'treynor':        round(treynor, 4),
            # Significativita' statistica con errori standard Newey-West
            'se_beta':        round(se_beta, 4),
            't_beta':         round(t_beta, 3),
            'p_beta':         round(p_beta, 4),
            'beta_ci':        [round(beta_ci[0], 3), round(beta_ci[1], 3)],
            'se_alpha':       round(se_alpha, 6),
            't_alpha':        round(t_alpha, 3),
            'p_alpha':        round(p_alpha, 4),
            'alpha_signif':   bool(p_alpha < 0.05),
            'beta_signif':    bool(p_beta < 0.05),
            'nw_lags':        nw_lags,
            'monthly_asset':  [round(float(x), 5) for x in r1],
            'monthly_market': [round(float(x), 5) for x in r2],
            'n_months':       n,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {'error': str(e)}



# ─── Fama-French factors download ────────────────────────────────────────────
import io, zipfile, ssl, certifi

_ff_cache = {}  # cache per non scaricare ogni volta

def fetch_ff_data(dataset):
    """Scarica i fattori Fama-French dal sito di Kenneth French (Dartmouth)"""
    import time
    if dataset in _ff_cache:
        age = time.time() - _ff_cache[dataset]['ts']
        if age < 86400:  # cache 24h
            return _ff_cache[dataset]['df']

    url = f'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{dataset}_CSV.zip'
    # Verifica del certificato ATTIVA, validata contro il bundle CA di certifi
    # (elenco Mozilla, aggiornato) invece di quello di sistema: su macOS
    # quest'ultimo e' spesso obsoleto ed e' la vera causa dei fallimenti HTTPS.
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15',
        'Accept': '*/*',
        'Referer': 'https://mba.tuck.dartmouth.edu/',
    })
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
            raw = r.read()
    except ssl.SSLCertVerificationError as e:
        raise ValueError(
            "Verifica del certificato fallita per il sito di Ken French. "
            "Aggiorna i certificati con 'python3 -m pip install --upgrade certifi' "
            "(su macOS esegui anche Install Certificates.command in Applicazioni/Python 3.x). "
            f"Dettaglio: {e}"
        )

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        fname = [n for n in z.namelist() if n.upper().endswith('.CSV')][0]
        with z.open(fname) as f:
            text = f.read().decode('utf-8', errors='ignore')

    # Parsa il blocco dati mensili (YYYYMM, prima della sezione annuale)
    rows = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith(','):
            if rows: break  # fine blocco mensile
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4 and len(parts[0]) == 6:
            try:
                date = int(parts[0])
                vals = [float(v)/100 for v in parts[1:] if v.strip() not in ('', 'RF')]
                rows.append([date] + vals)
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"Nessun dato parsato da {dataset}")

    import pandas as pd
    cols_map = {
        'F-F_Research_Data_Factors':       ['date','Mkt-RF','SMB','HML','RF'],
        'F-F_Research_Data_5_Factors_2x3': ['date','Mkt-RF','SMB','HML','RMW','CMA','RF'],
        'F-F_Research_Data_Factors_daily': ['date','Mkt-RF','SMB','HML','RF'],
    }
    ncols = len(rows[0])
    cols = cols_map.get(dataset, [f'c{i}' for i in range(ncols)])[:ncols]
    df = pd.DataFrame(rows, columns=cols)
    df.index = pd.to_datetime(df['date'].astype(str), format='%Y%m')
    df = df.drop(columns=['date'])

    _ff_cache[dataset] = {'df': df, 'ts': time.time()}
    print(f"  [FF] {dataset}: {len(df)} mesi caricati")
    return df


def calc_fama_french(symbol, model_type, period, rf_annual):
    """
    Fama-French 3 o 5 fattori
    model_type: 'ff3' o 'ff5'
    """
    try:
        import statsmodels.api as sm

        # 1. Scarica fattori FF
        dataset = 'F-F_Research_Data_Factors' if model_type == 'ff3' else 'F-F_Research_Data_5_Factors_2x3'
        ff = fetch_ff_data(dataset)

        # 2. Scarica prezzi titolo
        t = yf.Ticker(symbol)
        hist = t.history(period=period)['Close']
        # Rendimenti mensili
        monthly = hist.resample('ME').last().pct_change().dropna()
        monthly.index = monthly.index.to_period('M').to_timestamp()

        # 3. Allinea date
        common = ff.index.intersection(monthly.index)
        if len(common) < 12:
            return {'error': f'Solo {len(common)} mesi comuni con i fattori FF. Prova un periodo più lungo.'}

        asset_r = monthly.loc[common].values
        ff_aligned = ff.loc[common]

        mkt_rf = ff_aligned['Mkt-RF'].values
        smb    = ff_aligned['SMB'].values
        hml    = ff_aligned['HML'].values
        rf_m   = ff_aligned['RF'].values
        asset_excess = asset_r - rf_m

        # 4. Regressione multipla
        if model_type == 'ff3':
            factors = [mkt_rf, smb, hml]
            factor_names = ['Mkt-RF', 'SMB', 'HML']
        else:
            rmw = ff_aligned['RMW'].values
            cma = ff_aligned['CMA'].values
            factors = [mkt_rf, smb, hml, rmw, cma]
            factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        X = sm.add_constant(np.column_stack(factors))

        # Standard error robusti HAC (Newey-West), come in calc_capm: i residui
        # sono eteroschedastici e autocorrelati, e con 3-5 regressori il rischio
        # di dichiarare significativo un fattore che non lo e' cresce.
        nw_lags = max(1, int(np.floor(4 * (len(common) / 100) ** (2 / 9))))
        res = sm.OLS(asset_excess, X).fit(cov_type='HAC',
                                          cov_kwds={'maxlags': nw_lags,
                                                    'use_correction': True})

        alpha_m = res.params[0]
        betas   = res.params[1:]
        pvals   = res.pvalues
        tstats  = res.tvalues
        r2      = res.rsquared
        r2_adj  = res.rsquared_adj

        # 5. Annualizza
        alpha_annual = float((1 + alpha_m) ** 12 - 1)
        asset_ret    = float((1 + np.mean(asset_r)) ** 12 - 1)
        asset_vol    = float(np.std(asset_r, ddof=1) * np.sqrt(12))

        # Rendimento atteso dal modello: rf + somma dei premi, ciascuno pesato
        # per l'esposizione del titolo a quel fattore.
        factor_means_annual = [float((1 + np.mean(f)) ** 12 - 1) for f in factors]
        ff_expected = rf_annual + sum(b * fm for b, fm in zip(betas, factor_means_annual))

        sharpe = (asset_ret - rf_annual) / asset_vol if asset_vol > 0 else 0

        # Rendimenti mensili per scatter
        return {
            'symbol':          symbol,
            'model':           model_type,
            'n_months':        int(len(common)),
            'alpha_monthly':   round(float(alpha_m), 6),
            'alpha_annual':    round(alpha_annual, 4),
            'alpha_pval':      round(float(pvals[0]), 4),
            'alpha_tstat':     round(float(tstats[0]), 3),
            'alpha_signif':    bool(pvals[0] < 0.05),
            'nw_lags':         nw_lags,
            'betas': {
                factor_names[i]: {
                    'value': round(float(betas[i]), 4),
                    'pval':  round(float(pvals[i+1]), 4),
                    'tstat': round(float(tstats[i+1]), 3),
                    'significant': bool(pvals[i+1] < 0.05),
                }
                for i in range(len(factor_names))
            },
            'r_squared':       round(float(r2), 4),
            'r_squared_adj':   round(float(r2_adj), 4),
            'asset_ret':       round(asset_ret, 4),
            'ff_expected':     round(float(ff_expected), 4),
            'asset_vol':       round(asset_vol, 4),
            'sharpe':          round(sharpe, 4),
            'monthly_asset':   [round(float(x), 5) for x in asset_r],
            'monthly_mktrf':   [round(float(x), 5) for x in mkt_rf],
            'fitted':          [round(float(x), 5) for x in res.fittedvalues],
            'residuals':       [round(float(x), 5) for x in res.resid],
            'factor_means_annual': {
                factor_names[i]: round(factor_means_annual[i], 4)
                for i in range(len(factor_names))
            },
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {'error': str(e)}

# ─── Momentum screener ────────────────────────────────────────────────────────

_sp500_cache   = {}   # lista costituenti S&P 500 (cache 24h)
_momentum_cache = {}  # risultati momentum per universo (cache 30 min)

# Universo ETF curato: ~40 nomi liquidi e riconoscibili, con etichetta didattica.
CURATED_ETFS = [
    # Settori USA (SPDR)
    ('XLK', 'Tecnologia'), ('XLF', 'Finanziari'), ('XLE', 'Energia'), ('XLV', 'Salute'),
    ('XLY', 'Consumi discrezionali'), ('XLP', 'Beni di prima necessità'), ('XLI', 'Industriali'),
    ('XLB', 'Materiali'), ('XLU', 'Utility'), ('XLRE', 'Immobiliare'), ('XLC', 'Comunicazioni'),
    # Broad / fattori
    ('SPY', 'S&P 500'), ('QQQ', 'Nasdaq 100'), ('IWM', 'Small cap USA'), ('RSP', 'S&P 500 equal weight'),
    ('MTUM', 'Fattore momentum'), ('QUAL', 'Fattore quality'), ('VLUE', 'Fattore value'), ('USMV', 'Minima volatilità'),
    # Tematici
    ('SMH', 'Semiconduttori'), ('SOXX', 'Semiconduttori'), ('IGV', 'Software'), ('SKYY', 'Cloud'),
    ('CIBR', 'Cybersecurity'), ('BOTZ', 'Robotica & AI'), ('ARKK', 'Innovazione ARK'),
    ('XBI', 'Biotech'), ('TAN', 'Solare'), ('LIT', 'Batterie & litio'), ('ICLN', 'Energia pulita'),
    # Obbligazioni / commodity / oro
    ('TLT', 'Treasury 20+ anni'), ('IEF', 'Treasury 7-10 anni'), ('GLD', 'Oro'), ('SLV', 'Argento'),
    ('GDX', "Miniere d'oro"), ('DBC', 'Commodity'),
    # Geografici
    ('EFA', 'Sviluppati ex-USA'), ('EEM', 'Mercati emergenti'), ('VGK', 'Europa'),
    ('EWJ', 'Giappone'), ('FXI', 'Cina'), ('INDA', 'India'),
]


def fetch_sp500_symbols():
    """Lista costituenti S&P 500 da Wikipedia (cache 24h). Certificato verificato
    contro il bundle CA di certifi, come per i fattori Fama-French."""
    import time
    if 'syms' in _sp500_cache and time.time() - _sp500_cache['ts'] < 86400:
        return _sp500_cache['syms']
    import pandas as pd
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
            html = r.read()
    except ssl.SSLCertVerificationError as e:
        raise ValueError(
            "Verifica del certificato fallita per Wikipedia. "
            "Aggiorna i certificati con 'python3 -m pip install --upgrade certifi'. "
            f"Dettaglio: {e}"
        )
    tabs = pd.read_html(io.BytesIO(html))
    # Yahoo usa '-' dove Wikipedia usa '.' (es. BRK.B → BRK-B)
    syms = [str(s).replace('.', '-').strip() for s in tabs[0]['Symbol'].tolist()]
    _sp500_cache['syms'] = syms
    _sp500_cache['ts'] = time.time()
    print(f"  [SP500] {len(syms)} costituenti caricati")
    return syms


def calc_momentum(universe):
    """
    Screener di momentum. Per ogni titolo dell'universo:
      - rendimento a 6 e 12 settimane (≈30 e 60 sedute, total return)
      - volatilità annualizzata sulle ultime 12 settimane
      - punteggio risk-adjusted = rendimento 12s / volatilità
    Ordina per rendimento a 12 settimane. Cache 30 min per non martellare Yahoo.
    universe: 'sp500' | 'etf'
    """
    import time
    if universe in _momentum_cache and time.time() - _momentum_cache[universe]['ts'] < 1800:
        return _momentum_cache[universe]['data']

    import pandas as pd
    pairs   = CURATED_ETFS if universe == 'etf' else [(s, '') for s in fetch_sp500_symbols()]
    labels  = {s: lbl for s, lbl in pairs}
    tickers = [s for s, _ in pairs]

    # Download a blocchi: più robusto ai timeout, e un blocco fallito non
    # affonda gli altri. 5 mesi = margine oltre le 12 settimane richieste.
    frames = []
    CHUNK = 100
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            d = yf.download(chunk, period='5mo', progress=False,
                            auto_adjust=True, threads=True)['Close']
            if isinstance(d, pd.Series):        # un solo ticker → Series
                d = d.to_frame(name=chunk[0])
            frames.append(d)
        except Exception as e:
            print(f"  [Momentum] blocco {i}-{i+len(chunk)} fallito: {e}")
            continue
    if not frames:
        return {'error': 'Download dei prezzi fallito da Yahoo Finance. Riprova tra poco.'}

    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]    # dedup difensivo

    W6, W12 = 30, 60
    rows = []
    for sym in tickers:
        if sym not in px.columns:
            continue
        s = px[sym].dropna()
        if len(s) < W12 + 5:
            continue
        p_now = float(s.iloc[-1])
        ret6  = p_now / float(s.iloc[-W6 - 1]) - 1
        ret12 = p_now / float(s.iloc[-W12 - 1]) - 1
        dr    = s.iloc[-W12:].pct_change().dropna().values
        vol   = float(dr.std(ddof=1) * math.sqrt(252)) if len(dr) > 1 else 0.0
        risk_adj = (ret12 / vol) if vol > 1e-6 else 0.0
        rows.append({
            'symbol':   sym,
            'label':    labels.get(sym, ''),
            'ret6':     round(ret6 * 100, 1),
            'ret12':    round(ret12 * 100, 1),
            'vol':      round(vol * 100, 1),
            'risk_adj': round(risk_adj, 2),
        })

    rows.sort(key=lambda r: r['ret12'], reverse=True)
    result = {
        'universe':  universe,
        'as_of':     str(px.index[-1].date()) if len(px.index) else None,
        'requested': len(tickers),
        'valid':     len(rows),
        'rows':      rows,
    }
    _momentum_cache[universe] = {'data': result, 'ts': time.time()}
    print(f"  [Momentum] {universe}: {len(rows)}/{len(tickers)} titoli validi")
    return result


class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        status = str(args[1]) if len(args) > 1 else '?'
        path   = args[0].split(' ')[1] if ' ' in str(args[0]) else str(args[0])
        icon   = 'OK ' if status.startswith('2') else 'ERR'
        print(f"  [{icon}] {status}  {path}")

    def send_json(self, data, status=200):
        # NaN/Inf → null: json.dumps emetterebbe i token NaN/Infinity (JSON non valido)
        # che farebbero fallire res.json() nel browser.
        def clean(o):
            if isinstance(o, float):
                return o if math.isfinite(o) else None
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [clean(v) for v in o]
            return o
        body = json.dumps(clean(data), ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def send_html_file(self, fname):
        try:
            with open(fname, 'rb') as f:
                body = f.read()
        except OSError:
            self.send_json({'error': f'{fname} non trovato'}, 404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path
        try:
            if p in ('/', '/index.html', '/portfolio-markowitz.html'):
                self.send_html_file('portfolio-markowitz.html')
            elif p.startswith('/api/search/'):
                q = urllib.parse.unquote(p[len('/api/search/'):]).strip()
                self.send_json(search_tickers(q))
            elif p.startswith('/api/quote/'):
                sym = urllib.parse.unquote(p[len('/api/quote/'):]).strip()
                self.send_json(get_quote(sym))
            elif p.startswith('/api/chart/'):
                sym = urllib.parse.unquote(p[len('/api/chart/'):]).strip()
                self.send_json(get_chart(sym))
            elif p == '/api/ping':
                self.send_json({'status': 'ok'})
            elif p.startswith('/api/momentum/'):
                universe = urllib.parse.unquote(p[len('/api/momentum/'):]).strip() or 'sp500'
                if universe not in ('sp500', 'etf'):
                    universe = 'sp500'
                print(f'  [Momentum] universo {universe}...')
                self.send_json(calc_momentum(universe))
            elif p.startswith('/api/ff/'):
                parts = p[len('/api/ff/'):].split('/')
                if len(parts) >= 3:
                    sym      = urllib.parse.unquote(parts[0])
                    model_t  = parts[1]  # 'ff3' or 'ff5'
                    period   = parts[2] if len(parts) > 2 else '3y'
                    rf_val   = float(parts[3]) if len(parts) > 3 else 0.0365
                    print(f'  [FF] {sym} {model_t} ({period})')
                    self.send_json(calc_fama_french(sym, model_t, period, rf_val))
                else:
                    self.send_json({'error': 'Usage: /api/ff/SYMBOL/ff3|ff5/PERIOD/RF'}, 400)
            elif p.startswith('/api/dividends/'):
                parts = p[len('/api/dividends/'):].split('/')
                sym    = urllib.parse.unquote(parts[0])
                period = parts[1] if len(parts) > 1 else '5y'
                print(f'  [Dividendi] {sym} ({period})')
                self.send_json(calc_dividends(sym, period))
            elif p.startswith('/api/risk/'):
                parts = p[len('/api/risk/'):].split('/')
                sym    = urllib.parse.unquote(parts[0])
                period = parts[1] if len(parts) > 1 else '3y'
                rf_val = float(parts[2]) if len(parts) > 2 else 0.0365
                print(f'  [Risk] {sym} ({period})')
                self.send_json(calc_risk(sym, period, rf_val))
            elif p.startswith('/api/capm/'):
                parts = p[len('/api/capm/'):].split('/')
                if len(parts) >= 3:
                    sym   = urllib.parse.unquote(parts[0])
                    bench = urllib.parse.unquote(parts[1])
                    period = parts[2] if len(parts) > 2 else '1y'
                    rf_val = float(parts[3]) if len(parts) > 3 else 0.0365
                    print(f'  [CAPM] {sym} vs {bench} ({period})')
                    self.send_json(calc_capm(sym, bench, period, rf_val))
                else:
                    self.send_json({'error': 'Usage: /api/capm/SYMBOL/BENCHMARK/PERIOD/RF'}, 400)
            else:
                self.send_json({'error': 'not found'}, 404)
        except Exception as e:
            self.send_json({'error': str(e)}, 500)

    def do_POST(self):
        p = self.path
        try:
            if p == '/api/optimize':
                body = self.read_body()
                symbols_data = body.get('symbols_data', [])
                rf = float(body.get('rf', 3.65)) / 100
                if len(symbols_data) < 2:
                    self.send_json({'error': 'Servono almeno 2 titoli'}, 400)
                    return
                print(f"  [Markowitz] Ottimizzazione esatta per {len(symbols_data)} titoli...")
                result = optimize_markowitz(symbols_data, rf)
                print(f"  [Markowitz] OK — Sharpe tangenza: {result['tangency_long']['sharpe']}")
                self.send_json(result)
            elif p == '/api/portfolio-nav':
                body = self.read_body()
                items = body.get('items', [])
                if len(items) < 2:
                    self.send_json({'error': 'Servono almeno 2 titoli'}, 400)
                    return
                rf        = float(body.get('rf', 3.65)) / 100
                period    = body.get('period', '3y')
                if period not in ('1y', '3y', '5y', '10y'):
                    period = '3y'
                benchmark = (body.get('benchmark') or '').strip() or None
                print(f"  [NAV] {len(items)} titoli, {period}, bench {benchmark}...")
                result = calc_portfolio_nav(items, rf, period, benchmark)
                if 'error' not in result:
                    print(f"  [NAV] OK — total {result['total_ret']}%, maxDD {result['max_dd']}%")
                self.send_json(result)
            elif p == '/api/backtest':
                body = self.read_body()
                symbols = body.get('symbols', [])
                if len(symbols) < 2:
                    self.send_json({'error': 'Servono almeno 2 titoli'}, 400)
                    return
                rf        = float(body.get('rf', 3.65)) / 100
                hist      = body.get('hist', '5y')
                if hist not in ('5y', '10y'):
                    hist = '5y'
                lookback  = {'1y': 252, '2y': 504}.get(body.get('lookback', '1y'), 252)
                rebal     = {'1m': 21, '3m': 63, '6m': 126}.get(body.get('rebal', '3m'), 63)
                benchmark = (body.get('benchmark') or '').strip() or None
                print(f"  [Backtest] {len(symbols)} titoli, storico {hist}, lookback {lookback}g, ribil. {rebal}g, bench {benchmark}...")
                result = calc_backtest(symbols, rf, hist, lookback, rebal, benchmark)
                if 'error' not in result:
                    print(f"  [Backtest] OK — Sharpe: tangenza {result['metrics']['tangency']['sharpe']} vs equal {result['metrics']['equal']['sharpe']}")
                self.send_json(result)
            elif p == '/api/montecarlo':
                body = self.read_body()
                symbols_data = body.get('symbols_data', [])
                if len(symbols_data) < 2:
                    self.send_json({'error': 'Servono almeno 2 titoli'}, 400)
                    return
                rf      = float(body.get('rf', 3.65)) / 100
                years   = max(1, min(10, int(body.get('years', 5))))
                capital = max(1.0, float(body.get('capital', 10000)))
                target  = float(body.get('target', 0)) or None
                method  = body.get('method', 'boot')
                wmode   = body.get('wmode', 'tangency')
                hist    = body.get('hist', '5y')
                if hist not in ('1y', '3y', '5y', '10y'):
                    hist = '5y'
                print(f"  [MonteCarlo] {len(symbols_data)} titoli, {years}a, {method}/{wmode}, storico {hist}...")
                symbols_data = fetch_closes_for_mc(symbols_data, hist)
                result = calc_montecarlo(symbols_data, rf, years, capital, target, method, wmode)
                print(f"  [MonteCarlo] OK — mediana: {result['median']}, prob perdita: {result['prob_loss']}%")
                self.send_json(result)
            elif p == '/api/diversification':
                body = self.read_body()
                symbols_data = body.get('symbols_data', [])
                rf = float(body.get('rf', 3.65)) / 100
                if len(symbols_data) < 2:
                    self.send_json({'error': 'Servono almeno 2 titoli'}, 400)
                    return
                print(f"  [Diversificazione] Analisi marginale per {len(symbols_data)} titoli...")
                result = calc_diversification(symbols_data, rf)
                print(f"  [Diversificazione] OK — DR: {result['portfolio']['diversification_ratio']}")
                self.send_json(result)
            else:
                self.send_json({'error': 'not found'}, 404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = http.server.ThreadingHTTPServer(('localhost', PORT), Handler)
    print()
    print('  ╔══════════════════════════════════════════╗')
    print('  ║   Portfolio Optimizer — Server avviato   ║')
    print(f'  ║   http://localhost:{PORT}                    ║')
    print('  ╠══════════════════════════════════════════╣')
    print('  ║  1. Lascia aperta questa finestra        ║')
    print('  ║  2. Apri portfolio-markowitz.html        ║')
    print('  ║  Per fermare: Ctrl+C                     ║')
    print('  ╚══════════════════════════════════════════╝')
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  Server fermato.')
        sys.exit(0)

if __name__ == '__main__':
    main()
