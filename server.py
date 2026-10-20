import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy.signal import argrelextrema
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

app = FastAPI(title="BreakoutDesk Pattern Screener & Algo Terminal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
})

WATCHLIST = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 
    'ITC.NS', 'BHARTIARTL.NS', 'WIPRO.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 
    'SUNPHARMA.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS', 
    'HCLTECH.NS', 'TATASTEEL.NS', 'POWERGRID.NS', 'NTPC.NS', 'ADANIENT.NS', 
    'JSWSTEEL.NS', 'TATAMOTORS.NS', 'POLYCAB.NS', 'PERSISTENT.NS', 'COFORGE.NS',
    'ULTRACEMCO.NS', 'BAJAJFINSV.NS', 'NESTLEIND.NS', 'ONGC.NS', 'COALINDIA.NS',
    'GRASIM.NS', 'TECHM.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'HEROMOTOCO.NS',
    'EICHERMOT.NS', 'DIVISLAB.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
    'APOLLOHOSP.NS', 'TATACONSUM.NS', 'BEL.NS', 'TRENT.NS', 'HAL.NS', 'VBL.NS',
    'ZOMATO.NS', 'JIOFIN.NS', 'BSE.NS', 'IREDA.NS', 'NHPC.NS', 'REC.NS', 'PFC.NS',
    'IRFC.NS', 'RVNL.NS', 'SUZLON.NS', 'TATAPOWER.NS', 'MOTHERSON.NS', 'GAIL.NS',
    'INDUSTOWER.NS', 'INDIANB.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'BANDHANBNK.NS',
    'SAIL.NS', 'NMDC.NS', 'NATIONALUM.NS', 'HINDALCO.NS', 'JINDALSTEL.NS',
    'TATACOMM.NS', 'LTIM.NS', 'MPHASIS.NS', 'DIXON.NS', 'KALYANKJIL.NS',
    'MAXHEALTH.NS', 'MUTHOOTFIN.NS', 'CHOLAFIN.NS', 'OBEROIRLTY.NS', 'DLF.NS',
    'GODREJPROP.NS', 'PHOENIXLTD.NS', 'OFSS.NS', 'CYIENT.NS', 'KPITTECH.NS',
    'TATAELXSI.NS', 'ASTRAL.NS', 'SUPREMEIND.NS', 'PIIND.NS', 'SRF.NS',
    'DEEPAKNTR.NS', 'PIDILITIND.NS', 'BERGEPAINT.NS', 'COLPAL.NS', 'DABUR.NS',
    'GODREJCP.NS', 'MARICO.NS', 'TORNTPHARM.NS', 'LUPIN.NS', 'MANKIND.NS',
    'ALKEM.NS', 'GLENMARK.NS', 'AUROPHARMA.NS', 'ZYDUSLIFE.NS', 'IPCALAB.NS',
    'BOSCHLTD.NS', 'MRF.NS', 'BALKRISIND.NS', 'TIINDIA.NS', 'CUMMINSIND.NS',
    'ABB.NS', 'SIEMENS.NS', 'CGPOWER.NS', 'SOLARINDS.NS', 'KEI.NS', 'AIAENG.NS'
]

# --- GLOBAL TERMINAL STATE & ALGO CONFIGURATION ---
TRADING_CONFIG = {
    "paper_trading_enabled": True,
    "algo_trading_enabled": False,
    "selected_strategy": "Breakout Momentum v1",
    "paper_funds": 1000000.0,
    "initial_funds": 1000000.0,
    "position_size_pct": 0.10,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "notification_email": "",
    "active_positions": [],
    "trade_logs": []
}

# --- FUNDAMENTAL QUALITY CHECK ENGINE ---
def check_fundamentals(ticker_symbol, stock_obj=None):
    try:
        stock = stock_obj if stock_obj else yf.Ticker(ticker_symbol, session=session)
        info = stock.info or {}
        
        roe = info.get('returnOnEquity')
        roce = info.get('returnOnCapitalEmployed')
        
        if roe is not None and roe < 0.15:
            return False
        if roce is not None and roce < 0.15:
            return False

        financials = stock.financials
        if financials is not None and not financials.empty:
            rev_rows = [r for r in financials.index if 'Total Revenue' in str(r) or 'Revenue' in str(r)]
            if rev_rows:
                revs = financials.loc[rev_rows[0]].dropna().sort_index()
                if len(revs) >= 4:
                    start_rev, end_rev = revs.iloc[-4], revs.iloc[-1]
                    if start_rev > 0:
                        sales_cagr = (end_rev / start_rev) ** (1 / 3) - 1
                        if sales_cagr < 0.15:
                            return False

            net_rows = [r for r in financials.index if 'Net Income' in str(r)]
            if net_rows:
                profits = financials.loc[net_rows[0]].dropna().sort_index()
                if len(profits) >= 4:
                    start_prof, end_prof = profits.iloc[-4], profits.iloc[-1]
                    if start_prof > 0:
                        profit_cagr = (end_prof / start_prof) ** (1 / 3) - 1
                        if profit_cagr < 0.15:
                            return False

        return True
    except Exception:
        return True

# --- FUTURE STRATEGY ENGINE ---
def evaluate_custom_strategy(df, strategy_name):
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values

    if strategy_name == "VWAP Breakout":
        tp = (highs + lows + closes) / 3
        vol = df['Volume'].values
        vwap = np.sum(tp[-20:] * vol[-20:]) / np.sum(vol[-20:])
        return closes[-1] > vwap
        
    elif strategy_name == "EMA Crossover":
        ema20 = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        return ema20 > ema50

    elif strategy_name == "Open High Low Reversal":
        open_p, low_p, close_p = df['Open'].iloc[-1], df['Low'].iloc[-1], df['Close'].iloc[-1]
        return (open_p == low_p) and (close_p > open_p)

    return True

# --- NOTIFICATION HOOK ---
def send_entry_notification(ticker, pattern, entry_price, target, sl):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"🚨 [ENTRY TRIGGERED] {ticker} | Pattern: {pattern} | Entry: ₹{entry_price} | Target: ₹{target} | SL: ₹{sl} ({timestamp})"
    
    TRADING_CONFIG["trade_logs"].append({"timestamp": timestamp, "message": msg, "ticker": ticker, "type": "NOTIFICATION"})
    print(msg)

    bot_token = TRADING_CONFIG.get("telegram_bot_token")
    chat_id = TRADING_CONFIG.get("telegram_chat_id")
    if bot_token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=5)
        except Exception as e:
            print(f"Telegram notification dispatch error: {e}")

# --- AUTOMATED ALGO & PAPER EXECUTION MODULE ---
def process_paper_and_algo_trade(ticker, pattern, entry_price, target, sl, current_price):
    if not TRADING_CONFIG["algo_trading_enabled"] or not TRADING_CONFIG["paper_trading_enabled"]:
        return

    existing = [p for p in TRADING_CONFIG["active_positions"] if p["ticker"] == ticker]
    if existing:
        return

    available_funds = TRADING_CONFIG["paper_funds"]
    allocation = available_funds * TRADING_CONFIG["position_size_pct"]
    
    if allocation < entry_price:
        return

    quantity = int(allocation // entry_price)
    if quantity <= 0:
        return

    invested_amount = quantity * entry_price
    TRADING_CONFIG["paper_funds"] -= invested_amount

    new_position = {
        "ticker": ticker,
        "pattern": pattern,
        "entry_price": entry_price,
        "current_price": current_price,
        "target": target,
        "sl": sl,
        "quantity": quantity,
        "invested": round(invested_amount, 2),
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    TRADING_CONFIG["active_positions"].append(new_position)
    
    log_msg = f"🤖 [ALGO PAPER BUY] Executed {quantity} shares of {ticker} at ₹{entry_price} (Invested: ₹{round(invested_amount, 2)})"
    TRADING_CONFIG["trade_logs"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": log_msg,
        "ticker": ticker,
        "type": "EXECUTION"
    })

def identify_chart_pattern(df):
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values
    dates = [idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10] for idx in df.index]
    
    current_price = closes[-1]
    vol_sma = np.mean(volumes[-20:])
    is_volume_surge = volumes[-1] >= (vol_sma * 1.15)
    high_52w = np.max(highs[-252:]) if len(highs) >= 252 else np.max(highs)
    is_near_high = current_price >= (high_52w * 0.95)

    recent_lows_idx = argrelextrema(lows[-60:], np.less, order=2)[0]
    recent_maxs_idx = argrelextrema(highs[-60:], np.greater, order=2)[0]
    offset = len(df) - 60 if len(df) >= 60 else 0

    # 1. High Tight Flag
    price_40_bars_ago = closes[-40] if len(closes) >= 40 else closes[0]
    gain_pct = (current_price - price_40_bars_ago) / price_40_bars_ago
    recent_range = (np.max(highs[-10:]) - np.min(lows[-10:])) / current_price
    if gain_pct >= 0.30 and recent_range <= 0.09:
        trend_pts = [
            {"time": dates[-10], "value": round(float(highs[-10]), 2)},
            {"time": dates[-1], "value": round(float(lows[-1]), 2)}
        ]
        return "High Tight Flag", round(current_price * 1.02, 2), round(current_price * 1.08, 2), round(current_price * 0.95, 2), trend_pts

    # 2. Inverse Head & Shoulders
    if len(recent_lows_idx) >= 3:
        i1, i2, i3 = recent_lows_idx[-3], recent_lows_idx[-2], recent_lows_idx[-1]
        troughs = lows[-60:][[i1, i2, i3]]
        if troughs[1] < troughs[0] and troughs[1] < troughs[2]:
            trend_pts = [
                {"time": dates[offset + i1], "value": round(float(troughs[0]), 2)},
                {"time": dates[offset + i2], "value": round(float(troughs[1]), 2)},
                {"time": dates[offset + i3], "value": round(float(troughs[2]), 2)}
            ]
            return "Inverse H&S", round(current_price * 1.005, 2), round(current_price * 1.07, 2), round(troughs[1] * 0.97, 2), trend_pts

    # 3. Double Bottom
    if len(recent_lows_idx) >= 2:
        i1, i2 = recent_lows_idx[-2], recent_lows_idx[-1]
        b1, b2 = lows[-60:][i1], lows[-60:][i2]
        if abs(b1 - b2) / b1 < 0.04:
            trend_pts = [
                {"time": dates[offset + i1], "value": round(float(b1), 2)},
                {"time": dates[offset + i2], "value": round(float(b2), 2)}
            ]
            return "Double Bottom", round(current_price * 1.005, 2), round(current_price * 1.06, 2), round(min(b1, b2) * 0.97, 2), trend_pts

    # 4. Ascending Triangle
    if len(recent_maxs_idx) >= 2 and len(recent_lows_idx) >= 2:
        i_high1, i_high2 = recent_maxs_idx[-2], recent_maxs_idx[-1]
        recent_high_vals = highs[-60:][[i_high1, i_high2]]
        if abs(recent_high_vals[0] - recent_high_vals[1]) / recent_high_vals[0] < 0.03:
            resistance_level = recent_high_vals[1]
            trend_pts = [
                {"time": dates[offset + i_high1], "value": round(float(resistance_level), 2)},
                {"time": dates[-1], "value": round(float(resistance_level), 2)}
            ]
            return "Ascending Triangle", round(resistance_level * 1.002, 2), round(resistance_level * 1.07, 2), round(lows[-60:][recent_lows_idx[-1]] * 0.97, 2), trend_pts

    # 5. Rectangle Breakout
    if is_near_high:
        trend_pts = [
            {"time": dates[-30], "value": round(float(high_52w), 2)},
            {"time": dates[-1], "value": round(float(high_52w), 2)}
        ]
        return "Rectangle Breakout", round(current_price * 1.005, 2), round(current_price * 1.06, 2), round(current_price * 0.97, 2), trend_pts

    return "Consolidation", round(current_price * 1.01, 2), round(current_price * 1.04, 2), round(current_price * 0.97, 2), []

def fetch_and_scan_stock(ticker):
    try:
        t = yf.Ticker(ticker, session=session)
        
        if not check_fundamentals(ticker, stock_obj=t):
            return None

        df = t.history(period="3mo", auto_adjust=True)
        if df is None or df.empty or len(df) < 30:
            return None
            
        if not evaluate_custom_strategy(df, TRADING_CONFIG["selected_strategy"]):
            return None

        pattern, entry, target, sl, trend_pts = identify_chart_pattern(df)
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        price = float(last_row['Close'])
        prev_close = float(prev_row['Close'])
        chg = round(((price - prev_close) / prev_close) * 100, 2)
        
        vol_sma = df['Volume'].rolling(window=20).mean().iloc[-1]
        is_vol = df['Volume'].iloc[-1] >= (vol_sma * 1.15)

        if price >= entry:
            send_entry_notification(ticker, pattern, entry, target, sl)
            process_paper_and_algo_trade(ticker, pattern, entry, target, sl, price)

        ohlc = []
        for idx, row in df.iloc[-50:].iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
            ohlc.append({
                "time": date_str,
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2)
            })

        return {
            "ticker": ticker,
            "name": ticker.replace(".NS", ""),
            "price": round(price, 2),
            "chg": chg,
            "pattern": pattern,
            "entry": entry,
            "target": target,
            "sl": sl,
            "volume": "2.1x ⚡" if is_vol else "1.0x",
            "ohlc": ohlc,
            "trend_pts": trend_pts
        }
    except Exception:
        return None

@app.get("/")
def home():
    return {"status": "online", "message": "BreakoutDesk Pattern & Algo Screener is running!"}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/config")
def get_config():
    return {"status": "success", "config": TRADING_CONFIG}

@app.post("/api/config/update")
def update_config(settings: dict):
    global TRADING_CONFIG
    for key, val in settings.items():
        if key in TRADING_CONFIG:
            TRADING_CONFIG[key] = val
    return {"status": "success", "config": TRADING_CONFIG}

@app.get("/api/algo/dashboard")
def get_algo_dashboard():
    active_positions = TRADING_CONFIG["active_positions"]
    invested_capital = sum([p["invested"] for p in active_positions])
    current_value = sum([p["quantity"] * p["current_price"] for p in active_positions])
    unrealized_pnl = current_value - invested_capital

    return {
        "status": "success",
        "paper_trading_enabled": TRADING_CONFIG["paper_trading_enabled"],
        "algo_trading_enabled": TRADING_CONFIG["algo_trading_enabled"],
        "selected_strategy": TRADING_CONFIG["selected_strategy"],
        "available_paper_funds": round(TRADING_CONFIG["paper_funds"], 2),
        "invested_capital": round(invested_capital, 2),
        "current_portfolio_value": round(current_value, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "active_positions_count": len(active_positions),
        "active_positions": active_positions,
        "trade_logs": TRADING_CONFIG["trade_logs"][-50:]
    }

@app.get("/api/scan-market")
def scan_market():
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_and_scan_stock, ticker): ticker for ticker in WATCHLIST}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    return {"status": "success", "count": len(results), "data": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
