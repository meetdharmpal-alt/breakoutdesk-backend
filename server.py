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
import threading
import time

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
    "trade_journal": [],
    "trade_logs": []
}

def check_fundamentals(ticker_symbol, stock_obj=None):
    try:
        stock = stock_obj if stock_obj else yf.Ticker(ticker_symbol, session=session)
        info = stock.info or {}
        roe = info.get('returnOnEquity')
        roce = info.get('returnOnCapitalEmployed')
        
        if roe is not None and roe < 0.12:
            return False
        if roce is not None and roce < 0.12:
            return False
        return True
    except Exception:
        return True

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

def identify_chart_pattern(df):
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    dates = [idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)[:16] for idx in df.index]
    
    current_price = closes[-1]
    recent_lows_idx = argrelextrema(lows[-40:], np.less, order=2)[0]
    recent_maxs_idx = argrelextrema(highs[-40:], np.greater, order=2)[0]
    offset = len(df) - 40 if len(df) >= 40 else 0

    # 1. Inverse Head & Shoulders
    if len(recent_lows_idx) >= 3 and len(recent_maxs_idx) >= 2:
        l1, l2, l3 = recent_lows_idx[-3], recent_lows_idx[-2], recent_lows_idx[-1]
        troughs = lows[-40:][[l1, l2, l3]]
        right_shoulder = troughs[2]
        if troughs[1] < troughs[0] and troughs[1] < troughs[2]:
            necklace_level = max(highs[-40:][recent_maxs_idx[-2:]])
            entry = round(float(necklace_level), 2)
            sl = round(float(right_shoulder * 0.995), 2)
            target = round(float(entry + (entry - troughs[1])), 2)
            upper_trend = [{"time": dates[offset + l1], "value": entry}, {"time": dates[-1], "value": entry}]
            lower_trend = [{"time": dates[offset + l1], "value": round(float(troughs[0]), 2)}, {"time": dates[offset + l2], "value": round(float(troughs[1]), 2)}, {"time": dates[offset + l3], "value": round(float(troughs[2]), 2)}]
            return "Inverse H&S", entry, target, sl, upper_trend, lower_trend

    # 2. Double Bottom
    if len(recent_lows_idx) >= 2 and len(recent_maxs_idx) >= 1:
        l1, l2 = recent_lows_idx[-2], recent_lows_idx[-1]
        b1, b2 = lows[-40:][l1], lows[-40:][l2]
        if abs(b1 - b2) / b1 < 0.03:
            necklace_level = max(highs[-40:][l1:l2+1])
            entry = round(float(necklace_level), 2)
            lowest_trough = min(b1, b2)
            sl = round(float(lowest_trough * 0.995), 2)
            target = round(float(entry + (entry - lowest_trough)), 2)
            upper_trend = [{"time": dates[offset + l1], "value": entry}, {"time": dates[-1], "value": entry}]
            lower_trend = [{"time": dates[offset + l1], "value": round(float(b1), 2)}, {"time": dates[offset + l2], "value": round(float(b2), 2)}]
            return "Double Bottom", entry, target, sl, upper_trend, lower_trend

    # 3. Cup and Handle
    if len(recent_lows_idx) >= 2 and len(recent_maxs_idx) >= 2:
        cup_bottom = np.min(lows[-30:-8])
        rim_high = np.max(highs[-30:-8])
        handle_low = np.min(lows[-8:])
        cup_depth = rim_high - cup_bottom
        if handle_low > (rim_high - 0.5 * cup_depth):
            entry = round(float(rim_high), 2)
            sl = round(float(handle_low * 0.995), 2)
            target = round(float(entry + cup_depth), 2)
            upper_trend = [{"time": dates[-30], "value": entry}, {"time": dates[-1], "value": entry}]
            lower_trend = [{"time": dates[-20], "value": round(float(cup_bottom), 2)}, {"time": dates[-4], "value": round(float(handle_low), 2)}]
            return "Cup and Handle", entry, target, sl, upper_trend, lower_trend

    # 4. High Tight Flag
    price_20_bars_ago = closes[-20] if len(closes) >= 20 else closes[0]
    flagpole_height = np.max(highs[-10:]) - price_20_bars_ago
    recent_range = (np.max(highs[-6:]) - np.min(lows[-6:])) / current_price
    if flagpole_height > 0 and recent_range <= 0.06:
        flag_low = np.min(lows[-6:])
        entry = round(float(np.max(highs[-6:])), 2)
        sl = round(float(flag_low * 0.995), 2)
        target = round(float(entry + flagpole_height), 2)
        upper_trend = [{"time": dates[-6], "value": entry}, {"time": dates[-1], "value": entry}]
        lower_trend = [{"time": dates[-6], "value": round(float(flag_low), 2)}, {"time": dates[-1], "value": round(float(flag_low), 2)}]
        return "High Tight Flag", entry, target, sl, upper_trend, lower_trend

    # 5. Ascending Triangle
    if len(recent_maxs_idx) >= 2 and len(recent_lows_idx) >= 2:
        i_high1, i_high2 = recent_maxs_idx[-2], recent_maxs_idx[-1]
        recent_high_vals = highs[-40:][[i_high1, i_high2]]
        if abs(recent_high_vals[0] - recent_high_vals[1]) / recent_high_vals[0] < 0.02:
            resistance_level = recent_high_vals[1]
            swing_low_inside = lows[-40:][recent_lows_idx[-1]]
            base_width = resistance_level - lows[-40:][recent_lows_idx[-2]]
            entry = round(float(resistance_level), 2)
            sl = round(float(swing_low_inside * 0.995), 2)
            target = round(float(entry + base_width), 2)
            upper_trend = [{"time": dates[offset + i_high1], "value": entry}, {"time": dates[-1], "value": entry}]
            lower_trend = [{"time": dates[offset + recent_lows_idx[-2]], "value": round(float(lows[-40:][recent_lows_idx[-2]]), 2)}, {"time": dates[-1], "value": round(float(swing_low_inside), 2)}]
            return "Ascending Triangle", entry, target, sl, upper_trend, lower_trend

    entry = round(current_price * 1.002, 2)
    sl = round(current_price * 0.99, 2)
    target = round(entry + 2.1 * (entry - sl), 2)
    upper_trend = [{"time": dates[-15], "value": round(float(np.max(highs[-15:])), 2)}, {"time": dates[-1], "value": round(float(np.max(highs[-15:])), 2)}]
    lower_trend = [{"time": dates[-15], "value": round(float(np.min(lows[-15:])), 2)}, {"time": dates[-1], "value": round(float(np.min(lows[-15:])), 2)}]
    return "Rectangle Breakout", entry, target, sl, upper_trend, lower_trend

def execute_algo_buy(ticker, pattern, entry, target, sl, current_price):
    existing = [p for p in TRADING_CONFIG["active_positions"] if p["ticker"] == ticker]
    if existing:
        return

    available_funds = TRADING_CONFIG["paper_funds"]
    allocation = available_funds * TRADING_CONFIG["position_size_pct"]
    if allocation < entry:
        return

    quantity = int(allocation // entry)
    if quantity <= 0:
        return

    invested_amount = quantity * entry
    TRADING_CONFIG["paper_funds"] -= invested_amount

    new_position = {
        "ticker": ticker,
        "pattern": pattern,
        "entry_price": entry,
        "current_price": current_price,
        "target": target,
        "sl": sl,
        "quantity": quantity,
        "invested": round(invested_amount, 2),
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    TRADING_CONFIG["active_positions"].append(new_position)
    msg = f"[ALGO LIVE EXECUTION] BUY {quantity} shares of {ticker} @ Entry Rs. {entry} | Target: Rs. {target} | SL: Rs. {sl}"
    TRADING_CONFIG["trade_logs"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": msg, "ticker": ticker, "type": "EXECUTION"})
    print(msg)

def monitor_and_manage_positions():
    while True:
        try:
            if TRADING_CONFIG["algo_trading_enabled"] and TRADING_CONFIG["active_positions"]:
                for pos in list(TRADING_CONFIG["active_positions"]):
                    t = yf.Ticker(pos["ticker"], session=session)
                    todays_data = t.history(period="1d")
                    if todays_data is not None and not todays_data.empty:
                        ltp = float(todays_data['Close'].iloc[-1])
                        pos["current_price"] = ltp

                        if ltp >= pos["target"]:
                            exit_val = pos["quantity"] * pos["target"]
                            pnl = exit_val - pos["invested"]
                            pnl_pct = round((pnl / pos["invested"]) * 100, 2)
                            TRADING_CONFIG["paper_funds"] += exit_val

                            journal_entry = {
                                "ticker": pos["ticker"],
                                "pattern": pos["pattern"],
                                "entry_price": pos["entry_price"],
                                "exit_price": pos["target"],
                                "quantity": pos["quantity"],
                                "invested": pos["invested"],
                                "exit_value": round(exit_val, 2),
                                "pnl": round(pnl, 2),
                                "pnl_pct": pnl_pct,
                                "exit_reason": "TARGET HIT 🎯",
                                "entry_time": pos["entry_time"],
                                "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            TRADING_CONFIG["trade_journal"].insert(0, journal_entry)
                            TRADING_CONFIG["active_positions"].remove(pos)
                            
                            log_msg = f"[ALGO EXIT] TARGET HIT for {pos['ticker']} @ Rs. {pos['target']} | Net PnL: Rs. {round(pnl, 2)} ({pnl_pct}%)"
                            TRADING_CONFIG["trade_logs"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": log_msg, "ticker": pos["ticker"], "type": "EXIT"})

                        elif ltp <= pos["sl"]:
                            exit_val = pos["quantity"] * pos["sl"]
                            pnl = exit_val - pos["invested"]
                            pnl_pct = round((pnl / pos["invested"]) * 100, 2)
                            TRADING_CONFIG["paper_funds"] += exit_val

                            journal_entry = {
                                "ticker": pos["ticker"],
                                "pattern": pos["pattern"],
                                "entry_price": pos["entry_price"],
                                "exit_price": pos["sl"],
                                "quantity": pos["quantity"],
                                "invested": pos["invested"],
                                "exit_value": round(exit_val, 2),
                                "pnl": round(pnl, 2),
                                "pnl_pct": pnl_pct,
                                "exit_reason": "STOP LOSS HIT 🛑",
                                "entry_time": pos["entry_time"],
                                "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            TRADING_CONFIG["trade_journal"].insert(0, journal_entry)
                            TRADING_CONFIG["active_positions"].remove(pos)

                            log_msg = f"[ALGO EXIT] STOP LOSS HIT for {pos['ticker']} @ Rs. {pos['sl']} | Net PnL: Rs. {round(pnl, 2)} ({pnl_pct}%)"
                            TRADING_CONFIG["trade_logs"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": log_msg, "ticker": pos["ticker"], "type": "EXIT"})
        except Exception as e:
            print(f"Background monitor error: {e}")
        time.sleep(30)

threading.Thread(target=monitor_and_manage_positions, daemon=True).start()

# --- SWING SCANNER (Daily timeframe) ---
def fetch_and_scan_stock(ticker):
    try:
        t = yf.Ticker(ticker, session=session)
        if not check_fundamentals(ticker, stock_obj=t):
            return None

        df = t.history(period="3mo", interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 30:
            return None
            
        vol_sma = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        if current_vol <= (vol_sma * 1.49):
            return None

        if not evaluate_custom_strategy(df, TRADING_CONFIG["selected_strategy"]):
            return None

        pattern, entry, target, sl, upper_trend, lower_trend = identify_chart_pattern(df)
        
        risk = entry - sl
        reward = target - entry
        if risk <= 0 or (reward / risk) < 2.0:
            return None

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        price = float(last_row['Close'])
        prev_close = float(prev_row['Close'])
        chg = round(((price - prev_close) / prev_close) * 100, 2)

        if price >= entry and TRADING_CONFIG["algo_trading_enabled"]:
            execute_algo_buy(ticker, pattern, entry, target, sl, price)

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

        rr_ratio = round(reward / risk, 2)
        return {
            "ticker": ticker,
            "name": ticker.replace(".NS", ""),
            "price": round(price, 2),
            "chg": chg,
            "pattern": pattern,
            "entry": entry,
            "target": target,
            "sl": sl,
            "rr_ratio": f"1:{rr_ratio}",
            "volume": f"{round(current_vol/vol_sma, 2)}x ⚡",
            "ohlc": ohlc,
            "upper_trend": upper_trend,
            "lower_trend": lower_trend
        }
    except Exception:
        return None

# --- INTRADAY SCANNER (15-Minute timeframe) ---
def fetch_and_scan_intraday(ticker):
    try:
        t = yf.Ticker(ticker, session=session)
        df = t.history(period="5d", interval="15m", auto_adjust=True)
        if df is None or df.empty or len(df) < 30:
            return None
            
        vol_sma = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        if current_vol <= (vol_sma * 1.30):
            return None

        pattern, entry, target, sl, upper_trend, lower_trend = identify_chart_pattern(df)
        
        risk = entry - sl
        reward = target - entry
        if risk <= 0 or (reward / risk) < 2.0:
            return None

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        price = float(last_row['Close'])
        prev_close = float(prev_row['Close'])
        chg = round(((price - prev_close) / prev_close) * 100, 2)

        ohlc = []
        for idx, row in df.iloc[-60:].iterrows():
            date_str = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)[:16]
            ohlc.append({
                "time": date_str,
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2)
            })

        rr_ratio = round(reward / risk, 2)
        return {
            "ticker": ticker,
            "name": ticker.replace(".NS", ""),
            "price": round(price, 2),
            "chg": chg,
            "pattern": pattern,
            "entry": entry,
            "target": target,
            "sl": sl,
            "rr_ratio": f"1:{rr_ratio}",
            "volume": f"{round(current_vol/vol_sma, 2)}x ⚡",
            "ohlc": ohlc,
            "upper_trend": upper_trend,
            "lower_trend": lower_trend
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
        "trade_journal": TRADING_CONFIG["trade_journal"],
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

@app.get("/api/scan-intraday")
def scan_intraday():
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_and_scan_intraday, ticker): ticker for ticker in WATCHLIST}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    return {"status": "success", "count": len(results), "data": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
