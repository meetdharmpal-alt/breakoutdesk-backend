import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy.signal import argrelextrema
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading
import time

# Diagnostic Kite import check
KITE_IMPORT_ERROR = None
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except Exception as e:
    KITE_AVAILABLE = False
    KITE_IMPORT_ERROR = str(e)
    print(f"CRITICAL KITE IMPORT ERROR: {e}")

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
    "kite_api_key": "",
    "kite_api_secret": "",
    "kite_access_token": "",
    "kite_connected": False,
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

def identify_chart_pattern(df, is_intraday=False):
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)
    
    time_fmt = '%Y-%m-%d %H:%M' if is_intraday else '%Y-%m-%d'
    dates = [idx.strftime(time_fmt) if hasattr(idx, 'strftime') else str(idx)[:16] for idx in df.index]
    current_price = closes[-1]

    lookback = min(50, n)
    sub_highs = highs[-lookback:]
    sub_lows = lows[-lookback:]
    
    maxs_idx = argrelextrema(sub_highs, np.greater, order=2)[0]
    lows_idx = argrelextrema(sub_lows, np.less, order=2)[0]
    offset = n - lookback

    # 1. Ascending Triangle
    if len(maxs_idx) >= 2 and len(lows_idx) >= 2:
        m1, m2 = maxs_idx[-2], maxs_idx[-1]
        res_vals = sub_highs[[m1, m2]]
        if abs(res_vals[0] - res_vals[1]) / res_vals[0] < 0.03:
            resistance = float(res_vals[1])
            entry = round(resistance, 2)
            swing_low_1 = float(sub_lows[lows_idx[-2]])
            swing_low_2 = float(sub_lows[lows_idx[-1]])
            sl = round(float(swing_low_2 * 0.99), 2)
            target = round(float(entry + (resistance - swing_low_1)), 2)
            
            upper_trend = [{"time": dates[offset + m1], "value": resistance}, {"time": dates[-1], "value": resistance}]
            lower_trend = [{"time": dates[offset + lows_idx[-2]], "value": swing_low_1}, {"time": dates[offset + lows_idx[-1]], "value": swing_low_2}]
            return "Ascending Triangle", entry, target, sl, upper_trend, lower_trend

    # 2. Inverse Head & Shoulders
    if len(lows_idx) >= 3 and len(maxs_idx) >= 2:
        l1, l2, l3 = lows_idx[-3], lows_idx[-2], lows_idx[-1]
        troughs = sub_lows[[l1, l2, l3]]
        if troughs[1] < troughs[0] and troughs[1] < troughs[2]:
            neck = float(max(sub_highs[maxs_idx[-2:]]))
            entry = round(neck, 2)
            sl = round(float(troughs[2] * 0.99), 2)
            target = round(float(entry + (entry - troughs[1])), 2)
            
            upper_trend = [{"time": dates[offset + maxs_idx[-2]], "value": neck}, {"time": dates[-1], "value": neck}]
            lower_trend = [
                {"time": dates[offset + l1], "value": float(troughs[0])},
                {"time": dates[offset + l2], "value": float(troughs[1])},
                {"time": dates[offset + l3], "value": float(troughs[2])}
            ]
            return "Inverse H&S", entry, target, sl, upper_trend, lower_trend

    # 3. Double Bottom
    if len(lows_idx) >= 2:
        l1, l2 = lows_idx[-2], lows_idx[-1]
        b1, b2 = sub_lows[l1], sub_lows[l2]
        if abs(b1 - b2) / b1 < 0.04:
            peak_val = float(max(sub_highs[l1:l2+1])) if l2 > l1 else float(sub_highs[-1])
            entry = round(peak_val, 2)
            sl = round(float(min(b1, b2) * 0.99), 2)
            target = round(float(entry + (entry - min(b1, b2))), 2)
            
            upper_trend = [{"time": dates[offset + l1], "value": entry}, {"time": dates[-1], "value": entry}]
            lower_trend = [{"time": dates[offset + l1], "value": float(b1)}, {"time": dates[offset + l2], "value": float(b2)}]
            return "Double Bottom", entry, target, sl, upper_trend, lower_trend

    # 4. Cup and Handle
    if n >= 35:
        cup_bottom = float(np.min(lows[-35:-10]))
        rim_high = float(np.max(highs[-35:-10]))
        handle_low = float(np.min(lows[-10:]))
        cup_depth = rim_high - cup_bottom
        if handle_low > (rim_high - 0.6 * cup_depth):
            entry = round(rim_high, 2)
            sl = round(float(handle_low * 0.99), 2)
            target = round(float(entry + cup_depth), 2)
            
            upper_trend = [{"time": dates[-35], "value": rim_high}, {"time": dates[-1], "value": rim_high}]
            lower_trend = [{"time": dates[-20], "value": cup_bottom}, {"time": dates[-5], "value": handle_low}]
            return "Cup and Handle", entry, target, sl, upper_trend, lower_trend

    # 5. High Tight Flag
    flag_len = min(10, n)
    flag_high = float(np.max(highs[-flag_len:]))
    flag_low = float(np.min(lows[-flag_len:]))
    if (flag_high - flag_low) / current_price <= 0.08:
        entry = round(flag_high, 2)
        sl = round(float(flag_low * 0.99), 2)
        flagpole = flag_high - closes[-min(25, n)]
        target = round(float(entry + max(flagpole, entry - sl)), 2)
        
        upper_trend = [{"time": dates[-flag_len], "value": flag_high}, {"time": dates[-1], "value": flag_high}]
        lower_trend = [{"time": dates[-flag_len], "value": flag_low}, {"time": dates[-1], "value": flag_low}]
        return "High Tight Flag", entry, target, sl, upper_trend, lower_trend

    # Default Channel Breakout
    entry = round(float(current_price * 1.005), 2)
    sl = round(float(current_price * 0.98), 2)
    target = round(float(entry + 2.2 * (entry - sl)), 2)
    upper_trend = [{"time": dates[-15], "value": float(np.max(highs[-15:]))}, {"time": dates[-1], "value": float(np.max(highs[-15:]))}]
    lower_trend = [{"time": dates[-15], "value": float(np.min(lows[-15:]))}, {"time": dates[-1], "value": float(np.min(lows[-15:]))}]
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

    if KITE_AVAILABLE and TRADING_CONFIG["kite_connected"] and TRADING_CONFIG["kite_access_token"] and TRADING_CONFIG["kite_api_key"]:
        try:
            kite = KiteConnect(api_key=TRADING_CONFIG["kite_api_key"])
            kite.set_access_token(TRADING_CONFIG["kite_access_token"])
            clean_ticker = ticker.replace(".NS", "")
            kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=clean_ticker,
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                product=kite.PRODUCT_CNC,
                order_type=kite.ORDER_TYPE_MARKET
            )
            TRADING_CONFIG["trade_logs"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": f"[KITE LIVE ORDER] Placed MARKET BUY for {quantity}x {clean_ticker}", "ticker": ticker, "type": "KITE"})
        except Exception as e:
            TRADING_CONFIG["trade_logs"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": f"[KITE ERROR] Failed to place order: {str(e)}", "ticker": ticker, "type": "ERROR"})

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
    msg = f"[ALGO EXECUTION] BUY {quantity} shares of {ticker} @ Entry Rs. {entry} | Target: Rs. {target} | SL: Rs. {sl}"
    TRADING_CONFIG["trade_logs"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": msg, "ticker": ticker, "type": "EXECUTION"})

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
                            TRADING_CONFIG["trade_logs"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": f"[ALGO EXIT] TARGET HIT for {pos['ticker']} @ Rs. {pos['target']} | PnL: Rs. {round(pnl, 2)}", "ticker": pos["ticker"], "type": "EXIT"})

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
                            TRADING_CONFIG["trade_logs"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "message": f"[ALGO EXIT] STOP LOSS HIT for {pos['ticker']} @ Rs. {pos['sl']} | PnL: Rs. {round(pnl, 2)}", "ticker": pos["ticker"], "type": "EXIT"})
        except Exception as e:
            print(f"Background monitor error: {e}")
        time.sleep(30)

threading.Thread(target=monitor_and_manage_positions, daemon=True).start()

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

        pattern, entry, target, sl, upper_trend,
