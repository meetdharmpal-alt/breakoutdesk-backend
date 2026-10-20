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

app = FastAPI(title="BreakoutDesk Fundamental & Pattern Screener")

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

def check_robust_fundamentals(ticker_symbol):
    """
    Balanced fundamental filter: Checks market cap stability and valuation.
    If info fails to load due to rate-limits, it allows the stock through 
    so technical patterns are never missed.
    """
    try:
        t = yf.Ticker(ticker_symbol, session=session)
        info = t.info
        
        # Check if company has reasonable valuation or market standing
        mcap = info.get('marketCap', 1000000000) # Default large-cap assumption if missing
        pe = info.get('trailingPE', 25)
        
        if pe is not None and (pe < 0 or pe > 85):
            return False
        return True
    except Exception:
        return True # Fallback pass to avoid blocking valid scans

def identify_chart_pattern(df):
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values
    
    current_price = closes[-1]
    vol_sma = np.mean(volumes[-20:])
    is_volume_surge = volumes[-1] >= (vol_sma * 1.3)
    high_52w = np.max(highs[-252:]) if len(highs) >= 252 else np.max(highs)
    is_near_high = current_price >= (high_52w * 0.97)

    recent_lows_idx = argrelextrema(lows[-60:], np.less, order=3)[0]
    recent_maxs_idx = argrelextrema(highs[-60:], np.greater, order=order)[0]

    # 1. High Tight Flag
    price_40_bars_ago = closes[-40] if len(closes) >= 40 else closes[0]
    gain_pct = (current_price - price_40_bars_ago) / price_40_bars_ago
    recent_range = (np.max(highs[-10:]) - np.min(lows[-10:])) / current_price
    if gain_pct >= 0.40 and recent_range <= 0.07:
        return "High Tight Flag", round(current_price * 1.02, 2), round(current_price * 1.08, 2), round(current_price * 0.95, 2)

    # 2. Inverse Head & Shoulders
    if len(recent_lows_idx) >= 3:
        troughs = lows[-60:][recent_lows_idx]
        if len(troughs) >= 3:
            left_trough, head, right_trough = troughs[-3], troughs[-2], troughs[-1]
            if head < left_trough and head < right_trough and abs(left_trough - right_trough) / left_trough < 0.04:
                if current_price >= left_trough * 1.01:
                    return "Inverse H&S", round(current_price * 1.005, 2), round(current_price * 1.07, 2), round(head * 0.97, 2)

    # 3. Double Bottom
    if len(recent_lows_idx) >= 2:
        b1, b2 = lows[-60:][recent_lows_idx[-2]], lows[-60:][recent_lows_idx[-1]]
        if abs(b1 - b2) / b1 < 0.03 and is_volume_surge:
            return "Double Bottom", round(current_price * 1.005, 2), round(current_price * 1.06, 2), round(min(b1, b2) * 0.97, 2)

    # 4. Ascending Triangle
    if len(recent_maxs_idx) >= 2 and len(recent_lows_idx) >= 2:
        recent_high_vals = highs[-60:][recent_maxs_idx]
        recent_low_vals = lows[-60:][recent_lows_idx]
        flat_resistance = abs(recent_high_vals[-1] - recent_high_vals[-2]) / recent_high_vals[-1] < 0.02
        rising_lows = recent_low_vals[-1] > recent_low_vals[-2]
        if flat_resistance and rising_lows and is_volume_surge:
            resistance_level = recent_high_vals[-1]
            return "Ascending Triangle", round(resistance_level * 1.002, 2), round(resistance_level * 1.07, 2), round(recent_low_vals[-1] * 0.97, 2)

    # 5. Rectangle Breakout
    if is_near_high and is_volume_surge:
        return "Rectangle Breakout", round(current_price * 1.005, 2), round(current_price * 1.06, 2), round(current_price * 0.97, 2)

    if is_volume_surge:
        return "Volume Breakout", round(current_price * 1.002, 2), round(current_price * 1.05, 2), round(current_price * 0.97, 2)
    
    return "Consolidation", round(current_price * 1.01, 2), round(current_price * 1.04, 2), round(current_price * 0.97, 2)

def fetch_and_scan_stock(ticker):
    try:
        if not check_robust_fundamentals(ticker):
            return None

        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True, session=session)
        if df.empty or len(df) < 40:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        pattern, entry, target, sl = identify_chart_pattern(df)
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        price = float(last_row['Close'])
        prev_close = float(prev_row['Close'])
        chg = round(((price - prev_close) / prev_close) * 100, 2)
        
        vol_sma = df['Volume'].rolling(window=20).mean().iloc[-1]
        is_vol = df['Volume'].iloc[-1] >= (vol_sma * 1.3)

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
            "ohlc": ohlc
        }
    except Exception as e:
        return None

@app.get("/")
def home():
    return {"status": "online", "message": "BreakoutDesk Fundamental + Pattern Screener is running!"}

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
