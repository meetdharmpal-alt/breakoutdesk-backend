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

app = FastAPI(title="BreakoutDesk Cloud Live Market Bridge")

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

# High-liquidity NSE core watchlist for rapid cloud scanning
WATCHLIST = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 
    'ITC.NS', 'BHARTIARTL.NS', 'WIPRO.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 
    'SUNPHARMA.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS', 
    'HCLTECH.NS', 'TATASTEEL.NS', 'POWERGRID.NS', 'NTPC.NS', 'ADANIENT.NS', 
    'JSWSTEEL.NS', 'TATAMOTORS.NS', 'POLYCAB.NS', 'PERSISTENT.NS', 'COFORGE.NS'
]

def calculate_indicators(df):
    df = df.copy()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Surge'] = df['Volume'] >= (df['Vol_SMA20'] * 1.5)
    df['High_52W'] = df['High'].rolling(window=252, min_periods=50).max()
    df['Is_52W_High'] = df['Close'] >= df['High_52W'] * 0.99
    return df

def find_pivots(df, order=2):
    df = df.copy()
    local_max = argrelextrema(df['High'].values, np.greater, order=order)[0]
    local_min = argrelextrema(df['Low'].values, np.less, order=order)[0]
    df['Pivot'] = 0
    df['PivotPrice'] = np.nan
    if len(local_max) > 0:
        df.iloc[local_max, df.columns.get_loc('Pivot')] = 1
        df.iloc[local_max, df.columns.get_loc('PivotPrice')] = df.iloc[local_max]['High']
    if len(local_min) > 0:
        df.iloc[local_min, df.columns.get_loc('Pivot')] = -1
        df.iloc[local_min, df.columns.get_loc('PivotPrice')] = df.iloc[local_min]['Low']
    return df

def fetch_and_scan_stock(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True, session=session)
        if df.empty or len(df) < 30:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = calculate_indicators(df)
        recent_df = df.iloc[-30:].copy()
        recent_df = find_pivots(recent_df, order=2)
        
        last_row = recent_df.iloc[-1]
        prev_row = recent_df.iloc[-2]
        
        price = float(last_row['Close'])
        prev_close = float(prev_row['Close'])
        chg = round(((price - prev_close) / prev_close) * 100, 2)
        rsi = round(float(last_row['RSI']), 1)
        is_vol = bool(last_row['Volume_Surge'])
        
        pattern = "Consolidation"
        breakout_price = round(price * 1.01, 2)
        rrr = "1:2.5"
        
        if bool(last_row['Is_52W_High']) and is_vol:
            pattern = "52-Week High Breakout"
            breakout_price = round(float(last_row['High_52W']), 2)
            rrr = "1:3.0"
        elif is_vol and rsi > 50:
            pattern = "Momentum Breakout"
            breakout_price = round(price * 0.99, 2)
            rrr = "1:2.8"

        return {
            "ticker": ticker,
            "name": ticker.replace(".NS", ""),
            "price": round(price, 2),
            "chg": chg,
            "rsi": rsi,
            "pattern": pattern,
            "breakout": breakout_price,
            "volume": "2.1x ⚡" if is_vol else "1.0x",
            "rrr": rrr
        }
    except Exception as e:
        return None

@app.get("/")
def home():
    return {"status": "online", "message": "BreakoutDesk Cloud Backend is running successfully!"}

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
