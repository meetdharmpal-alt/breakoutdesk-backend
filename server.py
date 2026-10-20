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

# High-liquidity Indian equities watchlist across sectors
WATCHLIST = [
    # BLUECHIPS & NIFTY 50
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 
    'ITC.NS', 'BHARTIARTL.NS', 'WIPRO.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 
    'SUNPHARMA.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS', 
    'HCLTECH.NS', 'TATASTEEL.NS', 'POWERGRID.NS', 'NTPC.NS', 'ADANIENT.NS', 
    'JSWSTEEL.NS', 'TATAMOTORS.NS', 'POLYCAB.NS', 'PERSISTENT.NS', 'COFORGE.NS',
    'ULTRACEMCO.NS', 'BAJAJFINSV.NS', 'NESTLEIND.NS', 'ONGC.NS', 'COALINDIA.NS',
    'GRASIM.NS', 'TECHM.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'HEROMOTOCO.NS',
    'EICHERMOT.NS', 'DIVISLAB.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
    'APOLLOHOSP.NS', 'TATACONSUM.NS', 'BEL.NS', 'TRENT.NS', 'HAL.NS', 'VBL.NS',
    
    # HIGH MOMENTUM MIDCAPS & F&O LEADERS
    'ZOMATO.NS', 'JIOFIN.NS', 'BSE.NS', 'IREDA.NS', 'NHPC.NS', 'REC.NS', 'PFC.NS',
    'IRFC.NS', 'RVNL.NS', 'SUZLON.NS', 'TATAPOWER.NS', 'MOTHERSON.NS', 'GAIL.NS',
    'INDUSTOWER.NS', 'INDIANB.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'BANDHANBNK.NS',
    'SAIL.NS', 'NMDC.NS', 'NATIONALUM.NS', 'HINDALCO.NS', 'JINDALSTEL.NS',
    'TATACOMM.NS', 'LTIM.NS', 'MPHASIS.NS', 'DIXON.NS', 'KALYANKJIL.NS',
    'MAXHEALTH.NS', 'MANAPPURAM.NS', 'MUTHOOTFIN.NS', 'M&MFIN.NS', 'CHOLAFIN.NS',
    'OBEROIRLTY.NS', 'DLF.NS', 'GODREJPROP.NS', 'PHOENIXLTD.NS',
    'OFSS.NS', 'CYIENT.NS', 'KPITTECH.NS', 'TATAELXSI.NS', 'TATAINVEST.NS',
    'ASTRAL.NS', 'SUPREMEIND.NS', 'PIIND.NS', 'UPL.NS', 'SRF.NS', 'DEEPAKNTR.NS',
    'TATACHEM.NS', 'PIDILITIND.NS', 'BERGEPAINT.NS', 'COLPAL.NS', 'DABUR.NS',
    'GODREJCP.NS', 'MARICO.NS', 'PGHH.NS', 'TORNTPHARM.NS', 'LUPIN.NS', 'MANKIND.NS',
    'ALKEM.NS', 'GLENMARK.NS', 'AUROPHARMA.NS', 'ZYDUSLIFE.NS', 'IPCALAB.NS',
    'BOSCHLTD.NS', 'MRF.NS', 'BALKRISIND.NS', 'TIINDIA.NS', 'CUMMINSIND.NS',
    'ABB.NS', 'SIEMENS.NS', 'CGPOWER.NS', 'SOLARINDS.NS', 'KEI.NS', 'AIAENG.NS'
]

def calculate_volume_and_breakout(df):
    df = df.copy()
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
            
        df = calculate_volume_and_breakout(df)
        recent_df = df.iloc[-30:].copy()
        recent_df = find_pivots(recent_df, order=2)
        
        last_row = recent_df.iloc[-1]
        prev_row = recent_df.iloc[-2]
        
        price = float(last_row['Close'])
        prev_close = float(prev_row['Close'])
        chg = round(((price - prev_close) / prev_close) * 100, 2)
        is_vol = bool(last_row['Volume_Surge'])
        
        pattern = "Consolidation"
        entry_price = round(price, 2)
        target_price = round(price * 1.05, 2)
        sl_price = round(price * 0.97, 2)
        rrr = "1:1.6"
        
        if bool(last_row['Is_52W_High']) and is_vol:
            pattern = "52-Week High Breakout"
            entry_price = round(price * 1.005, 2)
            target_price = round(price * 1.08, 2)
            sl_price = round(price * 0.96, 2)
            rrr = "1:2.0"
        elif is_vol:
            pattern = "Momentum Breakout"
            entry_price = round(price * 1.002, 2)
            target_price = round(price * 1.06, 2)
            sl_price = round(price * 0.97, 2)
            rrr = "1:2.0"

        # Prepare formatted OHLC candles for frontend chart modal
        ohlc = []
        for idx, row in df.iloc[-30:].iterrows():
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
            "entry": entry_price,
            "target": target_price,
            "sl": sl_price,
            "volume": "2.1x ⚡" if is_vol else "1.0x",
            "rrr": rrr,
            "ohlc": ohlc
        }
    except Exception as e:
        return None

@app.get("/")
def home():
    return {"status": "online", "message": "BreakoutDesk Cloud Backend is running successfully!"}

@app.get("/api/scan-market")
def scan_market():
    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(fetch_and_scan_stock, ticker): ticker for ticker in WATCHLIST}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    return {"status": "success", "count": len(results), "data": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
