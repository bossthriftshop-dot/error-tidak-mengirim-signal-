"""
technical_indicators.py (refactor advanced)
===========================================

Versi refaktor: multi-zone detection, scoring, pattern, boundary, pivot,
feature extractor, zone plotting & event logging.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt

# ========== ZONE & STRUCTURE ==========

def calculate_atr_dynamic(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean().iloc[-1]
    return float(atr_val)

def detect_structure(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    if len(df) < 20:
        return "INDECISIVE", None
    df = df.copy()
    df['is_swing_high'] = (df['high'].shift(2) < df['high'].shift(1)) & \
                          (df['high'].shift(1) < df['high']) & \
                          (df['high'] > df['high'].shift(-1)) & \
                          (df['high'].shift(-1) > df['high'].shift(-2))
    df['is_swing_low'] = (df['low'].shift(2) > df['low'].shift(1)) & \
                         (df['low'].shift(1) > df['low']) & \
                         (df['low'] < df['low'].shift(-1)) & \
                         (df['low'].shift(-1) < df['low'].shift(-2))
    swing_highs = df[df['is_swing_high']].index
    swing_lows = df[df['is_swing_low']].index
    res = []
    last_swing_high = None
    last_swing_low = None
    if len(swing_highs) > 0:
        last_idx = swing_highs[-1]
        last_swing_high = df.loc[last_idx, 'high']
        if len(swing_highs) > 1:
            prev_idx = swing_highs[-2]
            prev_high = df.loc[prev_idx, 'high']
            if last_swing_high > prev_high:
                res.append('HH')
            elif last_swing_high < prev_high:
                res.append('LH')
    if len(swing_lows) > 0:
        last_idx = swing_lows[-1]
        last_swing_low = df.loc[last_idx, 'low']
        if len(swing_lows) > 1:
            prev_idx = swing_lows[-2]
            prev_low = df.loc[prev_idx, 'low']
            if last_swing_low > prev_low:
                res.append('HL')
            elif last_swing_low < prev_low:
                res.append('LL')
    current_price = df['close'].iloc[-1]
    if last_swing_high is not None and current_price > last_swing_high:
        res.append("BULLISH_BOS")
    if last_swing_low is not None and current_price < last_swing_low:
        res.append("BEARISH_BOS")
    structure_str = "/".join(res) if res else "INDECISIVE"
    return structure_str, {'last_high': last_swing_high, 'last_low': last_swing_low}

def detect_order_blocks_multi(
    df: pd.DataFrame, lookback: int = 15, structure_filter: Optional[str]=None, max_age: int=20
) -> List[Dict[str, Any]]:
    order_blocks = []
    now_price = df['close'].iloc[-1]
    atr = calculate_atr_dynamic(df)
    last_idx = df.index[-1]
    for i in range(len(df)-lookback-2, len(df)-2):
        candle = df.iloc[i]
        age = last_idx - df.index[i]
        if age > max_age:
            continue
        # Bullish OB
        if candle['close'] < candle['open']:
            found_bos = False
            for j in range(i+1, min(i+lookback+1, len(df))):
                if df.iloc[j]['close'] > candle['high']:
                    found_bos = True
                    break
            if found_bos:
                if structure_filter and structure_filter != "BULLISH_BOS":
                    continue
                strength = (abs(candle['open'] - candle['close']) / atr) if atr else 1
                order_blocks.append({
                    'type': 'BULLISH_OB',
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'time': candle['time'],
                    'age': age,
                    'strength': strength,
                    'distance': abs(now_price - ((candle['open']+candle['close'])/2)),
                })
        # Bearish OB
        elif candle['close'] > candle['open']:
            found_bos = False
            for j in range(i+1, min(i+lookback+1, len(df))):
                if df.iloc[j]['close'] < candle['low']:
                    found_bos = True
                    break
            if found_bos:
                if structure_filter and structure_filter != "BEARISH_BOS":
                    continue
                strength = (abs(candle['open'] - candle['close']) / atr) if atr else 1
                order_blocks.append({
                    'type': 'BEARISH_OB',
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'time': candle['time'],
                    'age': age,
                    'strength': strength,
                    'distance': abs(now_price - ((candle['open']+candle['close'])/2)),
                })
    order_blocks = sorted(order_blocks, key=lambda x: (x['distance'], -x['strength'], x['age']))
    return order_blocks

def detect_fvg_multi(df: pd.DataFrame, min_gap: float=0.0002, max_age: int=20) -> List[Dict[str, Any]]:
    fvg_zones = []
    now_price = df['close'].iloc[-1]
    atr = calculate_atr_dynamic(df)
    last_idx = df.index[-1]
    for i in range(1, len(df)-1):
        c_prev, c_now, c_next = df.iloc[i-1], df.iloc[i], df.iloc[i+1]
        age = last_idx - df.index[i]
        if age > max_age:
            continue
        # Bullish FVG
        gap = c_next['low'] - c_prev['high']
        if gap > min_gap:
            strength = gap / atr if atr else 1
            fvg_zones.append({
                'type': 'FVG_BULLISH',
                'start': float(c_prev['high']),
                'end': float(c_next['low']),
                'time': c_now['time'],
                'age': age,
                'strength': strength,
                'distance': abs(now_price - ((c_prev['high']+c_next['low'])/2))
            })
        # Bearish FVG
        gap = c_prev['low'] - c_next['high']
        if gap > min_gap:
            strength = gap / atr if atr else 1
            fvg_zones.append({
                'type': 'FVG_BEARISH',
                'start': float(c_next['high']),
                'end': float(c_prev['low']),
                'time': c_now['time'],
                'age': age,
                'strength': strength,
                'distance': abs(now_price - ((c_next['high']+c_prev['low'])/2))
            })
    fvg_zones = sorted(fvg_zones, key=lambda x: (x['distance'], -x['strength'], x['age']))
    return fvg_zones

# ========== PATTERN, VOLUME, BOUNDARY, PIVOT ==========

def detect_pinbar(df: pd.DataFrame, min_ratio: float=2.0) -> List[Dict[str, Any]]:
    """Pinbar: ekor minimal 2x body."""
    pattern = []
    for i in range(2, len(df)):
        c = df.iloc[i]
        body = abs(c['close'] - c['open'])
        upper = c['high'] - max(c['close'], c['open'])
        lower = min(c['close'], c['open']) - c['low']
        # Bullish pinbar
        if lower > min_ratio * body and upper < 0.5 * body:
            pattern.append({'type': 'PINBAR_BULL', 'time': c['time'], 'idx': i})
        # Bearish pinbar
        elif upper > min_ratio * body and lower < 0.5 * body:
            pattern.append({'type': 'PINBAR_BEAR', 'time': c['time'], 'idx': i})
    return pattern

def detect_engulfing(df: pd.DataFrame) -> List[Dict[str, Any]]:
    pattern = []
    for i in range(1, len(df)):
        prev, c = df.iloc[i-1], df.iloc[i]
        # Bullish engulfing
        if c['close'] > c['open'] and prev['close'] < prev['open'] and c['close'] > prev['open'] and c['open'] < prev['close']:
            pattern.append({'type': 'ENGULFING_BULL', 'time': c['time'], 'idx': i})
        # Bearish engulfing
        elif c['close'] < c['open'] and prev['close'] > prev['open'] and c['close'] < prev['open'] and c['open'] > prev['close']:
            pattern.append({'type': 'ENGULFING_BEAR', 'time': c['time'], 'idx': i})
    return pattern

def detect_volume_spike(df: pd.DataFrame, window: int=20, threshold: float=2.0) -> List[Dict[str, Any]]:
    if 'tick_volume' not in df.columns:
        return []
    pattern = []
    vol_ma = df['tick_volume'].rolling(window).mean()
    for i in range(window, len(df)):
        if df['tick_volume'].iloc[i] > threshold * vol_ma.iloc[i]:
            pattern.append({'type': 'VOLUME_SPIKE', 'time': df['time'].iloc[i], 'idx': i})
    return pattern

def get_daily_high_low(df: pd.DataFrame) -> Dict[str, Any]:
    today = pd.to_datetime(df['time'].iloc[-1]).date()
    today_data = df[pd.to_datetime(df['time']).dt.date == today]
    high = today_data['high'].max()
    low = today_data['low'].min()
    last_close = today_data['close'].iloc[-1] if len(today_data) else df['close'].iloc[-1]
    return {
        'daily_high': high,
        'daily_low': low,
        'distance_to_high': abs(last_close - high),
        'distance_to_low': abs(last_close - low)
    }

def get_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    # Classic pivot, pakai data daily terakhir
    today = pd.to_datetime(df['time'].iloc[-1]).date()
    prev_day = today - pd.Timedelta(days=1)
    prev_data = df[pd.to_datetime(df['time']).dt.date == prev_day]
    if len(prev_data) == 0:
        prev_data = df.iloc[-20:]  # fallback: 20 bar terakhir
    high, low, close = prev_data['high'].max(), prev_data['low'].min(), prev_data['close'].iloc[-1]
    pivot = (high + low + close) / 3
    r1 = 2*pivot - low
    s1 = 2*pivot - high
    return {'pivot': pivot, 'r1': r1, 's1': s1}

# ========== FEATURE EXTRACTOR ==========

def extract_features_full(
    df: pd.DataFrame,
    structure: str,
    order_blocks: List[Dict[str, Any]],
    fvg_zones: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    boundary: Dict[str, Any],
    pivot: Dict[str, float]
) -> np.ndarray:
    """Keluarkan vektor fitur komprehensif untuk AI/ML."""
    last_price = df['close'].iloc[-1]
    features = [
        last_price,
        calculate_atr_dynamic(df),
        min([ob['distance'] for ob in order_blocks]) if order_blocks else 0,
        max([ob['strength'] for ob in order_blocks]) if order_blocks else 0,
        min([fvg['distance'] for fvg in fvg_zones]) if fvg_zones else 0,
        max([fvg['strength'] for fvg in fvg_zones]) if fvg_zones else 0,
        int(any(p['type'].startswith('ENGULFING') for p in patterns)),
        boundary.get('distance_to_high', 0),
        boundary.get('distance_to_low', 0),
        pivot.get('r1', 0) - last_price,
        pivot.get('s1', 0) - last_price,
        # Tambah fitur lain sesuai kebutuhan...
    ]
    return np.array(features, dtype=float)

# ========== TARGET GENERATOR (LABEL OTOMATIS) ==========

def generate_label_fvg(df: pd.DataFrame, fvg_zones: List[Dict[str, Any]], horizon: int=10, pip_thresh: float=0.001) -> List[int]:
    """
    Label 1 jika X bar setelah FVG harga naik/turun >= pip_thresh, else 0.
    """
    label = [0]*len(df)
    for fvg in fvg_zones:
        idx = df[df['time']==fvg['time']].index[0]
        base_price = (fvg['start'] + fvg['end'])/2
        future_idx = min(idx + horizon, len(df)-1)
        if 'BULLISH' in fvg['type']:
            if df['high'].iloc[future_idx] - base_price >= pip_thresh:
                label[idx] = 1
        elif 'BEARISH' in fvg['type']:
            if base_price - df['low'].iloc[future_idx] >= pip_thresh:
                label[idx] = 1
    return label

# ========== ZONE PLOTTING & EVENT LOGGING ==========

def plot_zones(df, order_blocks, fvg_zones, patterns=None):
    plt.figure(figsize=(15, 7))
    plt.plot(df['time'], df['close'], label='Close Price', color='black')
    for ob in order_blocks:
        plt.axhspan(ob['low'], ob['high'], color='green' if 'BULL' in ob['type'] else 'red', alpha=0.25)
    for fvg in fvg_zones:
        plt.axhspan(fvg['start'], fvg['end'], color='blue' if 'BULL' in fvg['type'] else 'orange', alpha=0.15)
    if patterns:
        for p in patterns:
            idx = p['idx']
            plt.scatter(df['time'].iloc[idx], df['close'].iloc[idx], marker='^' if 'BULL' in p['type'] else 'v', color='purple')
    plt.legend()
    plt.title("Price with OB/FVG/Pattern Zones")
    plt.show()

def log_zone_events(order_blocks, fvg_zones, patterns):
    """Log bar yang trigger event, useful untuk AI/analitik."""
    event_log = []
    for ob in order_blocks:
        event_log.append({'time': ob['time'], 'price': (ob['high']+ob['low'])/2, 'type': ob['type']})
    for fvg in fvg_zones:
        event_log.append({'time': fvg['time'], 'price': (fvg['start']+fvg['end'])/2, 'type': fvg['type']})
    for p in patterns:
        event_log.append({'time': p['time'], 'price': None, 'type': p['type']})
    event_log = sorted(event_log, key=lambda x: x['time'])
    return event_log

# ========== EQH/EQL & LIQUIDITY SWEEP ==========

def detect_eqh_eql(df: pd.DataFrame, window: int = 10, tolerance: float = 0.0003) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    if len(df) < window:
        return [], []
    eqh, eql = [], []
    for i in range(window, len(df)):
        hi = df['high'].iloc[i-window:i]
        lo = df['low'].iloc[i-window:i]
        if (hi.max() - hi.min()) < (hi.mean() * tolerance):
            eqh.append({'time': df['time'].iloc[i], 'value': float(hi.mean())})
        if (lo.max() - lo.min()) < (lo.mean() * tolerance):
            eql.append({'time': df['time'].iloc[i], 'value': float(lo.mean())})
    return eqh, eql

def detect_liquidity_sweep(df: pd.DataFrame, min_candles_for_swing: int = 5) -> List[Dict[str, Any]]:
    liquidity_sweeps = []
    if len(df) < min_candles_for_swing * 2:
        return liquidity_sweeps
    df = df.copy()
    df['is_swing_high'] = (df['high'].rolling(min_candles_for_swing, center=True).max() == df['high'])
    df['is_swing_low'] = (df['low'].rolling(min_candles_for_swing, center=True).min() == df['low'])
    swing_highs_idx = df[df['is_swing_high']].index.tolist()
    swing_lows_idx = df[df['is_swing_low']].index.tolist()
    current_candle_idx = len(df) - 1
    for low_idx in reversed(swing_lows_idx):
        if low_idx < current_candle_idx - 1:
            prev_swing_low_val = df['low'].loc[low_idx]
            for i in range(low_idx + 1, current_candle_idx + 1):
                if df['low'].iloc[i] < prev_swing_low_val and df['close'].iloc[i] > prev_swing_low_val:
                    if df['close'].iloc[current_candle_idx] > df['open'].iloc[current_candle_idx]:
                        liquidity_sweeps.append({
                            'type': 'BULLISH_LS',
                            'swept_level': float(prev_swing_low_val),
                            'sweep_time': df['time'].loc[i],
                            'current_time': df['time'].iloc[current_candle_idx]
                        })
                        break
    for high_idx in reversed(swing_highs_idx):
        if high_idx < current_candle_idx - 1:
            prev_swing_high_val = df['high'].loc[high_idx]
            for i in range(high_idx + 1, current_candle_idx + 1):
                if df['high'].iloc[i] > prev_swing_high_val and df['close'].iloc[i] < prev_swing_high_val:
                    if df['close'].iloc[current_candle_idx] < df['open'].iloc[current_candle_idx]:
                        liquidity_sweeps.append({
                            'type': 'BEARISH_LS',
                            'swept_level': float(prev_swing_high_val),
                            'sweep_time': df['time'].loc[i],
                            'current_time': df['time'].iloc[current_candle_idx]
                        })
                        break
    return liquidity_sweeps

# ========== OTE ZONE ==========

def calculate_optimal_trade_entry(swing_start_price: float, swing_end_price: float, direction: str) -> Dict[str, float]:
    fib_levels = {
        '0.618': 0.618,
        '0.705': 0.705,
        '0.790': 0.790
    }
    ote_zone: Dict[str, float] = {}
    high_val = max(swing_start_price, swing_end_price)
    low_val = min(swing_start_price, swing_end_price)
    price_range = high_val - low_val
    if direction.upper() == "BUY":
        ote_zone['upper'] = high_val - (price_range * fib_levels['0.618'])
        ote_zone['mid'] = high_val - (price_range * fib_levels['0.705'])
        ote_zone['lower'] = high_val - (price_range * fib_levels['0.790'])
    elif direction.upper() == "SELL":
        ote_zone['upper'] = low_val + (price_range * fib_levels['0.790'])
        ote_zone['mid'] = low_val + (price_range * fib_levels['0.705'])
        ote_zone['lower'] = low_val + (price_range * fib_levels['0.618'])
    return ote_zone

# ========== END OF MODULE ==========
