# main.py (Refactored)
#
# Deskripsi:
# Versi ini telah direfaktor untuk menggunakan file `config.json` eksternal
# dan memiliki struktur kode yang lebih modular dan mudah dikelola.

import logging
import os
import time
import numpy as np
import xgboost as xgb
from datetime import datetime
from typing import Dict, List, Any
import json

from data_fetching import get_candlestick_data, DataCache
from gng_model import initialize_gng_models
from signal_generator import (
    analyze_tf_opportunity,
    build_signal_format,
    make_signal_id,
    get_open_positions_per_tf,
    get_active_orders,
    is_far_enough
)
from server_comm import send_signal_to_server

# --- Global States ---
DATA_CACHE = DataCache()
# active_signals dan signal_cooldown akan diinisialisasi di main()
active_signals: Dict[str, Dict[str, Dict[str, any]]] = {}
signal_cooldown: Dict[str, datetime] = {}

def load_config(filepath: str = "config.json") -> Dict[str, Any]:
    """Memuat konfigurasi dari file JSON."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        logging.basicConfig(
            level=getattr(logging, config['logging']['level'].upper(), logging.INFO),
            format=config['logging']['format']
        )
        logging.info("Konfigurasi berhasil dimuat dari %s", filepath)
        return config
    except FileNotFoundError:
        logging.critical("File konfigurasi '%s' tidak ditemukan. Bot tidak bisa berjalan.", filepath)
        exit()
    except json.JSONDecodeError:
        logging.critical("File konfigurasi '%s' tidak valid. Periksa format JSON.", filepath)
        exit()
    except Exception as e:
        logging.critical("Error saat memuat konfigurasi: %s", e)
        exit()

def initialize_models(config: Dict[str, Any]) -> tuple[dict, dict, dict]:
    """Inisialisasi semua model (GNG, XGBoost) untuk semua simbol."""
    gng_models = {}
    gng_feature_stats = {}
    xgb_models = {}

    symbols = config['symbols_to_analyze']
    timeframes = config['timeframes_list']
    model_dir = config.get('model_dir', 'gng_models')
    mt5_path = config['mt5_terminal_path']

    logging.info("--- Inisialisasi Model GNG ---")
    for symbol in symbols:
        try:
            models, stats = initialize_gng_models(
                symbol=symbol, timeframes=timeframes, model_dir=model_dir,
                mt5_path=mt5_path, get_data_func=get_candlestick_data
            )
            gng_models[symbol] = models
            gng_feature_stats[symbol] = stats
            logging.info("Model GNG untuk %s berhasil diinisialisasi.", symbol)
        except Exception as e:
            logging.error("Gagal inisialisasi model GNG untuk %s: %s", symbol, e)
            gng_models[symbol] = {}
            gng_feature_stats[symbol] = {}

    logging.info("--- Inisialisasi Model AI (XGBoost) ---")
    for symbol in symbols:
        try:
            model_path = f"xgboost_model_{symbol}.json"
            logging.info("Mencoba memuat model AI untuk %s dari '%s'...", symbol, model_path)
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            xgb_models[symbol] = model
            logging.info("Model AI untuk %s berhasil dimuat.", symbol)
        except Exception as e:
            logging.error("GAGAL memuat model AI untuk %s: %s. Simbol ini akan berjalan TANPA konfirmasi AI.", symbol, e)
            xgb_models[symbol] = None

    return gng_models, gng_feature_stats, xgb_models

def get_recent_signals_from_memory(signals_for_symbol: Dict[str, Dict[str, any]], minutes: int) -> List[float]:
    """Membersihkan sinyal lama dari memori dan mengembalikan harga entry dari sinyal baru."""
    recent_prices = []
    now = datetime.now()
    signals_to_clear = []

    for sig_id, signal_data in list(signals_for_symbol.items()):
        try:
            signal_time = datetime.strptime(signal_data['timestamp'], "%Y-%m-%d %H:%M:%S")
            if (now - signal_time).total_seconds() > (minutes * 60):
                signals_to_clear.append(sig_id)
            else:
                original_signal = signal_data['signal_json']
                entry_val_str = (original_signal.get("BuyEntry") or original_signal.get("SellEntry") or
                                 original_signal.get("BuyStop") or original_signal.get("SellStop") or
                                 original_signal.get("Buylimit") or original_signal.get("Selllimit"))
                if entry_val_str:
                    recent_prices.append(float(entry_val_str))
        except (ValueError, KeyError) as e:
            logging.warning("Error memproses sinyal memori (ID: %s): %s. Sinyal akan dihapus.", sig_id, e)
            signals_to_clear.append(sig_id)

    if signals_to_clear:
        logging.info("Membersihkan %d sinyal lama dari memori (lebih dari %d menit).", len(signals_to_clear), minutes)
        for sig_id in signals_to_clear:
            if sig_id in signals_for_symbol:
                del signals_for_symbol[sig_id]

    return recent_prices

def handle_opportunity(opp: Dict[str, Any], symbol: str, tf: str, config: Dict[str, Any], xgb_model: xgb.XGBClassifier):
    """Memproses, memvalidasi, dan mengirim sinyal jika ada peluang yang memenuhi syarat."""
    global active_signals, signal_cooldown

    strategy_params = config['strategy_params']
    confidence_threshold = strategy_params['confidence_threshold']

    if abs(opp['score']) < confidence_threshold:
        logging.info("⏳ [%s | %s] SINYAL WAIT. Skor (%.2f) di bawah threshold (%.1f).", symbol, tf, opp['score'], confidence_threshold)
        return False

    logging.info("✅ [%s | %s] SINYAL ENTRY! Peluang %s memenuhi syarat. Skor: %.2f (Min: %.1f).", symbol, tf, opp['signal'], opp['score'], confidence_threshold)

    # --- Validasi AI ---
    if xgb_model and opp.get('features') is not None and opp['features'].size > 0:
        features = np.array(opp['features']).reshape(1, -1)
        win_probability = xgb_model.predict_proba(features)[0][1]
        logging.info("[%s | %s] Meminta pendapat AI... Prediksi: %.2f%% kemungkinan WIN.", symbol, tf, win_probability * 100)
        if win_probability < strategy_params['xgboost_confidence_threshold']:
            logging.warning("[%s | %s] AI menyarankan tidak mengambil ini (Probabilitas %.2f%% < Threshold %.2f%%). Peluang dilewati.", symbol, tf, win_probability * 100, strategy_params['xgboost_confidence_threshold'] * 100)
            return False
        logging.info("[%s | %s] Lampu hijau dari AI! Melanjutkan.", symbol, tf)

    # --- Pemeriksaan Keamanan ---
    open_pos_count = get_open_positions_per_tf(symbol, tf, config['mt5_terminal_path'])
    logging.info("[%s | %s] Pemeriksaan 1/3: Batas posisi. Terbuka: %d (Maks: %d).", symbol, tf, open_pos_count, config['max_position_per_tf'])
    if open_pos_count >= config['max_position_per_tf']:
        logging.warning("   -> Gagal. Batas posisi terbuka tercapai.")
        return False
    logging.info("   -> Lolos.")

    mt5_orders = get_active_orders(symbol, config['mt5_terminal_path'])
    recent_signals = get_recent_signals_from_memory(active_signals[symbol], config['signal_memory_minutes'])
    all_known_orders = list(dict.fromkeys(mt5_orders + recent_signals))
    entry_price = float(opp['entry_price_chosen'])

    logging.info("[%s | %s] Pemeriksaan 2/3: Jarak entry. Entry di %.5f vs order lain.", symbol, tf, entry_price)
    if not is_far_enough(entry_price, all_known_orders, 0.1, config['min_distance_pips_per_tf'].get(tf, 10)):
        logging.warning("   -> Gagal. Terlalu dekat dengan order lain.")
        return False
    logging.info("   -> Lolos.")

    # --- Persiapan & Pengiriman Sinyal ---
    order_type_to_use = opp.get('order_type', opp['signal'])
    signal_json = build_signal_format(
        symbol=symbol, entry_price=entry_price, direction=opp['signal'],
        sl=float(opp['sl']), tp=float(opp['tp']), order_type=order_type_to_use
    )
    sig_id = make_signal_id(signal_json)

    logging.info("[%s | %s] Pemeriksaan 3/3: Duplikasi. Sinyal ID: %s.", symbol, tf, sig_id)
    if sig_id in active_signals[symbol]:
        logging.warning("   -> Gagal. Sinyal ini duplikat dari yang baru saja dikirim.")
        return False
    logging.info("   -> Lolos. Siap dikirim!")

    send_status = send_signal_to_server(
        symbol=symbol, signal_json=signal_json, api_key=config['api_key'],
        server_url=config['server_url'], secret_key=config['secret_key'],
        order_type=order_type_to_use
    )

    if send_status == 'SUCCESS':
        active_signals[symbol][sig_id] = {'signal_json': signal_json, 'tf': tf, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'info': opp['info'], 'status': 'pending'}
        logging.info("[%s | %s] Sinyal berhasil dikirim!", symbol, tf)
        signal_cooldown[symbol] = datetime.now()
        logging.info("[%s] Cooldown %d menit diaktifkan.", symbol, config['signal_cooldown_minutes'])
        return True # Sinyal berhasil dikirim
    elif send_status == 'REJECTED':
        logging.warning("[%s | %s] Server menolak sinyal (kemungkinan duplikat atau filter server).", symbol, tf)
    else: # FAILED
        logging.error("[%s | %s] Pengiriman GAGAL karena error koneksi.", symbol, tf)

    return False

def process_symbol(symbol: str, config: Dict[str, Any], models: Dict[str, Any]):
    """Menjalankan seluruh siklus analisis untuk satu simbol."""
    global signal_cooldown

    # --- Pengecekan Cooldown ---
    cooldown_minutes = config['signal_cooldown_minutes']
    if symbol in signal_cooldown:
        time_since_signal = (datetime.now() - signal_cooldown[symbol]).total_seconds() / 60
        if time_since_signal < cooldown_minutes:
            logging.info("[%s] Sabar dulu... Masih dalam masa tenang. Sisa: %.1f menit.", symbol, cooldown_minutes - time_since_signal)
            return
        else:
            logging.info("[%s] Masa tenang selesai.", symbol)
            del signal_cooldown[symbol]

    logging.info("--- Menganalisis Simbol: %s ---", symbol)

    # --- Loop Analisis per Timeframe ---
    for tf in config['timeframes_list']:
        logging.info("[%s | %s] Menganalisis timeframe...", symbol, tf)

        try:
            opp = analyze_tf_opportunity(
                symbol=symbol, tf=tf, mt5_path=config['mt5_terminal_path'],
                gng_model=models['gng'].get(symbol, {}).get(tf),
                gng_feature_stats=models['gng_stats'].get(symbol, {}),
                confidence_threshold=0.0, # Kirim 0.0 agar semua dianalisis, filter di handle_opportunity
                min_distance_pips_per_tf=config['min_distance_pips_per_tf'],
                htf_bias=None
            )

            if opp and opp.get('signal') != "WAIT":
                signal_sent = handle_opportunity(opp, symbol, tf, config, models['xgb'].get(symbol))
                if signal_sent:
                    break # Hentikan loop timeframe untuk simbol ini karena sinyal sudah dikirim
        except Exception as e:
            logging.error("Error saat menganalisis %s | %s: %s", symbol, tf, e, exc_info=True)


def main():
    """Fungsi utama untuk menjalankan bot."""
    global active_signals, signal_cooldown

    config = load_config()
    symbols_to_analyze = config['symbols_to_analyze']

    # Inisialisasi state global berdasarkan config
    active_signals = {symbol: {} for symbol in symbols_to_analyze}
    signal_cooldown = {}

    gng_models, gng_stats, xgb_models = initialize_models(config)
    all_models = {'gng': gng_models, 'gng_stats': gng_stats, 'xgb': xgb_models}

    logging.info("="*50)
    logging.info("Bot Trading AI v3.0 (Refactored) Siap Beraksi!")
    logging.info("Simbol dianalisis: %s", ', '.join(symbols_to_analyze))
    logging.info("Threshold Aturan: %.1f | Threshold AI: %.0f%%",
                 config['strategy_params']['confidence_threshold'],
                 config['strategy_params']['xgboost_confidence_threshold'] * 100)
    logging.info("="*50)

    try:
        while True:
            logging.info("-" * 50)
            logging.info("Memulai siklus analisis baru untuk semua simbol...")
            
            for symbol in symbols_to_analyze:
                process_symbol(symbol, config, all_models)

            sleep_duration = config.get('main_loop_sleep_seconds', 20)
            logging.info("Semua simbol telah dianalisis. Istirahat %d detik...", sleep_duration)
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logging.info("Perintah berhenti diterima. Bot akan dimatikan.")
    finally:
        logging.info("Aplikasi Selesai.")


if __name__ == '__main__':
    main()
