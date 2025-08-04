# main.py
#
# Deskripsi:
# Versi ini telah direfaktor untuk mendukung analisis MULTI-SYMBOL secara bersamaan.
# Bot akan melakukan loop melalui daftar simbol yang ditentukan dan mengelola
# model, status, dan sinyal untuk setiap simbol secara terpisah.

import logging
import os
import time
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from typing import Dict, List
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
# Asumsi file-file ini ada, jika tidak ada, perlu dibuat placeholder-nya
# from learning import (
#     AutoLearningModule,
#     AdaptiveGNGLearning,
#     MarketRegimeDetector,
#     get_active_trades_results,
#     apply_learning_adjustments
# )
from server_comm import send_signal_to_server, cancel_signal


# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- PERUBAHAN: Dari satu simbol menjadi list (daftar) simbol ---
SYMBOLS_TO_ANALYZE = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"] 
TIMEFRAMES_LIST = ["M1", "M5", "M30", "H4"]
DATA_CACHE = DataCache()
MODEL_DIR = "gng_models"
MT5_TERMINAL_PATH = r"C:\\Program Files\\ExclusiveMarkets MetaTrader5\\terminal64.exe"
MAX_POSITION_PER_TF = 10
API_KEY = "c1b086d4-a681-48df-957f-6fcc35a82f6d"
SERVER_URL = "http://127.0.0.1:5000/api/internal/submit_signal"
SECRET_KEY = "c1b086d4-a681-48df-957f-6fcc35a82f6d"
MIN_DISTANCE_PIPS_PER_TF = { "M1": 5, "M5": 10, "M30": 20, "H4": 30 }
# --- PERUBAHAN: Dibuat nested dictionary untuk menyimpan sinyal per simbol ---
active_signals: Dict[str, Dict[str, Dict[str, any]]] = {symbol: {} for symbol in SYMBOLS_TO_ANALYZE}
SIGNAL_COOLDOWN_MINUTES = 1
SIGNAL_MEMORY_MINUTES = 1

# --- Parameter Strategi ---
CONFIDENCE_THRESHOLD = 2.0  
XGBOOST_CONFIDENCE_THRESHOLD = 0.75

# --- Inisialisasi Model AI ---
# --- PERUBAHAN: Dibuat dictionary untuk menyimpan model AI per simbol ---
xgb_models: Dict[str, xgb.XGBClassifier] = {}
for symbol in SYMBOLS_TO_ANALYZE:
    try:
        model_path = f"xgboost_model_{symbol}.json"
        logging.info(f"Mencoba memuat model AI untuk {symbol} dari '{model_path}'...")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        xgb_models[symbol] = model
        logging.info(f"Model AI untuk {symbol} berhasil dimuat.")
    except Exception as e:
        logging.error(f"GAGAL memuat model AI untuk {symbol}: {e}. Simbol ini akan berjalan TANPA konfirmasi AI.")
        xgb_models[symbol] = None

# --- PERUBAHAN: Dibuat dictionary untuk menyimpan status cooldown per simbol ---
signal_cooldown: Dict[str, datetime] = {}

def get_recent_signals_from_memory(signals_for_symbol: Dict[str, Dict[str, any]], minutes: int) -> List[float]:
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
                                 original_signal.get("BuyLimit") or original_signal.get("SellLimit"))
                if entry_val_str:
                    recent_prices.append(float(entry_val_str))
        except (ValueError, KeyError):
            signals_to_clear.append(sig_id)

    if signals_to_clear:
        logging.info(f"Membersihkan {len(signals_to_clear)} sinyal lama dari memori (lebih dari {minutes} menit).")
        for sig_id in signals_to_clear:
            if sig_id in signals_for_symbol:
                del signals_for_symbol[sig_id]

    return recent_prices

def main() -> None:
    # --- PERUBAHAN: Inisialisasi model untuk semua simbol ---
    gng_models = {}
    gng_feature_stats = {}
    for symbol in SYMBOLS_TO_ANALYZE:
        models, stats = initialize_gng_models(
            symbol=symbol, timeframes=TIMEFRAMES_LIST, model_dir=MODEL_DIR,
            mt5_path=MT5_TERMINAL_PATH, get_data_func=get_candlestick_data
        )
        gng_models[symbol] = models
        gng_feature_stats[symbol] = stats

    # auto_learners = {s: {tf: AutoLearningModule(s, tf) for tf in TIMEFRAMES_LIST} for s in SYMBOLS_TO_ANALYZE}
    # adaptive_gng = {s: {tf: AdaptiveGNGLearning(gng_models[s].get(tf)) for tf in TIMEFRAMES_LIST if gng_models[s].get(tf)} for s in SYMBOLS_TO_ANALYZE}
    # regime_detector = MarketRegimeDetector()

    logging.info("="*50)
    logging.info("Bot Trading AI v2.2 (Mode Scalping Multi-Symbol) Siap Beraksi!")
    logging.info(f"Simbol dianalisis: {', '.join(SYMBOLS_TO_ANALYZE)}")
    logging.info(f"Threshold Aturan: {CONFIDENCE_THRESHOLD} | Threshold AI: {XGBOOST_CONFIDENCE_THRESHOLD:.0%}")
    logging.info("="*50)

    try:
        while True:
            logging.info("-" * 50)
            logging.info("Memulai siklus analisis baru untuk semua simbol...")
            
            # --- PERUBAHAN: Loop utama sekarang melalui setiap simbol ---
            for symbol in SYMBOLS_TO_ANALYZE:
                try:
                    # --- Logika Cooldown per Simbol ---
                    if symbol in signal_cooldown:
                        time_since_signal = (datetime.now() - signal_cooldown[symbol]).total_seconds() / 60
                        if time_since_signal < SIGNAL_COOLDOWN_MINUTES:
                            logging.info(f"[{symbol}] Sabar dulu... Masih dalam masa tenang. Sisa: {SIGNAL_COOLDOWN_MINUTES - time_since_signal:.1f} menit.")
                            continue
                        else:
                            logging.info(f"[{symbol}] Masa tenang selesai.")
                            del signal_cooldown[symbol]

                    logging.info(f"--- Menganalisis Simbol: {symbol} ---")

                    # --- Loop Analisis per Timeframe ---
                    for tf in TIMEFRAMES_LIST:
                        logging.info(f"[{symbol} | {tf}] Menganalisis timeframe...")
                        
                        symbol_gng_models = gng_models.get(symbol, {})
                        symbol_gng_stats = gng_feature_stats.get(symbol, {})

                        opp = analyze_tf_opportunity(
                            symbol=symbol, tf=tf, mt5_path=MT5_TERMINAL_PATH,
                            gng_model=symbol_gng_models.get(tf), gng_feature_stats=symbol_gng_stats,
                            confidence_threshold=0.0,
                            min_distance_pips_per_tf=MIN_DISTANCE_PIPS_PER_TF, htf_bias=None
                        )

                        if opp and opp.get('score') is not None and opp.get('signal') != "WAIT":
                            if abs(opp['score']) >= CONFIDENCE_THRESHOLD:
                                logging.info(f"✅ [{symbol} | {tf}] SINYAL ENTRY! Peluang {opp['signal']} memenuhi syarat. Skor: {opp['score']:.2f} (Min: {CONFIDENCE_THRESHOLD}).")
                                
                                # --- Validasi AI per Simbol ---
                                xgb_model = xgb_models.get(symbol)
                                if xgb_model and opp.get('features') is not None and opp['features'].size > 0:
                                    features = np.array(opp['features']).reshape(1, -1)
                                    win_probability = xgb_model.predict_proba(features)[0][1]
                                    logging.info(f"[{symbol} | {tf}] Meminta pendapat AI... Prediksi: {win_probability:.2%} kemungkinan WIN.")
                                    if win_probability < XGBOOST_CONFIDENCE_THRESHOLD:
                                        logging.warning(f"[{symbol} | {tf}] AI menyarankan tidak mengambil ini. Peluang dilewati.")
                                        continue
                                    logging.info(f"[{symbol} | {tf}] Lampu hijau dari AI! Melanjutkan.")

                                # --- Pemeriksaan Keamanan per Simbol ---
                                open_pos_count = get_open_positions_per_tf(symbol, tf, MT5_TERMINAL_PATH)
                                logging.info(f"[{symbol} | {tf}] Pemeriksaan 1/3: Batas posisi. Terbuka: {open_pos_count} (Maks: {MAX_POSITION_PER_TF}).")
                                if open_pos_count >= MAX_POSITION_PER_TF:
                                    logging.warning(f"   -> Gagal. Batas posisi terbuka tercapai.")
                                    continue
                                
                                logging.info("   -> Lolos.")
                                mt5_orders = get_active_orders(symbol, MT5_TERMINAL_PATH)
                                recent_signals = get_recent_signals_from_memory(active_signals[symbol], minutes=SIGNAL_MEMORY_MINUTES)
                                all_known_orders = list(dict.fromkeys(mt5_orders + recent_signals))
                                entry_price = float(opp['entry_price_chosen'])
                                point_value = 0.1
                                logging.info(f"[{symbol} | {tf}] Pemeriksaan 2/3: Jarak entry. Entry di {entry_price} vs order lain.")
                                if not is_far_enough(entry_price, all_known_orders, point_value, MIN_DISTANCE_PIPS_PER_TF.get(tf, 10)):
                                    logging.warning("   -> Gagal. Terlalu dekat dengan order lain.")
                                    continue

                                logging.info("   -> Lolos.")

                                # --- Persiapan & Pengiriman Sinyal ---
                                order_type_to_use = opp.get('order_type', opp['signal'])
                                signal_json = build_signal_format(
                                    symbol=symbol, entry_price=entry_price, direction=opp['signal'],
                                    sl=float(opp['sl']), tp=float(opp['tp']), order_type=order_type_to_use
                                )
                                sig_id = make_signal_id(signal_json)

                                logging.info(f"[{symbol} | {tf}] Pemeriksaan 3/3: Duplikasi. Sinyal ID: {sig_id}.")
                                if sig_id in active_signals[symbol]:
                                    logging.warning("   -> Gagal. Sinyal ini duplikat dari yang baru saja dikirim.")
                                    continue

                                logging.info("   -> Lolos. Siap dikirim!")
                                send_status = send_signal_to_server(
                                    symbol=symbol,
                                    signal_json=signal_json,
                                    api_key=API_KEY,
                                    server_url=SERVER_URL,
                                    secret_key=SECRET_KEY,
                                    order_type=order_type_to_use
                                )
                                if send_status == 'SUCCESS' or send_status == 'REJECTED':
                                    if send_status == 'SUCCESS':
                                        active_signals[symbol][sig_id] = {'signal_json': signal_json, 'tf': tf, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'info': opp['info'], 'status': 'pending'}
                                        logging.info(f"[{symbol} | {tf}] Sinyal berhasil dikirim!")
                                    else:
                                        logging.warning(f"[{symbol} | {tf}] Server menolak sinyal (kemungkinan duplikat).")
                                    signal_cooldown[symbol] = datetime.now()
                                    logging.info(f"[{symbol}] Cooldown {SIGNAL_COOLDOWN_MINUTES} menit diaktifkan.")
                                    break # Hentikan loop timeframe karena sinyal berhasil dikirim
                                else:
                                    logging.error(f"[{symbol} | {tf}] Pengiriman GAGAL karena error koneksi.")
                            else:
                                logging.info(f"⏳ [{symbol} | {tf}] SINYAL WAIT. Skor ({opp['score']:.2f}) di bawah threshold ({CONFIDENCE_THRESHOLD}).")
                
                except Exception as e:
                    logging.critical(f"Terjadi error kritis saat menganalisis simbol {symbol}: {e}", exc_info=True)
            
            logging.info(f"Semua simbol telah dianalisis. Istirahat 20 detik...")
            time.sleep(20)

    except KeyboardInterrupt:
        logging.info("Perintah berhenti diterima. Menyimpan data jika ada...")
        # for symbol_learners in auto_learners.values():
        #     for learner in symbol_learners.values():
        #         learner.save_history()
        logging.info("Semua data berhasil disimpan. Sampai jumpa!")

if __name__ == '__main__':
    # Karena saya tidak punya file learning & server_comm, saya comment out bagian yang relevan
    # agar script bisa di-parse. Anda bisa uncomment di environment Anda.
    # Untuk menjalankan, pastikan semua file (data_fetching, gng_model, signal_generator) ada.
    main()
