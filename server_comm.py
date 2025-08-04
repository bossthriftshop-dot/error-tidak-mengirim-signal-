# server_comm.py
#
# Deskripsi:
# Versi ini telah disempurnakan untuk mendukung semua tipe order:
# Market (Buy/Sell Entry), Stop (Buy/Sell Stop), dan Limit (Buy/Sell Limit).

from __future__ import annotations
import logging
import requests
from typing import Dict

def send_signal_to_server(symbol: str, signal_json: Dict[str, str], api_key: str, server_url: str, secret_key: str) -> str:
    """Mengirim sinyal trading ke server dan mengembalikan status keberhasilan."""
    if not isinstance(signal_json, dict):
        logging.error("Tipe data signal_json tidak valid (harus dictionary). Sinyal tidak dikirim.")
        return 'FAILED'

    # --- Logika deteksi tipe sinyal yang diperbarui ---
    signal_type = "WAIT"

    if signal_json.get("DeleteLimit/Stop"):
        signal_type = 'CANCEL'
    elif signal_json.get("BuyEntry") or signal_json.get("BuyStop") or signal_json.get("BuyLimit"):
        signal_type = 'BUY'
    elif signal_json.get("SellEntry") or signal_json.get("SellStop") or signal_json.get("SellLimit"):
        signal_type = 'SELL'

    payload = {
        "signal": signal_type, "signal_json": signal_json, "symbol": symbol,
        "api_key": api_key, "secret_key": secret_key
    }

    try:
        response = requests.post(server_url, json=payload, timeout=10)
        log_message = f"Sinyal {signal_type} untuk {symbol} dikirim."

        if response.status_code == 200:
            logging.info(f"✅ {log_message} Status: BERHASIL.")
            return 'SUCCESS'
        elif 400 <= response.status_code < 500:
            logging.error(f"❌ {log_message} Status: DITOLAK. Respons: {response.text}")
            return 'REJECTED'
        else:
            logging.error(f"❌ {log_message} Status: GAGAL. Kode: {response.status_code}, Respons: {response.text}")
            return 'FAILED'
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error koneksi saat mengirim sinyal: {e}")
        return 'FAILED'

def cancel_signal(signal_id: str, active_signals: Dict[str, Dict[str, any]], api_key: str, server_url: str, secret_key: str) -> None:
    """Membangun dan mengirim sinyal pembatalan untuk semua tipe order (Market, Limit, Stop)."""
    if signal_id not in active_signals:
        return

    original = active_signals[signal_id]['signal_json']
    symbol = original.get("Symbol")

    entry_val = (original.get("BuyEntry") or original.get("SellEntry") or
                 original.get("BuyStop") or original.get("SellStop") or
                 original.get("BuyLimit") or original.get("SellLimit"))

    if not symbol or not entry_val:
        logging.error(f"Data tidak lengkap untuk membatalkan sinyal ID {signal_id}.")
        return

    cancel_json = {
        "Symbol": symbol,
        "DeleteLimit/Stop": entry_val,
        "BuyEntry": "", "BuySL": "", "BuyTP": "", "SellEntry": "", "SellSL": "", "SellTP": "",
        "BuyStop": "", "BuyStopSL": "", "BuyStopTP": "", "SellStop": "", "SellStopSL": "", "SellStopTP": "",
        "BuyLimit": "", "BuyLimitSL": "", "BuyLimitTP": "", "SellLimit": "", "SellLimitSL": "", "SellLimitTP": "",
    }

    send_signal_to_server(symbol, cancel_json, api_key, server_url, secret_key)
    del active_signals[signal_id]