import json
import logging
from typing import Dict, Any, List
from collections import Counter

def analyze_and_adapt_profiles(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Menganalisis feedback per profil dan mengadaptasi bobot (weights) untuk setiap profil.
    """
    learning_params = config.get("learning", {})
    if not learning_params.get("enabled", False):
        # Jika learning dinonaktifkan, kembalikan bobot dasar untuk semua profil
        base_weights = config.get("base_weights", {})
        adapted_weights = {
            profile: base_weights.copy()
            for profile in config.get("strategy_profiles", {})
        }
        return adapted_weights

    feedback_file = learning_params.get("feedback_file", "trade_feedback.json")
    try:
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logging.warning("File feedback '%s' tidak ditemukan atau kosong. Menggunakan bobot dasar.", feedback_file)
        # Kembalikan bobot dasar jika tidak ada feedback
        base_weights = config.get("base_weights", {})
        adapted_weights = {
            profile: base_weights.copy()
            for profile in config.get("strategy_profiles", {})
        }
        return adapted_weights

    # Inisialisasi bobot adaptif dengan bobot dasar
    base_weights = config.get("base_weights", {})
    adapted_weights = {
        profile: base_weights.copy()
        for profile in config.get("strategy_profiles", {})
    }

    # Pisahkan feedback berdasarkan profil
    trades_by_profile: Dict[str, List[Dict]] = {}
    for trade in feedback_data:
        profile_name = trade.get("profile_name")
        if profile_name:
            if profile_name not in trades_by_profile:
                trades_by_profile[profile_name] = []
            trades_by_profile[profile_name].append(trade)

    logging.info("--- Memulai Siklus Pembelajaran Adaptif ---")

    # Lakukan adaptasi untuk setiap profil
    for profile_name, trades in trades_by_profile.items():
        if profile_name not in config.get("strategy_profiles", {}):
            continue

        lookback = learning_params.get("lookback_trades_per_profile", 50)
        min_trades = learning_params.get("min_trades_for_adaptation", 10)
        recent_trades = trades[-lookback:]

        if len(recent_trades) < min_trades:
            logging.info("Profil '%s': Data feedback tidak cukup (%d dari %d), menggunakan bobot dasar.", profile_name, len(recent_trades), min_trades)
            continue

        # Analisis win rate per komponen 'info'
        component_performance = {}
        for trade in recent_trades:
            info_components = trade.get("info", "").split('; ')
            result = 1 if trade.get("result") == "win" else 0

            for component in info_components:
                if not component: continue
                # Ekstrak nama komponen (misal: "BULLISH_OB" dari "BUY_LIMIT based on BULLISH_OB OTE")
                clean_comp = component.split(' ')[-1] if "based on" in component else component.split(' ')[0]

                if clean_comp not in component_performance:
                    component_performance[clean_comp] = {'wins': 0, 'total': 0}
                component_performance[clean_comp]['wins'] += result
                component_performance[clean_comp]['total'] += 1

        # Adaptasi bobot berdasarkan kinerja komponen
        win_rate_target = learning_params.get("win_rate_target", 0.60)
        adjustment_factor = learning_params.get("weight_adjustment_factor", 0.1)

        logging.info("Profil '%s': Menganalisis %d trade terakhir...", profile_name, len(recent_trades))

        current_weights = adapted_weights[profile_name]
        for component, data in component_performance.items():
            if component in current_weights and data['total'] > 5: # Butuh minimal 5 trade untuk komponen
                win_rate = data['wins'] / data['total']
                original_weight = base_weights.get(component, 0)

                # Jika kinerja buruk, kurangi bobot. Jika bagus, naikkan.
                if win_rate < win_rate_target - 0.15:
                    adjustment = abs(original_weight * adjustment_factor)
                    current_weights[component] -= adjustment
                    logging.warning("Profil '%s': Kinerja %s buruk (WR: %.2f%%). Bobot diubah menjadi %.2f", profile_name, component, win_rate*100, current_weights[component])
                elif win_rate > win_rate_target + 0.15:
                    adjustment = abs(original_weight * adjustment_factor)
                    current_weights[component] += adjustment
                    logging.info("Profil '%s': Kinerja %s bagus (WR: %.2f%%). Bobot diubah menjadi %.2f", profile_name, component, win_rate*100, current_weights[component])

    logging.info("--- Siklus Pembelajaran Adaptif Selesai ---")
    return adapted_weights
