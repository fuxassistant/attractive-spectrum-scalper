"""Reusable formulas extracted from spectrum/player/data.py; see docs/MIGRATION.json."""


FEATURES=('spread_ticks','imbalance_l1','depth_bid_12','depth_ask_12',
          'displacement_5s','displacement_30s','displacement_60s','path_30s','efficiency_30s',
          'trade_buy_30s','trade_sell_30s','trade_unknown_30s','add_bid_30s','add_ask_30s',
          'remove_bid_30s','remove_ask_30s','matched_fill_remove_bid_30s','matched_fill_remove_ask_30s',
          'fft_mid_change_32s_power','fft_mid_change_64s_power','fft_mid_change_128s_power',
          'fft_mid_change_32s_mean','fft_mid_change_128s_mean','fft_mid_change_128s_entropy',
          'fft_signed_trade_32s_power','fft_signed_trade_128s_power',
          'fft_bid_add_32s_power','fft_ask_add_32s_power','fft_bid_remove_32s_power','fft_ask_remove_32s_power')
