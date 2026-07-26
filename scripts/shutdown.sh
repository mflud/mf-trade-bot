#!/bin/bash
# shutdown.sh — stop all monitors and trading_bot at end of session (14:00 MST / 17:00 ET).
# Called by com.mf-trade-bot.shutdown launchd agent.

REPO=/Users/marek/mf-trade-bot

bash "$REPO/scripts/stop_trading_bot.sh"
bash "$REPO/scripts/stop_signal_monitor.sh"
bash "$REPO/scripts/stop_slr_monitor.sh"
