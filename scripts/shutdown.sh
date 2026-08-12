#!/bin/bash
# shutdown.sh — stop all monitors and trading_bot at end of session (14:00 MST / 17:00 ET).
# Called by com.mf-trade-bot.shutdown launchd agent.

REPO=/Users/marek/mf-trade-bot

bash "$REPO/scripts/stop_focused_bot.sh"
bash "$REPO/scripts/stop_focused_monitor.sh"

# Kill any stale monitor processes that may have been left running
pkill -f "src/mes_monitor.py"    2>/dev/null || true
pkill -f "src/ml_monitor.py"     2>/dev/null || true
pkill -f "src/signal_monitor.py" 2>/dev/null || true
pkill -f "src/slr_monitor.py"    2>/dev/null || true
pkill -f "src/dom_client.py"     2>/dev/null || true
