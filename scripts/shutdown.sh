#!/bin/bash
# shutdown.sh — stop all monitors and trading_bot at end of session (14:00 MST / 17:00 ET).
# Called by com.mf-trade-bot.shutdown launchd agent.

REPO=/Users/marek/mf-trade-bot

bash "$REPO/scripts/stop_focused_bot.sh"
bash "$REPO/scripts/stop_focused_monitor.sh"
