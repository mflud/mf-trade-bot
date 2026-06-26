#!/bin/bash
# watchdog.sh — restart any crashed screen monitors + trading_bot during session hours.
# Called by com.mf-trade-bot.watchdog launchd agent every 5 minutes.

REPO=/Users/marek/mf-trade-bot

# Respect manual shutdown — botdown.sh writes this lock file
if [[ -f "$REPO/logs/bot.disabled" ]]; then
    exit 0
fi

# Only run Mon–Fri (launchd StartCalendarInterval has no weekday filter)
dow=$(TZ=America/New_York date +%u)   # 1=Mon … 7=Sun
if [[ $dow -ge 6 ]]; then
    exit 0
fi

bash "$REPO/scripts/start_trading_bot.sh"
