#!/bin/bash
# botup.sh — restart all bot processes after a botdown.sh shutdown.
#
# Starts:  bar_collector, mes_monitor, trading_bot
# Re-enables: watchdog, bar-check
#
# Usage:  bash scripts/botup.sh

REPO=/Users/marek/mf-trade-bot
LOCKFILE="$REPO/logs/bot.disabled"

echo ""
echo "=== Bot Startup  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""

# ── 1. Remove lock file ───────────────────────────────────────────────────────
rm -f "$LOCKFILE"
echo "  [OK] lock file removed"

# ── 2. Load bar_collector ─────────────────────────────────────────────────────
if ! launchctl list | grep -q "com.mf-trade-bot.bar-collector"; then
    launchctl load ~/Library/LaunchAgents/com.mf-trade-bot.bar-collector.plist
    echo "  [OK] bar_collector loaded"
else
    echo "  [--] bar_collector already loaded"
fi

# ── 3. Load bar-check ─────────────────────────────────────────────────────────
if ! launchctl list | grep -q "com.mf-trade-bot.bar-check"; then
    launchctl load ~/Library/LaunchAgents/com.mf-trade-bot.bar-check.plist
    echo "  [OK] bar-check loaded"
else
    echo "  [--] bar-check already loaded"
fi

# ── 4. Wait for bar_collector to log in and write token ──────────────────────
echo "  Waiting for bar_collector token (up to 30s)…"
for i in $(seq 1 30); do
    if [ -f "$REPO/data/auth_token.txt" ]; then
        AGE=$(( $(date +%s) - $(stat -f %m "$REPO/data/auth_token.txt") ))
        if [ "$AGE" -lt 60 ]; then
            echo "  [OK] token ready (${AGE}s old)"
            break
        fi
    fi
    sleep 1
done

# ── 5. Start mes_monitor ──────────────────────────────────────────────────────
bash "$REPO/scripts/start_mes_monitor.sh"
echo "  [OK] mes_monitor started"

# ── 6. Start trading_bot ─────────────────────────────────────────────────────
bash "$REPO/scripts/start_trading_bot.sh"
echo "  [OK] trading_bot started"

# ── 7. Re-enable watchdog ─────────────────────────────────────────────────────
if ! launchctl list | grep -q "com.mf-trade-bot.watchdog"; then
    launchctl load ~/Library/LaunchAgents/com.mf-trade-bot.watchdog.plist
    echo "  [OK] watchdog loaded"
else
    echo "  [--] watchdog already loaded"
fi

echo ""
echo "All bot processes started. Run  bash scripts/botstat.sh  to verify."
echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S'): botup.sh executed" >> "$REPO/logs/cron.log"
