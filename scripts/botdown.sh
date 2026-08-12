#!/bin/bash
# botdown.sh — cleanly stop ALL bot processes and prevent auto-restart.
#
# Stops:  trading_bot (focused_bot), focused_monitor, bar_collector, bar-check
# Disables: watchdog (so nothing auto-restarts until botup.sh is run)
#
# IMPORTANT: Use this before manually trading a Combine to avoid cross-account hedging.
#
# Usage:  bash scripts/botdown.sh

REPO=/Users/marek/mf-trade-bot
LOCKFILE="$REPO/logs/bot.disabled"

echo ""
echo "=== Bot Shutdown  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""

# ── 1. Disable watchdog first so it can't restart anything ───────────────────
if launchctl list | grep -q "com.mf-trade-bot.watchdog"; then
    launchctl unload ~/Library/LaunchAgents/com.mf-trade-bot.watchdog.plist 2>/dev/null
    echo "  [OK] watchdog unloaded"
else
    echo "  [--] watchdog already unloaded"
fi

# ── 2. Write lock file (watchdog.sh checks this even if plist re-fires) ───────
touch "$LOCKFILE"
echo "  [OK] lock file written ($LOCKFILE)"

# ── 3. Stop trading_bot ───────────────────────────────────────────────────────
PIDFILE="$REPO/logs/trading_bot.pid"
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "  [OK] trading_bot stopped (pid $PID)"
    else
        echo "  [--] trading_bot pid $PID already gone"
    fi
    rm -f "$PIDFILE"
else
    pkill -f "src/trading_bot.py" 2>/dev/null && echo "  [OK] trading_bot stopped (no pidfile, used pkill)" \
        || echo "  [--] trading_bot not running"
fi

# ── 4. Stop focused_monitor ───────────────────────────────────────────────────
pkill -f "src/focused_monitor.py" 2>/dev/null && echo "  [OK] focused_monitor stopped" \
    || echo "  [--] focused_monitor not running"

# ── 5. Unload bar_collector (launchd-managed) ────────────────────────────────
if launchctl list | grep -q "com.mf-trade-bot.bar-collector"; then
    launchctl unload ~/Library/LaunchAgents/com.mf-trade-bot.bar-collector.plist 2>/dev/null
    pkill -f "src/bar_collector.py" 2>/dev/null || true
    echo "  [OK] bar_collector unloaded"
else
    echo "  [--] bar_collector already unloaded"
fi

# ── 6. Unload bar-check health check ─────────────────────────────────────────
if launchctl list | grep -q "com.mf-trade-bot.bar-check"; then
    launchctl unload ~/Library/LaunchAgents/com.mf-trade-bot.bar-check.plist 2>/dev/null
    echo "  [OK] bar-check unloaded"
else
    echo "  [--] bar-check already unloaded"
fi

# ── 7. Final safety check — confirm trading_bot is truly dead ────────────────
sleep 1
if pgrep -f "src/trading_bot.py" > /dev/null 2>&1; then
    echo "  [WARN] trading_bot still running — force killing"
    pkill -9 -f "src/trading_bot.py" 2>/dev/null
else
    echo "  [OK] confirmed: trading_bot not running"
fi

echo ""
echo "All bot processes stopped. Run  bash scripts/botup.sh  to restart."
echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S'): botdown.sh executed" >> "$REPO/logs/cron.log"
