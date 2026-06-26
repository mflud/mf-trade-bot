#!/bin/bash
# botdown.sh — cleanly stop ALL bot processes and prevent auto-restart.
#
# Stops:  trading_bot, mes_monitor, bar_collector, bar_recorder health-check
# Disables: watchdog (so nothing auto-restarts until botup.sh is run)
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

# ── 4. Stop mes_monitor screen session ───────────────────────────────────────
if /usr/bin/screen -list | grep -q "mes_monitor"; then
    /usr/bin/screen -S mes_monitor -X quit 2>/dev/null
    echo "  [OK] mes_monitor screen session quit"
else
    echo "  [--] mes_monitor not running"
fi

# ── 5. Unload bar_collector (launchd-managed) ────────────────────────────────
if launchctl list | grep -q "com.mf-trade-bot.bar-collector"; then
    launchctl unload ~/Library/LaunchAgents/com.mf-trade-bot.bar-collector.plist 2>/dev/null
    # Kill any lingering Python process
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

echo ""
echo "All bot processes stopped. Run  bash scripts/botup.sh  to restart."
echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S'): botdown.sh executed" >> "$REPO/logs/cron.log"
