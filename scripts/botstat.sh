#!/bin/bash
# botstat.sh — show status of all MF trade bot processes
REPO=/Users/marek/mf-trade-bot

check() {
    local name=$1
    local pattern=$2
    if pgrep -f "$pattern" > /dev/null 2>&1; then
        echo "  [UP]   $name"
    else
        echo "  [DOWN] $name"
    fi
}

screen_check() {
    local name=$1
    local session=$2
    local pattern=$3
    if /usr/bin/screen -list | grep -q "$session" || pgrep -f "$pattern" > /dev/null 2>&1; then
        echo "  [UP]   $name"
    else
        echo "  [DOWN] $name"
    fi
}

echo ""
echo "=== MF Trade Bot Status  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""
echo "--- Processes ---"
screen_check "bar_collector " "bar_collector" "bar_collector.py"
screen_check "signal_monitor" "signal_monitor" "signal_monitor.py"
screen_check "slr_monitor   " "slr_monitor" "slr_monitor.py"
screen_check "pl_monitor    " "pl_monitor" "pl_monitor.py"
screen_check "trading_bot   " "trading_bot" "trading_bot.py"
check        "bar_recorder  " "bar_recorder"
echo ""

echo "--- Bar DB ---"
DB="$REPO/data/bars.db"
if [ -f "$DB" ]; then
    AGE=$(( $(date +%s) - $(stat -f %m "$DB") ))
    echo "  bars.db last updated: ${AGE}s ago"
    /usr/bin/sqlite3 "$DB" "
        SELECT '  ' || symbol || ' ' || minutes || 'm: ' ||
               COUNT(*) || ' bars, last=' ||
               strftime('%Y-%m-%dT%H:%M', MAX(ts), '-7 hours') || ' MST'
        FROM bars GROUP BY symbol, minutes ORDER BY symbol, minutes;
    " 2>/dev/null || echo "  (db read error)"
else
    echo "  bars.db not found — bar_collector not yet run"
fi
echo ""

echo "--- DOM DB ---"
DOM_DB="$REPO/data/dom.db"
if [ -f "$DOM_DB" ]; then
    DOM_AGE=$(( $(date +%s) - $(stat -f %m "$DOM_DB") ))
    if [ "$DOM_AGE" -le 5 ]; then
        echo "  [OK] dom.db updated ${DOM_AGE}s ago (built into bar_collector)"
    else
        echo "  [WARN] dom.db last updated ${DOM_AGE}s ago — bar_collector may be stale"
    fi
else
    echo "  [DOWN] dom.db not found"
fi
echo ""

echo "--- Recent bar_collector log ---"
tail -5 "$REPO/logs/bar_collector.log" 2>/dev/null | sed 's/^/  /'
echo ""
