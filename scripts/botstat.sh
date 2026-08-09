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
    if /usr/bin/screen -list | grep -q "$session"; then
        echo "  [UP]   $name"
    else
        echo "  [DOWN] $name"
    fi
}

echo ""
echo "=== MF Trade Bot Status  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""
if [ -f "$REPO/logs/bot.disabled" ]; then
    echo "  *** BOT DISABLED (botdown.sh was run — run botup.sh to restart) ***"
    echo ""
fi
echo "--- Processes ---"
check        "bar_collector    " "bar_collector.py"
screen_check "mes_monitor      " "mes_monitor"
check        "trading_bot      " "trading_bot.py"
check        "bar_recorder     " "bar_recorder.py"
check        "ml_trading_bot MES" "ml_trading_bot.py --symbol MES"
check        "ml_trading_bot MNQ" "ml_trading_bot.py --symbol MNQ"
screen_check "ml_monitor        " "ml_monitor"
echo ""

# ML bot state summary
for SYM in MES MNQ; do
    ML_STATE="$REPO/logs/ml_state_${SYM}.json"
    if [ -f "$ML_STATE" ]; then
        echo "--- ML Bot State: $SYM ---"
        python3 -c "
import json, sys
s = json.load(open('$ML_STATE'))
feat = s.get('feat') or {}
sig  = s.get('signal') or {}
pos  = s.get('position')
st   = s.get('session_stats') or {}
upd  = s.get('updated_at','')[:19]
print(f\"  Updated: {upd}\")
print(f\"  Close: {feat.get('close',0):.2f}  vol_regime: {feat.get('vol_regime',0):.2f}  prob: {sig.get('prob',0):.3f}\")
if pos:
    pnl = (feat.get('close',pos['entry_pts']) - pos['entry_pts']) * (1 if pos['direction']=='LONG' else -1)
    print(f\"  Position: {pos['direction']} @ {pos['entry_pts']:.2f}  P&L est: {pnl:+.2f}pts  bars: {pos['bars_held']}\")
else:
    print('  Position: none')
wins = st.get('wins',0); trades = st.get('trades',0)
wr = f\"{wins/trades*100:.0f}%\" if trades else '—'
print(f\"  Session: {st.get('signals',0)} signals  {trades} trades  WR:{wr}  P&L:{st.get('pnl_pts',0):+.2f}pts\")
" 2>/dev/null || echo "  (ml_state_${SYM}.json unreadable)"
        echo ""
    fi
done

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
