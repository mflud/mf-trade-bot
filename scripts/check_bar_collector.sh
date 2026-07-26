#!/usr/bin/env bash
# check_bar_collector.sh — Alert if bars.db hasn't been updated recently.
# Run from cron every 5 minutes during market hours.
#
# Sends a macOS notification if bars.db is stale for > STALE_SECS.
# Cooldown file prevents repeat alerts within COOLDOWN_SECS.

REPO=/Users/marek/mf-trade-bot
DB_PATH="$REPO/data/bars.db"
COOLDOWN_FILE="$REPO/logs/bar_stale_alert.ts"

STALE_SECS=300       # alert if bars.db not updated in 5 minutes
COOLDOWN_SECS=3600   # don't re-alert within 1 hour

# ── Only run Mon–Fri during active session: 09:00–16:00 ET = 06:00–13:00 MST ─
hour=$(date +%H)
day=$(date +%u)   # 1=Mon … 7=Sun
if [[ $day -ge 6 ]] || [[ $hour -lt 6 ]] || [[ $hour -ge 14 ]]; then
    exit 0
fi

# ── Check bars.db mtime ───────────────────────────────────────────────────────
if [[ ! -f "$DB_PATH" ]]; then
    exit 0
fi

now=$(date +%s)
mtime=$(stat -f %m "$DB_PATH")
age=$(( now - mtime ))

if [[ $age -lt $STALE_SECS ]]; then
    exit 0
fi

# ── Cooldown check ────────────────────────────────────────────────────────────
if [[ -f "$COOLDOWN_FILE" ]]; then
    last_alert=$(cat "$COOLDOWN_FILE")
    since=$(( now - last_alert ))
    if [[ $since -lt $COOLDOWN_SECS ]]; then
        exit 0
    fi
fi

# ── Send macOS notification ───────────────────────────────────────────────────
minutes=$(( age / 60 ))
osascript -e "display notification \"bars.db stale ${minutes}min — close TopstepX, then run barUnloadLoad.sh\" with title \"⚠️ bar_collector\" sound name \"Ping\""

echo "$now" > "$COOLDOWN_FILE"
