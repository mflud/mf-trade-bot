#!/bin/bash
# botfix.sh — detect and optionally fix duplicate/dead bot processes
# Usage:  scripts/botfix.sh          (report only)
#         scripts/botfix.sh --fix    (kill duplicates, restart dead monitors)
#
# trading_bot is NEVER auto-started by --fix (paper vs live is a manual choice).

REPO=/Users/marek/mf-trade-bot
FIX=false
[[ "$1" == "--fix" ]] && FIX=true

PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
ISSUES=0

echo ""
echo "=== Bot Process Check  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""

# ── helper: Python interpreter PIDs running a script (excludes screen/bash wrappers) ──
# pgrep -f finds all PIDs with the script in their args (including screen/bash parents).
# The while loop then checks each PID's actual executable name via ps -o comm=,
# keeping only those where the binary itself is Python.
python_pids() {
    local script="src/$1"
    pgrep -f "$script" 2>/dev/null | while read -r pid; do
        local comm
        comm=$(ps -p "$pid" -o comm= 2>/dev/null)
        [[ "$comm" =~ [Pp]ython ]] && echo "$pid"
    done
}

# ── trading_bot ───────────────────────────────────────────────────────────────
TB_PIDS=($(python_pids trading_bot.py))
printf "trading_bot     %d instance(s)  PIDs: %s\n" "${#TB_PIDS[@]}" "${TB_PIDS[*]:-none}"
if [[ ${#TB_PIDS[@]} -gt 1 ]]; then
    ISSUES=$((ISSUES + 1))
    # Keep oldest by elapsed time (highest etimes = started longest ago)
    KEEP=$(ps -p ${TB_PIDS[@]} -o pid=,etimes= 2>/dev/null \
           | sort -k2 -rn | head -1 | awk '{print $1}')
    KILL_LIST=($(printf '%s\n' "${TB_PIDS[@]}" | grep -v "^${KEEP}$"))
    echo "  !! DUPLICATE — keep $KEEP (oldest), kill: ${KILL_LIST[*]}"
    if $FIX; then
        kill "${KILL_LIST[@]}" && echo "  -> killed ${KILL_LIST[*]}"
    else
        echo "     (trading_bot is NOT auto-restarted; kill duplicates manually)"
    fi
elif [[ ${#TB_PIDS[@]} -eq 0 ]]; then
    echo "  -- not running (start manually when ready)"
fi

# ── screen-managed monitor helper ─────────────────────────────────────────────
check_monitor() {
    local name=$1 script=$2
    local PIDS SESSION status

    PIDS=($(python_pids "$script"))
    SESSION=$(/usr/bin/screen -list 2>/dev/null \
              | grep -o '[0-9]*\.'"$name" | head -1 | cut -d. -f1)

    printf "%-15s %d instance(s)  PIDs: %-20s screen: %s\n" \
        "$name" "${#PIDS[@]}" "${PIDS[*]:-none}" "${SESSION:-none}"

    if [[ ${#PIDS[@]} -gt 1 ]]; then
        ISSUES=$((ISSUES + 1))
        echo "  !! DUPLICATE"
        if $FIX; then
            kill "${PIDS[@]}" 2>/dev/null
            /usr/bin/screen -wipe >/dev/null 2>&1
            sleep 1
            "$REPO/scripts/start_${name}.sh"
            echo "  -> killed duplicates and restarted $name"
        fi
    elif [[ ${#PIDS[@]} -eq 0 && -n "$SESSION" ]]; then
        ISSUES=$((ISSUES + 1))
        echo "  !! DEAD screen session (Python process gone)"
        if $FIX; then
            /usr/bin/screen -wipe >/dev/null 2>&1
            sleep 1
            "$REPO/scripts/start_${name}.sh"
            echo "  -> wiped dead session and restarted $name"
        fi
    elif [[ ${#PIDS[@]} -eq 0 ]]; then
        ISSUES=$((ISSUES + 1))
        echo "  !! DOWN (not running, no screen session)"
        if $FIX; then
            "$REPO/scripts/start_${name}.sh"
            echo "  -> started $name"
        fi
    elif [[ -z "$SESSION" ]]; then
        ISSUES=$((ISSUES + 1))
        echo "  !! ORPHAN — Python running but not inside a screen session"
        if $FIX; then
            kill "${PIDS[@]}" 2>/dev/null
            sleep 1
            "$REPO/scripts/start_${name}.sh"
            echo "  -> killed orphan and restarted inside screen"
        fi
    fi
}

check_monitor mes_monitor   mes_monitor.py

# ── bar_collector (launchd-managed, informational only) ───────────────────────
BC_PIDS=($(python_pids bar_collector.py))
printf "bar_collector   %d instance(s)  PIDs: %s\n" "${#BC_PIDS[@]}" "${BC_PIDS[*]:-none}"
if [[ ${#BC_PIDS[@]} -gt 1 ]]; then
    ISSUES=$((ISSUES + 1))
    echo "  !! DUPLICATE — managed by launchd; do not kill manually."
    echo "     Run: launchctl unload ~/Library/LaunchAgents/com.mf-trade-bot.bar-collector.plist"
    echo "     Then: launchctl load  ~/Library/LaunchAgents/com.mf-trade-bot.bar-collector.plist"
elif [[ ${#BC_PIDS[@]} -eq 0 ]]; then
    ISSUES=$((ISSUES + 1))
    echo "  !! DOWN — check: launchctl list | grep mf-trade-bot"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
if [[ $ISSUES -eq 0 ]]; then
    echo "All clean — no issues found."
else
    printf "%d issue(s) found.\n" "$ISSUES"
    if ! $FIX; then
        echo "Run with --fix to automatically resolve (except trading_bot)."
    fi
fi
echo ""
