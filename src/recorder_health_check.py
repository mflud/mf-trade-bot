"""
Watchdog health check for DOM and bar recorders.

Scheduled via launchd to run 3× daily. Detects:
  - bar_recorder.py not running  → restarts via launchctl
  - data/dom.db not updated recently during market hours → notifies
    (DOM is now written by bar_collector, not a separate dom_client process)

Logs to logs/watchdog.log and sends macOS notifications on any problem.

Usage:
  python src/recorder_health_check.py
"""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO           = Path(__file__).resolve().parent.parent
LOG_PATH       = REPO / "logs" / "watchdog.log"
DOM_DB         = REPO / "data" / "dom.db"
ET             = ZoneInfo("America/New_York")
DOM_STALE_SECS = 60    # dom.db not updated for this long during market hours → alert


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[watchdog] {ts}  {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def notify(title: str, body: str):
    """macOS notification via osascript."""
    script = f'display notification "{body}" with title "{title}" sound name "Basso"'
    subprocess.run(["osascript", "-e", script], capture_output=True)
    log(f"  → notified: {body}")


def pgrep(pattern: str) -> list[int]:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(p) for p in r.stdout.strip().split() if p.strip().isdigit()]


def is_trading_hours() -> bool:
    """True on weekdays 09:00–17:00 ET (active session only)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 9 * 60 <= mins < 17 * 60


def dom_db_age_seconds() -> float | None:
    """Return seconds since dom.db was last modified, or None if it doesn't exist."""
    if not DOM_DB.exists():
        return None
    return datetime.now(timezone.utc).timestamp() - DOM_DB.stat().st_mtime


# ── Checks ────────────────────────────────────────────────────────────────────

def check_bar_recorder() -> bool:
    pids = pgrep("bar_recorder.py")
    if pids:
        log(f"bar_recorder  OK  (PID={pids[0]})")
        return True
    log("bar_recorder  MISSING — restarting via launchctl")
    notify("mf-trade-bot watchdog", "bar_recorder.py was not running — restarting now")
    subprocess.run([
        "launchctl", "kickstart", "-k",
        f"gui/{os.getuid()}/com.mf-trade-bot.bar-recorder",
    ])
    return False


def check_dom_db() -> bool:
    """Check that dom.db (written by bar_collector) is being updated regularly."""
    age = dom_db_age_seconds()
    if age is None:
        if is_trading_hours():
            log("dom.db        MISSING during market hours")
            notify("mf-trade-bot watchdog", "data/dom.db not found — bar_collector may be down")
            return False
        log("dom.db        not found (outside market hours — OK)")
        return True

    age_str = f"{age:.0f}s"
    if age > DOM_STALE_SECS and is_trading_hours():
        log(f"dom.db        STALE  (last updated {age_str} ago) — bar_collector may be stuck")
        notify("mf-trade-bot watchdog", f"dom.db stale ({age_str}) — bar_collector may need restart")
        return False

    log(f"dom.db        OK  (last updated {age_str} ago, written by bar_collector)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("── health check start ──────────────────────────────────────────")
    bar_ok = check_bar_recorder()
    dom_ok = check_dom_db()
    overall = "OK" if (bar_ok and dom_ok) else "ACTION TAKEN"
    log(f"── health check done  ({overall}) ──────────────────────────────")
    sys.exit(0 if (bar_ok and dom_ok) else 1)
