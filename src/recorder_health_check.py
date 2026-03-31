"""
Watchdog health check for DOM and bar recorders.

Scheduled via launchd to run 3× daily. Detects:
  - bar_recorder.py not running  → restarts via launchctl
  - dom_client.py not running    → notifies (bash wrapper should restart it)
  - dom_client.py stuck in login-retry loop (running but no data saved
    for > STUCK_THRESHOLD seconds during market hours) → kills the Python
    process so the bash wrapper (run_dom_recorder.sh) restarts it fresh.

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
DOM_LOG        = REPO / "logs" / "dom_recorder.log"
ET             = ZoneInfo("America/New_York")
STUCK_HOURS    = 1.5   # DOM has no save for this long during market hours → stuck


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
    """True on weekdays 08:00–22:00 ET (covers RTH + most Globex)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 8 * 60 <= mins < 22 * 60


def dom_last_save_age_seconds() -> float | None:
    """
    Scan the last 1000 lines of dom_recorder.log for the most recent
    'saved →' entry and return its age in seconds. Returns None if not found.
    """
    if not DOM_LOG.exists():
        return None
    try:
        lines = DOM_LOG.read_text(errors="replace").splitlines()
        for line in reversed(lines[-1000:]):
            if "saved →" in line:
                # Line: [recorder] 2026-03-29T18:15:01.123456+00:00  last=...
                parts = line.split("]", 1)
                if len(parts) < 2:
                    continue
                ts_str = parts[1].strip().split()[0]
                ts = datetime.fromisoformat(ts_str)
                return (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception as e:
        log(f"  dom_last_save_age parse error: {e}")
    return None


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


def check_dom_recorder() -> bool:
    pids = pgrep("dom_client.py")
    age  = dom_last_save_age_seconds()
    age_str = f"{age/3600:.1f}h" if age is not None else "unknown"

    if not pids:
        # The bash wrapper (launchd-managed) should restart it automatically.
        # Just alert so we know.
        log(f"dom_client    MISSING  (last save: {age_str} ago)")
        notify("mf-trade-bot watchdog", f"dom_client.py not running — check logs (last save {age_str} ago)")
        return False

    # Process exists — check if it's actually saving data
    stuck = (
        age is not None
        and age > STUCK_HOURS * 3600
        and is_trading_hours()
    )
    if stuck:
        log(f"dom_client    STUCK  (PID={pids[0]}, last save {age_str} ago) — killing for restart")
        notify("mf-trade-bot watchdog", f"DOM recorder stuck ({age_str} without data) — restarting")
        for pid in pids:
            subprocess.run(["kill", str(pid)])
        return False

    log(f"dom_client    OK  (PID={pids[0]}, last save {age_str} ago)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("── health check start ──────────────────────────────────────────")
    bar_ok = check_bar_recorder()
    dom_ok = check_dom_recorder()
    overall = "OK" if (bar_ok and dom_ok) else "ACTION TAKEN"
    log(f"── health check done  ({overall}) ──────────────────────────────")
    sys.exit(0 if (bar_ok and dom_ok) else 1)
