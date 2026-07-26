#!/usr/bin/env bash
# barRecorderRestart.sh — Restart bar_recorder via launchctl (unload + load).
PLIST=~/Library/LaunchAgents/com.mf-trade-bot.bar-recorder.plist

echo "Unloading bar_recorder…"
launchctl unload "$PLIST"
sleep 3
echo "Loading bar_recorder…"
launchctl load "$PLIST"
echo "Done."
