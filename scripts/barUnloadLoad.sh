#!/usr/bin/env bash
# barUnloadLoad.sh — Restart bar_collector via launchctl (unload + load).
PLIST=~/Library/LaunchAgents/com.mf-trade-bot.bar-collector.plist

echo "Unloading bar_collector…"
launchctl unload "$PLIST"
sleep 3
echo "Loading bar_collector…"
launchctl load "$PLIST"
echo "Done."
