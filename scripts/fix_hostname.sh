#!/bin/bash
# Pins the Mac Studio hostname to prevent Bonjour/mDNS from auto-incrementing it.
# Installed as a LaunchDaemon — runs at every boot as root.

HOSTNAME="Mareks-Studio-Main"

/usr/sbin/scutil --set HostName      "$HOSTNAME"
/usr/sbin/scutil --set LocalHostName "$HOSTNAME"
/usr/sbin/scutil --set ComputerName  "$HOSTNAME"
