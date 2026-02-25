#!/usr/bin/env bash
# Install the 24/7 trader as a macOS LaunchAgent (runs every hour, survives reboot).
# Run from project root: ./install_24_7.sh

set -e
cd "$(dirname "$0")"
PROJECT_DIR="$PWD"
PLIST_NAME="com.btcomega.trader.plist"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
DEST_PLIST="$LAUNCH_AGENTS/$PLIST_NAME"

# Ensure plist points to this project
if [ ! -f "$PLIST_NAME" ]; then
  echo "Missing $PLIST_NAME. Run this from btc-omega-vi project root."
  exit 1
fi

# Update plist with this project path (in case repo was moved)
sed -i.bak "s|/Users/bollineni/btc-omega-vi|$PROJECT_DIR|g" "$PLIST_NAME"
rm -f "${PLIST_NAME}.bak" 2>/dev/null || true

mkdir -p "$LAUNCH_AGENTS"
cp "$PLIST_NAME" "$DEST_PLIST"
echo "Installed: $DEST_PLIST"

# Unload first if already loaded
launchctl unload "$DEST_PLIST" 2>/dev/null || true
launchctl load "$DEST_PLIST"
echo "Loaded. Trader will run every 60 minutes (and once at load)."
echo ""
echo "Commands:"
echo "  Start:   launchctl load $DEST_PLIST"
echo "  Stop:    launchctl unload $DEST_PLIST"
echo "  Log:     tail -f $PROJECT_DIR/trader.log"
echo ""
echo "Set BTCOmega_MODE and Coinbase keys in .env for live trading."
