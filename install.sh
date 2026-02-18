#!/usr/bin/env bash
# JarvisChat Installer
# Run as root on jarvis: bash install.sh

set -e

APP_DIR="/opt/jarvischat"
SERVICE_FILE="/etc/systemd/system/jarvischat.service"

echo "=== JarvisChat Installer ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[+] Found Python $PYTHON_VERSION"

# Create app directory
echo "[+] Creating $APP_DIR..."
mkdir -p "$APP_DIR"

# Copy files
echo "[+] Copying application files..."
cp app.py "$APP_DIR/"
cp requirements.txt "$APP_DIR/"

# Create virtual environment
echo "[+] Creating virtual environment..."
cd "$APP_DIR"
python3 -m venv venv

# Install dependencies
echo "[+] Installing dependencies..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# Install systemd service
echo "[+] Installing systemd service..."
cp "$OLDPWD/jarvischat.service" "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable jarvischat
systemctl start jarvischat

echo ""
echo "=== Installation Complete ==="
echo ""
echo "JarvisChat is running on http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "Commands:"
echo "  systemctl status jarvischat    # Check status"
echo "  systemctl restart jarvischat   # Restart"
echo "  journalctl -u jarvischat -f    # View logs"
echo ""
echo "Database stored at: $APP_DIR/jarvischat.db"
