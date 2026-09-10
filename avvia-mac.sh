#!/bin/bash
# Portfolio Optimizer — Avvio per Mac/Linux
# Doppio clic sul file oppure: bash avvia-mac.sh

DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  Avvio Portfolio Optimizer..."
echo ""

# Controlla Python
if ! command -v python3 &>/dev/null; then
    echo "  ERRORE: Python 3 non trovato."
    echo "  Su Mac: installa da https://www.python.org/downloads/"
    echo "  Su Linux: sudo apt install python3"
    exit 1
fi

# Avvia server in background
python3 "$DIR/server.py" &
SERVER_PID=$!

sleep 1.5

# Apri il browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$DIR/portfolio-markowitz.html"
else
    xdg-open "$DIR/portfolio-markowitz.html" 2>/dev/null || \
    sensible-browser "$DIR/portfolio-markowitz.html" 2>/dev/null
fi

echo "  Server avviato (PID: $SERVER_PID)"
echo "  Lascia aperto questo terminale mentre usi l'app."
echo "  Premi Ctrl+C per fermare il server."
echo ""

# Aspetta il server
wait $SERVER_PID
