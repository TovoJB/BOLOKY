#!/bin/bash

# Function to clean up background processes on exit
cleanup() {
    echo ""
    echo "🛑 Arrêt des serveurs (Backend & Frontend)..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# 1. Démarrage du Backend
echo "🚀 Démarrage du Backend (FastAPI)..."
cd backend
python3 main.py &
BACKEND_PID=$!

# 2. Démarrage du Frontend
echo "🚀 Démarrage du Frontend (Next.js)..."
cd ../translation-ui
npm run dev &
FRONTEND_PID=$!

# Attendre que les deux processus se terminent
wait
