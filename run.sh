#!/bin/bash
# ML Pipeline Dashboard Runner
# Usage: ./run.sh [html|api|all]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-8500}
API_PORT=${API_PORT:-8000}
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

cd "$SCRIPT_DIR"

case "${1:-html}" in
    html)
        echo "Opening dashboard..."
        xdg-open "$SCRIPT_DIR/dashboard.html" 2>/dev/null || echo "Open manually: file://$SCRIPT_DIR/dashboard.html"
        python3 -m http.server $PORT --directory "$SCRIPT_DIR"
        ;;
    api)
        echo "Starting FastAPI on http://localhost:$API_PORT"
        $VENV_PYTHON -m uvicorn api:app --host 0.0.0.0 --port $API_PORT
        ;;
    all)
        echo "=== ML Pipeline Dashboard ==="
        echo "HTML Dashboard: http://localhost:$PORT/dashboard.html"
        echo "API docs:       http://localhost:$API_PORT/docs"
        echo "Streamlit App:  http://localhost:$STREAMLIT_PORT"
        echo ""
        python3 -m http.server $PORT --directory "$SCRIPT_DIR" &
        $VENV_PYTHON -m uvicorn api:app --host 0.0.0.0 --port $API_PORT &
        $VENV_PYTHON -m streamlit run app_combined.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0
        ;;
    streamlit)
        echo "Starting Streamlit on http://localhost:$STREAMLIT_PORT"
        $VENV_PYTHON -m streamlit run app_combined.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0
        ;;
    *)
        echo "Usage: $0 {html|api|all|streamlit}"
        echo "  html      - Dashboard (HTML)"
        echo "  api       - FastAPI server"
        echo "  streamlit - Streamlit Dashboard"
        echo "  all       - All services"
        exit 1
        ;;
esac