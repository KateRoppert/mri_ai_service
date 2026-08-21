#!/usr/bin/env bash
# Start/stop the stack on barguzin (shared GPU workstation) for remote runs.
#
# Wraps the two-file compose invocation so nobody has to remember it:
# docker-compose.barguzin.yml is NOT auto-loaded (that name is reserved for
# docker-compose.override.yml, and auto-loading it would make every machine
# start the stack on boot).
#
#   ./scripts/barguzin.sh up       — build nothing, just start
#   ./scripts/barguzin.sh build    — rebuild images, then start
#   ./scripts/barguzin.sh down     — stop everything
#   ./scripts/barguzin.sh status   — what is running, and is the GPU visible
#   ./scripts/barguzin.sh logs     — follow web logs
#
# Users connect from their own machine over SSH:
#   ssh -L 8000:localhost:8000 -p 8819 e.roppert@bigdata.nsu.ru
#   then open http://localhost:8000

set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.barguzin.yml --profile full)

case "${1:-up}" in
  up)
    "${COMPOSE[@]}" up -d
    echo
    echo "Стек поднят. С своей машины:"
    echo "  ssh -L 8000:localhost:8000 -p 8819 e.roppert@bigdata.nsu.ru"
    echo "  затем откройте http://localhost:8000"
    ;;
  build)
    # Full build takes ~40-60 min (torch, nnUNet, HD-BET). Run under tmux.
    "${COMPOSE[@]}" build
    "${COMPOSE[@]}" up -d
    ;;
  down)
    "${COMPOSE[@]}" down
    ;;
  status)
    "${COMPOSE[@]}" ps
    echo
    echo "--- GPU ---"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || echo "nvidia-smi недоступен"
    echo
    echo "--- модель сегментации ---"
    docker logs mri_ai_service-service-gbm-seg-1 2>&1 | grep -E "model preset|model ready" | tail -2 \
      || echo "(сервис не запущен)"
    ;;
  logs)
    "${COMPOSE[@]}" logs -f web
    ;;
  *)
    echo "Использование: $0 {up|build|down|status|logs}" >&2
    exit 1
    ;;
esac
