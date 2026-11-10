#!/usr/bin/env bash
# Despliegue del LLM ligero para el boletín del visor (backend OpenAI-compatible).
#
# Uso:
#   bash scripts/serve_llm.sh            # vLLM (GPU, recomendado)
#   LLM_MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/serve_llm.sh   # aún más ligero
#
# Una vez arriba el servidor en :8002, arranca el visor con el backend conectado:
#   LLM_BASE_URL=http://localhost:8002/v1 uv run uvicorn ardintra.api.mini_api:app --port 8001
# y el botón 📋 Boletín generará prosa operativa con el LLM local (fallback
# determinista si el servidor no responde).
set -euo pipefail

MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"   # 7B AWQ (~5.6 GB VRAM, buen español); 3B para más velocidad
PORT="${LLM_PORT:-8002}"
CTX="${LLM_CTX:-8192}"

# venv dedicado (vLLM es pesado: torch+cuda ~6 GB de disco; no contamina el venv del proyecto)
if [ ! -x ".venv-llm/bin/vllm" ]; then
  echo "→ creando .venv-llm e instalando vLLM (una vez, varios GB)…"
  uv venv .venv-llm --python 3.12
  uv pip install --python .venv-llm/bin/python vllm
fi

echo "→ sirviendo $MODEL en :$PORT (Ctrl-C para parar)"
echo "  luego: LLM_BASE_URL=http://localhost:$PORT/v1 uv run uvicorn ardintra.api.mini_api:app --port 8001"
exec .venv-llm/bin/vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$CTX" \
  --gpu-memory-utilization 0.85
