#!/bin/bash
set -e

# 可选：通过 SAM3_START_CMD 环境变量启动 SAM3 服务，例如：
# export SAM3_START_CMD='cd /path/to/sam3-github && conda run -n sam3 python sam3_server.py'
if [ -n "${SAM3_START_CMD}" ]; then
  echo "Starting SAM3 service with SAM3_START_CMD..."
  bash -lc "${SAM3_START_CMD}" &
  SAM3_PID=$!
  echo "SAM3 service PID: ${SAM3_PID}"
else
  # 若未指定 SAM3_START_CMD，则尝试调用本仓库自带的 sam3 启动脚本
  SAM3_START_SCRIPT="$(dirname "$0")/sam3/sam3-github/start_sam3.sh"
  if [ -x "${SAM3_START_SCRIPT}" ]; then
    echo "Starting SAM3 service via ${SAM3_START_SCRIPT}..."
    "${SAM3_START_SCRIPT}" &
    SAM3_PID=$!
    echo "SAM3 service PID: ${SAM3_PID}"
  else
    echo "⚠️ 未设置 SAM3_START_CMD，且未找到可执行的 ${SAM3_START_SCRIPT}，将不启动 SAM3 服务。"
  fi
fi

cd "$(dirname "$0")"

# 默认使用 SAM3 HTTP 作为统一推理后端（可被外部环境变量覆盖）
if [ -z "${INFERENCE_ENGINE}" ]; then
  export INFERENCE_ENGINE=sam3_http
fi

if [ -z "${SAM3_HTTP_URL}" ]; then
  export SAM3_HTTP_URL=http://localhost:6666/sam3/segment
fi

echo "INFERENCE_ENGINE=${INFERENCE_ENGINE}"
echo "SAM3_HTTP_URL=${SAM3_HTTP_URL}"

echo "Starting auto_labling Flask app..."
bash start.sh

# Flask 退出后，尝试关闭 SAM3 服务（如果是当前脚本启动的）
if [ -n "${SAM3_PID}" ]; then
  echo "Stopping SAM3 service (PID=${SAM3_PID})..."
  kill "${SAM3_PID}" || true
fi
