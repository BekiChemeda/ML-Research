#!/usr/bin/env bash
# Living Amharic Agent Launcher
cd /workspace/ML-Research/code

API_ID="33704259"
API_HASH="fe8e18a31b64a14e633d85594e8bf1cb"
BOT_TOKEN="$1"

echo "======================================================================"
echo "🌟 LAUNCHING LIVING AMHARIC AGENT: AMAN (አማን)"
echo "======================================================================"
echo "API_ID: $API_ID"
echo "Listening to channels: tikvahethiopia, bbcnewsamharic, fana_broadcast, alainamharic"
echo "======================================================================"

if [ -z "$BOT_TOKEN" ]; then
    /venv/main/bin/python autonomous_human_mamba.py \
        --api_id "$API_ID" \
        --api_hash "$API_HASH" \
        --channels tikvahethiopia bbcnewsamharic fana_broadcast alainamharic
else
    /venv/main/bin/python autonomous_human_mamba.py \
        --api_id "$API_ID" \
        --api_hash "$API_HASH" \
        --bot_token "$BOT_TOKEN" \
        --channels tikvahethiopia bbcnewsamharic fana_broadcast alainamharic
fi
