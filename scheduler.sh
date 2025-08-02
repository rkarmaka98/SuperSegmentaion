#!/bin/bash
PID=45016
START_TIME=$(date +%s)


while kill -0 $PID 2>/dev/null; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo "[INFO] Waiting for PID $PID to finish..."
    sleep 1000
done
echo "[DONE] ✅ Process ended after $ELAPSED seconds"
echo "[NEXT] 🚀 Starting second script"
cd /home/puser1/Documents/pytorch-superpoint ; /usr/bin/env /usr/bin/python3 /home/puser1/.vscode-server/extensions/ms-python.debugpy-2025.10.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 51773 -- /home/puser1/Documents/pytorch-superpoint/train4.py train_joint configs/superpoint_cityscapes_finetune.yaml superpoint_cityscapes --eval --debug
echo "[COMPLETE] 🎉 Second script finished"