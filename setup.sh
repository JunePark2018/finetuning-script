#!/bin/bash
# RunPod / Vast.ai 환경 초기 설정 스크립트
# 사용법: bash setup.sh

set -e

echo "=== 패키지 설치 ==="
pip install --upgrade pip
pip install --upgrade typing_extensions
pip install unsloth
pip install "transformers>=5.2"
pip install trl==0.22.2 datasets Pillow accelerate scikit-learn huggingface_hub

echo ""
echo "=== Unsloth 설치 확인 ==="
python -c "from unsloth import FastVisionModel; print('Unsloth OK')"

echo ""
echo "=== 데이터셋 다운로드 ==="
if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN이 설정되지 않았습니다."
    echo "사용법: HF_TOKEN=hf_xxx bash setup.sh"
    exit 1
fi

python -c "
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ['HF_TOKEN'])
snapshot_download(
    'Himedia-AI-01/pest-detection-korean',
    repo_type='dataset',
    local_dir='/workspace/data',
    max_workers=2,
)
print('데이터셋 다운로드 완료: /workspace/data')
"

echo ""
echo "=== 설정 완료 ==="
echo "학습 시작: python train.py"
