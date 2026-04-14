#!/bin/bash
# RunPod / Vast.ai 환경 초기 설정 스크립트
# 사용법: bash setup.sh

set -e

echo "=== 패키지 설치 ==="
pip install --upgrade pip
pip install --upgrade typing_extensions
pip install unsloth
pip install "transformers>=5.2"
pip install trl==0.22.2 datasets Pillow accelerate scikit-learn huggingface_hub wandb requests

echo ""
echo "=== Unsloth 설치 확인 ==="
python -c "from unsloth import FastVisionModel; print('Unsloth OK')"

echo ""
echo "=== W&B 로그인 (선택) ==="
if [ -n "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY 감지 — W&B 활성화"
else
    echo "WANDB_API_KEY 미설정 — W&B 없이 진행합니다."
    echo "사용하려면: export WANDB_API_KEY=your_key"
fi

echo ""
echo "=== 데이터셋 확인 ==="
DATA_DIR="${DATA_DIR:-data}"   # 환경변수 미설정 시 기본 ./data (repo 루트 기준)
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "$DATA_DIR/train.jsonl이 없습니다."
    echo "data/ 폴더가 repo 루트에 있는지 확인하거나, DATA_DIR 환경변수로 다른 경로를 지정하세요."
    exit 1
fi
echo "데이터셋 확인 완료: $DATA_DIR/"

echo ""
echo "=== 설정 완료 ==="
echo "학습 시작: python train.py"
