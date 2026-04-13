#!/bin/bash
# RunPod / Vast.ai 환경 초기 설정 스크립트
# 사용법: bash setup.sh

set -e

echo "=== 패키지 설치 ==="
pip install --upgrade pip
pip install --upgrade typing_extensions
pip install unsloth
pip install "transformers>=5.2"
pip install trl==0.22.2 datasets Pillow accelerate scikit-learn huggingface_hub wandb

echo ""
echo "=== Unsloth 설치 확인 ==="
python -c "from unsloth import FastVisionModel; print('Unsloth OK')"

echo ""
echo "=== W&B 로그인 ==="
wandb login

echo ""
echo "=== 데이터셋 확인 ==="
if [ ! -f "data/train.jsonl" ]; then
    echo "data/train.jsonl이 없습니다."
    echo "data/ 폴더에 데이터셋을 먼저 배치해주세요."
    exit 1
fi
echo "데이터셋 확인 완료: data/"

echo ""
echo "=== 설정 완료 ==="
echo "학습 시작: python train.py"
