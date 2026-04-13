# pest-detection-korean-subset

노지 작물 해충 진단 - **썩덩나무노린재 + 정상** 2클래스 서브셋 파인튜닝

## 시작하기 전에 — 환경변수 설정

RunPod 대시보드 → Pod 설정 → **Environment Variables**에 아래 값을 입력하세요.

| 환경변수 | 필수 | 값 | 설명 |
|---|---|---|---|
| `HF_TOKEN` | 선택 | `hf_xxx...` | HuggingFace Hub 업로드 시 필요. [발급](https://huggingface.co/settings/tokens) |
| `WANDB_API_KEY` | 선택 | `xxx...` | W&B 실험 추적. 미설정 시 W&B 없이 학습. [발급](https://wandb.ai/authorize) |
| `DISCORD_WEBHOOK_URL` | 선택 | `https://discord.com/api/webhooks/...` | 디스코드 알림. 미설정 시 알림 없이 진행 |

그리고 코드에서 **한 곳만 직접 수정**이 필요합니다:

| 파일 | 변경할 값 | 설명 |
|---|---|---|
| `train.py` / `pest_detection.ipynb` | `YOUR_REPO_NAME` → 실제 레포 이름 | HuggingFace Hub 업로드 대상 (예: `팀이름/모델이름`) |

---

## 개요

Qwen3.5-9B 비전-언어 모델을 LoRA로 파인튜닝하여, 작물 사진에서 **썩덩나무노린재** 해충을 판별하거나 **정상**으로 분류합니다.

| 항목 | 내용 |
|---|---|
| 모델 | `unsloth/Qwen3.5-9B` (bf16 LoRA) |
| 클래스 | 썩덩나무노린재, 정상 |
| 데이터 | Train 1,000건 / Val 200건 / Test 200건 |
| 환경 | 32GB+ VRAM (A5000/A6000) |
| 실험 추적 | Weights & Biases (wandb) — 선택 |
| 알림 | Discord Webhook (Embed) — 선택 |

## 빠른 시작 (RunPod / Vast.ai)

### 1. 레포 클론

```bash
git clone https://github.com/JunePark2018/pest-detection-korean-subset.git
cd pest-detection-korean-subset
```

### 2. 환경 설정

```bash
bash setup.sh
```

이 스크립트가 하는 일:
- pip 패키지 설치 (unsloth, trl, wandb 등)
- Unsloth 설치 확인
- W&B 활성화 확인 (`WANDB_API_KEY` 설정 시)
- `/workspace/data/` 데이터셋 존재 여부 확인

### 3. 학습

```bash
python train.py
```

학습이 시작되면 W&B 대시보드 링크가 터미널에 출력됩니다. 브라우저에서 열어 실시간으로 loss를 확인하세요.

체크포인트가 있으면 자동으로 이어서 학습합니다 (`resume_from_checkpoint=True`).

### 4. 평가

```bash
python evaluate.py --model pest-detector-lora
```

test 데이터셋 200건으로 Accuracy, Precision, Recall, F1, Confusion Matrix를 출력합니다.

### 5. 추론

```bash
python inference.py --image test.jpg --model pest-detector-lora
```

## 프로젝트 구조

```
.
├── train.py                 # 전처리 + 학습 + 저장 (올인원 스크립트)
├── evaluate.py              # 학습 후 test 데이터셋으로 성능 평가
├── inference.py             # 학습된 모델로 추론
├── pest_detection.ipynb     # 노트북 버전 (단계별 실행)
├── setup.sh                 # GPU 서버 초기 설정
├── requirements.txt         # 의존성
└── data/                    # 데이터셋
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    ├── train/
    │   ├── 썩덩나무노린재/   # 이미지 + bbox JSON
    │   └── 정상/            # 이미지만
    ├── val/
    │   ├── 썩덩나무노린재/
    │   └── 정상/
    └── test/
        ├── 썩덩나무노린재/
        └── 정상/
```

## 데이터셋 구조

각 데이터 샘플은 JSONL 형식의 대화 구조입니다:

```json
{
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "당신은 작물 해충 식별 전문가입니다..."}]},
    {"role": "user", "content": [
      {"type": "image", "image": "train/썩덩나무노린재/xxx.jpg"},
      {"type": "text", "text": "이 사진에 있는 해충의 이름을 알려주세요."}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "썩덩나무노린재"}]}
  ]
}
```

- **썩덩나무노린재** 이미지에는 `.jpg.json` 파일이 함께 있으며, 바운딩 박스 좌표(`xtl`, `ytl`, `xbr`, `ybr`)가 포함되어 있습니다.
- **정상** 이미지에는 bbox JSON이 없습니다.
- 전처리 시 해충 이미지는 원본(50%) / bbox tight 크롭(25%) / bbox context 크롭(25%) 비율로 적용됩니다.

## 주요 하이퍼파라미터

| 파라미터 | 값 |
|---|---|
| LoRA r | 16 |
| LoRA alpha | 16 |
| Batch size | 6 (effective 12) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Optimizer | AdamW 8bit |
| Scheduler | Linear |

## Discord 알림

`DISCORD_WEBHOOK_URL`이 설정되면, 각 단계의 시작/완료/에러를 디스코드 채널에 Embed 형식으로 전송합니다. 학습/추론/에러 단계에는 `@everyone` 멘션이 포함됩니다.

| 스크립트 | 단계 | 내용 |
|---|---|---|
| train.py | [1/9]~[9/9] | 데이터 확인 → 전처리 → 로딩 → 모델 → LoRA → 학습 → 저장 → 추론 → 업로드 |
| 노트북 | [1/10]~[10/10] | 패키지 설치 포함, 나머지 동일 |
| evaluate.py | [1/3]~[3/3] | 모델 로딩 → 추론 → 결과 집계 |

웹훅 URL은 디스코드 채널 설정 → 연동 → 웹후크에서 생성할 수 있습니다.

## W&B (Weights & Biases)

`WANDB_API_KEY`가 설정되면 자동으로 W&B에 기록되고, 없으면 W&B 없이 학습됩니다.

- 대시보드에서 `train/loss`, `eval/loss`, `learning_rate` 등을 실시간 확인
- run name: `pest-subset-qwen3.5`
