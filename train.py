"""
노지 작물 해충 진단 (서브셋) - Qwen3.5-9B LoRA 파인튜닝 스크립트
대상: 썩덩나무노린재 + 정상 (2클래스)
데이터셋: 기본 ./data (DATA_DIR 환경변수로 오버라이드 가능)
환경: 32GB+ VRAM (A5000/A6000), bf16 LoRA
"""

import json
import os
import random
import time
import requests

from PIL import Image

# ════════════════════════════════════════
# Discord Webhook 설정
# ════════════════════════════════════════

DISCORD_BOT = {
    "username": "RunPod",
    "avatar_url": "https://i.imgur.com/0HOIh4r.png",
}
DISCORD_COLOR = 12648430
DISCORD_THUMBNAIL = "https://i.imgur.com/3ClKkzk.jpeg"


def notify_discord(message):
    """DISCORD_WEBHOOK_URL이 설정되어 있으면 메시지 전송"""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": message}, timeout=10)
    except Exception as e:
        print(f"Discord 알림 실패: {e}")


def notify_discord_json(payload):
    """DISCORD_WEBHOOK_URL이 설정되어 있으면 JSON payload를 그대로 전송 (Embed 등)"""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Discord 알림 실패: {e}")


def discord_embed(description, thumbnail=False):
    """Embed payload 생성 헬퍼"""
    embed = {"description": description, "color": DISCORD_COLOR}
    if thumbnail:
        embed["thumbnail"] = {"url": DISCORD_THUMBNAIL}
    return {**DISCORD_BOT, "embeds": [embed]}


# ════════════════════════════════════════
# 하이퍼파라미터 (환경변수로 오버라이드 가능)
# ════════════════════════════════════════

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 6))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 2))

LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 16))

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 3))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 50))
MAX_STEPS = int(os.environ.get("MAX_STEPS", -1))  # > 0이면 num_train_epochs를 무시하고 step 수 직접 고정

print("=" * 60)
print("하이퍼파라미터")
print("=" * 60)
print(f"  BATCH_SIZE     = {BATCH_SIZE}")
print(f"  GRAD_ACCUM     = {GRAD_ACCUM}")
print(f"  Total Batch    = {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LORA_R         = {LORA_R}")
print(f"  LORA_ALPHA     = {LORA_ALPHA}")
print(f"  LEARNING_RATE  = {LEARNING_RATE}")
if MAX_STEPS > 0:
    print(f"  MAX_STEPS      = {MAX_STEPS} (num_train_epochs 무시됨)")
else:
    print(f"  NUM_EPOCHS     = {NUM_EPOCHS}")
print(f"  WARMUP_STEPS   = {WARMUP_STEPS}")

# 고유 run name 생성 (파라미터 조합) — MAX_STEPS 사용 시 ep → st로 표시
_epoch_or_step = f"st{MAX_STEPS}" if MAX_STEPS > 0 else f"ep{NUM_EPOCHS}"
RUN_NAME = f"r{LORA_R}_a{LORA_ALPHA}_lr{LEARNING_RATE}_bs{BATCH_SIZE}x{GRAD_ACCUM}_{_epoch_or_step}_w{WARMUP_STEPS}"
OUTPUT_DIR = f"pest-detector-{RUN_NAME}"
LORA_DIR = f"pest-lora-{RUN_NAME}"

print(f"  RUN_NAME       = {RUN_NAME}")
print(f"  OUTPUT_DIR     = {OUTPUT_DIR}")
print(f"  LORA_DIR       = {LORA_DIR}")
print("=" * 60)

# W&B project 기본값 (환경변수 미설정 시) — 모든 sweep run이 한 project에 모여 비교 가능
os.environ.setdefault("WANDB_PROJECT", "pest-detection-subset")

Image.MAX_IMAGE_PIXELS = None

# ════════════════════════════════════════
# 1. 데이터셋 경로
# ════════════════════════════════════════

print("\n[1/9] 데이터셋 경로 확인...")
notify_discord_json(discord_embed("📂 [1/9] 데이터셋 경로를 확인합니다.", thumbnail=True))
try:
    DATA_DIR = os.environ.get("DATA_DIR", "data")

    assert os.path.exists(os.path.join(DATA_DIR, "train.jsonl")), \
        f"데이터셋이 없습니다: {DATA_DIR}/train.jsonl"
    print(f"  DATA_DIR = {DATA_DIR}")
    print(f"  train.jsonl ✓")
    print(f"  val.jsonl   {'✓' if os.path.exists(os.path.join(DATA_DIR, 'val.jsonl')) else '✗'}")
    print(f"  test.jsonl  {'✓' if os.path.exists(os.path.join(DATA_DIR, 'test.jsonl')) else '✗'}")
    notify_discord_json(discord_embed("✅ [1/9] 데이터셋 경로 확인 완료. (train.jsonl, val.jsonl)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [1/9] 데이터셋 경로 확인 실패: {e}"))
    raise

# ════════════════════════════════════════
# 2. 이미지 전처리 (크롭 → 디스크 저장)
# ════════════════════════════════════════

print("\n[2/9] 이미지 전처리...")
notify_discord_json(discord_embed("🖼️ [2/9] 이미지 전처리를 시작합니다. (크롭 → 디스크 저장)"))
try:
    PROMPTS = [
        "이 사진에 있는 해충의 이름을 알려주세요.",
        "이 벌레는 무엇인가요?",
        "사진 속 해충을 식별해주세요.",
        "이 작물에 있는 해충의 종류가 무엇인가요?",
        "이 사진에서 어떤 해충이 보이나요?",
    ]

    SYSTEM_MSG = (
        "당신은 작물 해충 식별 전문가입니다. "
        '사진을 보고 해충의 이름만 한국어로 답하세요. '
        '해충이 없으면 "정상"이라고만 답하세요. '
        "부가 설명 없이 이름만 출력하세요."
    )

    BBOX_GROW_STAGE = 33


    def crop_to_bbox(img, bbox, padding_ratio=0.0):
        xtl, ytl = bbox["xtl"], bbox["ytl"]
        xbr, ybr = bbox["xbr"], bbox["ybr"]
        bw, bh = xbr - xtl, ybr - ytl
        pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
        x1 = max(0, xtl - pad_x)
        y1 = max(0, ytl - pad_y)
        x2 = min(img.width, xbr + pad_x)
        y2 = min(img.height, ybr + pad_y)
        return img.crop((x1, y1, x2, y2))


    def find_label_json(split, class_name, img_filename):
        json_path = os.path.join(DATA_DIR, split, class_name, img_filename + ".json")
        if not os.path.exists(json_path):
            return None
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for obj in data["annotations"]["object"]:
            if obj["grow"] == BBOX_GROW_STAGE and obj.get("points"):
                return obj["points"][0]
        return None


    def preprocess_split(split="train"):
        """원본 이미지를 크롭하여 디스크에 저장하고 새 JSONL 생성"""
        jsonl_path = os.path.join(DATA_DIR, f"{split}.jsonl")
        out_dir = os.path.join(DATA_DIR, f"{split}_cropped")
        out_jsonl = os.path.join(DATA_DIR, f"{split}_cropped.jsonl")

        if os.path.exists(out_jsonl):
            with open(out_jsonl, "r") as f:
                count = sum(1 for _ in f)
            print(f"  [{split}] 이미 전처리 완료: {count}건 (캐시 사용)")
            return count

        with open(jsonl_path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        print(f"  [{split}] 전처리 시작: {total}건")

        out_file = open(out_jsonl, "w", encoding="utf-8")
        count = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if (i + 1) % 500 == 0 or (i + 1) == total:
                    print(f"\r  [{split}] {i + 1}/{total} ({(i + 1) * 100 // total}%)", end="", flush=True)

                record = json.loads(line)
                messages = record["messages"]
                label = messages[-1]["content"][0]["text"]

                img_rel_path = None
                for msg in messages:
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            img_rel_path = content["image"].replace("\\", "/")
                            break

                if img_rel_path is None:
                    continue

                parts = img_rel_path.split("/")
                class_name = parts[1]
                img_filename = parts[2]

                class_dir = os.path.join(out_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                base_name = os.path.splitext(img_filename)[0]
                out_filename = f"{base_name}.jpg"
                out_path = os.path.join(class_dir, out_filename)
                out_rel_path = f"{split}_cropped/{class_name}/{out_filename}"

                img_path = os.path.join(DATA_DIR, img_rel_path)
                img = Image.open(img_path).convert("RGB")

                if label == "정상":
                    result = img
                else:
                    bbox = find_label_json(split, class_name, img_filename)
                    if bbox:
                        r = random.random()
                        if r < 0.5:
                            result = img
                        elif r < 0.75:
                            result = crop_to_bbox(img, bbox, padding_ratio=0.0)
                        else:
                            result = crop_to_bbox(img, bbox, padding_ratio=0.5)
                    else:
                        result = img

                result.save(out_path, "JPEG", quality=95)
                if result is not img:
                    result.close()
                img.close()

                new_record = {
                    "messages": [
                        {"role": "system", "content": [
                            {"type": "text", "text": SYSTEM_MSG}
                        ]},
                        {"role": "user", "content": [
                            {"type": "image", "image": out_rel_path},
                            {"type": "text", "text": random.choice(PROMPTS)},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": label}
                        ]},
                    ]
                }
                out_file.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                count += 1

        out_file.close()
        print(f"\n  [{split}] 완료: {count}건 → {out_dir}")
        return count


    random.seed(42)
    t0 = time.time()
    num_train = preprocess_split("train")
    num_val = preprocess_split("val")
    print(f"  전처리 소요 시간: {time.time() - t0:.1f}s")
    notify_discord_json(discord_embed(f"✅ [2/9] 전처리 완료! (train {num_train}건, val {num_val}건)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [2/9] 이미지 전처리 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 3. 데이터 로딩 (경로 기반 — RAM 절약)
# ════════════════════════════════════════

print("\n[3/9] 데이터 로딩...")
notify_discord_json(discord_embed("📊 [3/9] 데이터를 로딩합니다."))
try:
    def load_dataset_from_cropped_jsonl(split="train"):
        jsonl_path = os.path.join(DATA_DIR, f"{split}_cropped.jsonl")
        dataset = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for msg in record["messages"]:
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            content["image"] = os.path.join(DATA_DIR, content["image"])
                dataset.append(record)
        random.shuffle(dataset)
        return dataset

    random.seed(42)
    t0 = time.time()
    train_dataset = load_dataset_from_cropped_jsonl("train")
    val_dataset = load_dataset_from_cropped_jsonl("val")
    print(f"  Train: {len(train_dataset)}건, Val: {len(val_dataset)}건")
    print(f"  로딩 소요 시간: {time.time() - t0:.1f}s")
    notify_discord_json(discord_embed(f"✅ [3/9] 데이터 로딩 완료! (Train {len(train_dataset)}건, Val {len(val_dataset)}건)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [3/9] 데이터 로딩 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 4. 모델 로딩
# ════════════════════════════════════════

print("\n[4/9] 모델 로딩...")
notify_discord_json(discord_embed("🤖 [4/9] Qwen3.5-9B 모델을 로딩합니다."))
try:
    import torch
    from unsloth import FastVisionModel

    os.environ["HF_HOME"] = "/workspace/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

    print(f"  HF_HOME = {os.environ['HF_HOME']}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    t0 = time.time()
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3.5-9B",
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    print(f"  모델 로딩 소요 시간: {time.time() - t0:.1f}s")
    notify_discord_json(discord_embed("✅ [4/9] 모델 로딩 완료!"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [4/9] 모델 로딩 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 5. LoRA 설정
# ════════════════════════════════════════

print(f"\n[5/9] LoRA 설정 (r={LORA_R}, alpha={LORA_ALPHA})...")
notify_discord_json(discord_embed(f"⚙️ [5/9] LoRA 어댑터를 설정합니다. (r={LORA_R}, alpha={LORA_ALPHA})"))
try:
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  학습 가능 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    notify_discord_json(discord_embed("✅ [5/9] LoRA 설정 완료!"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [5/9] LoRA 설정 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 6. 학습
# ════════════════════════════════════════

_train_len = f"{MAX_STEPS} steps" if MAX_STEPS > 0 else f"{NUM_EPOCHS} epochs"
print(f"\n[6/9] 학습 시작 ({_train_len}, batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, lr={LEARNING_RATE})...")
notify_discord_json(discord_embed(f"@everyone\n🚀 [6/9] 학습을 시작합니다! ({_train_len}, batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM})"))
try:
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            max_steps=MAX_STEPS,  # -1 (기본): epoch 사용 / > 0: epoch 무시하고 step 수 직접 제어
            learning_rate=LEARNING_RATE,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=OUTPUT_DIR,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
            run_name=RUN_NAME,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
        ),
    )

    # 체크포인트 존재 확인 → resume 여부 결정 (빈 OUTPUT_DIR도 안전하게 처리)
    ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")] if os.path.exists(OUTPUT_DIR) else []
    if ckpts:
        print(f"  체크포인트 발견: {sorted(ckpts)[-1]} → 이어서 학습")
        resume = True
    else:
        print(f"  체크포인트 없음 → 처음부터 학습")
        resume = False

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    train_time = time.time() - t0
    print(f"  학습 소요 시간: {train_time/60:.1f}분")
    notify_discord_json(discord_embed("@everyone\n✅ [6/9] 학습 완료!"))
except Exception as e:
    notify_discord_json(discord_embed(f"@everyone\n❌ [6/9] 학습 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 7. 모델 저장
# ════════════════════════════════════════

print("\n[7/9] 모델 저장...")
notify_discord_json(discord_embed("💾 [7/9] LoRA 어댑터를 저장합니다."))
try:
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    lora_files = os.listdir(LORA_DIR)
    lora_size = sum(os.path.getsize(os.path.join(LORA_DIR, f)) for f in lora_files) / 1024**2
    print(f"  저장 경로: {LORA_DIR}/")
    print(f"  파일 수: {len(lora_files)}, 총 크기: {lora_size:.1f} MB")
    notify_discord_json(discord_embed("✅ [7/9] 모델 저장 완료! (pest-detector-lora/)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [7/9] 모델 저장 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 8. 평가 (test 200건 → evaluation_results.json)
# ════════════════════════════════════════

print("\n[8/9] 평가 (test 200건)...")
notify_discord_json(discord_embed("@everyone\n🔍 [8/9] 학습된 모델을 test 데이터셋으로 평가합니다."))
EVAL_JSON_PATH = None
try:
    # 학습 후 GPU 메모리 정리 (LoRA 리로드 전)
    del trainer, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    from evaluate import evaluate as run_evaluation
    _, EVAL_JSON_PATH = run_evaluation(LORA_DIR)
    print(f"  평가 결과 저장: {EVAL_JSON_PATH}")
    notify_discord_json(discord_embed("@everyone\n✅ [8/9] 평가 완료! evaluation_results.json 저장됨."))
except Exception as e:
    notify_discord_json(discord_embed(f"@everyone\n❌ [8/9] 평가 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 9. HuggingFace Hub에 업로드
# ════════════════════════════════════════

print("\n[9/9] HuggingFace Hub 업로드...")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if HF_TOKEN:
    HUB_REPO = f"pest-{RUN_NAME}"
    notify_discord_json(discord_embed(f"☁️ [9/9] HuggingFace Hub에 업로드합니다. ({HUB_REPO})"))
    try:
        from huggingface_hub import HfApi, create_repo

        print(f"  대상 레포: {HUB_REPO}")
        t0 = time.time()
        repo_url = create_repo(HUB_REPO, token=HF_TOKEN, exist_ok=True, private=False)
        api = HfApi(token=HF_TOKEN)
        # LORA_DIR 전체 업로드 (LoRA 어댑터 + tokenizer + evaluation_results.json)
        api.upload_folder(
            folder_path=LORA_DIR,
            repo_id=repo_url.repo_id,
            commit_message=f"Upload {RUN_NAME}",
        )
        uploaded_files = os.listdir(LORA_DIR)
        print(f"  업로드 완료! ({time.time() - t0:.1f}s) — 파일 {len(uploaded_files)}개")
        has_eval = "evaluation_results.json" in uploaded_files
        notify_discord_json(discord_embed(
            f"✅ [9/9] 업로드 완료! ({HUB_REPO}) 🎉\n"
            f"파일 {len(uploaded_files)}개 (evaluation_results.json {'포함' if has_eval else '없음'})"
        ))
    except Exception as e:
        notify_discord_json(discord_embed(f"❌ [9/9] Hub 업로드 중 에러 발생: {e}"))
        raise
else:
    print("  HF_TOKEN 미설정 — 업로드 건너뜀")
    notify_discord_json(discord_embed("✅ [9/9] Hub 업로드 건너뜀 (HF_TOKEN 미설정). 파이프라인 완료! 🎉"))

print("\n" + "=" * 60)
print("파이프라인 완료!")
print("=" * 60)
