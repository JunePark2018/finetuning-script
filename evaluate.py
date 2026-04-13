"""
노지 작물 해충 진단 (서브셋) - 평가 스크립트
대상: 썩덩나무노린재 + 정상 (2클래스)
사용법: python evaluate.py --model pest-detector-lora
"""

import argparse
import json
import os
import time
import requests

from PIL import Image
from unsloth import FastVisionModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


DISCORD_BOT = {
    "username": "RunPod",
    "avatar_url": "https://i.imgur.com/0HOIh4r.png",
}
DISCORD_COLOR = 12648430


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


def discord_embed(description):
    """Embed payload 생성 헬퍼"""
    return {**DISCORD_BOT, "embeds": [{"description": description, "color": DISCORD_COLOR}]}


DATA_DIR = "/workspace/data"

SYSTEM_MSG = (
    "당신은 작물 해충 식별 전문가입니다. "
    '사진을 보고 해충의 이름만 한국어로 답하세요. '
    '해충이 없으면 "정상"이라고만 답하세요. '
    "부가 설명 없이 이름만 출력하세요."
)

CLASS_NAMES = ["썩덩나무노린재", "정상"]


def load_test_dataset():
    """test.jsonl에서 (이미지 경로, 정답 라벨) 리스트 반환"""
    jsonl_path = os.path.join(DATA_DIR, "test.jsonl")
    assert os.path.exists(jsonl_path), f"test.jsonl이 없습니다: {jsonl_path}"

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            messages = record["messages"]
            label = messages[-1]["content"][0]["text"]

            img_rel_path = None
            for msg in messages:
                for content in msg["content"]:
                    if content["type"] == "image" and "image" in content:
                        img_rel_path = content["image"].replace("\\", "/")
                        break

            if img_rel_path:
                img_path = os.path.join(DATA_DIR, img_rel_path)
                samples.append((img_path, label))

    return samples


def predict_single(model, tokenizer, image_path):
    """단일 이미지에 대해 추론하여 예측 라벨 반환"""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": SYSTEM_MSG}
        ]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "이 사진에 있는 해충의 이름을 알려주세요."},
        ]},
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=20,
        use_cache=True,
        temperature=0.1,
    )

    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    image.close()
    return prediction


def evaluate(model_path):
    """test 데이터셋 전체에 대해 평가 실행"""
    # 모델 로딩
    notify_discord_json(discord_embed("🔍 [1/3] 평가 모델을 로딩합니다."))
    try:
        print(f"모델 로딩: {model_path}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=False,
        )
        FastVisionModel.for_inference(model)
        notify_discord_json(discord_embed("✅ [1/3] 모델 로딩 완료!"))
    except Exception as e:
        notify_discord_json(discord_embed(f"❌ [1/3] 모델 로딩 중 에러 발생: {e}"))
        raise

    # 데이터 로딩 + 추론
    notify_discord_json(discord_embed("📊 [2/3] 테스트 데이터 추론을 시작합니다."))
    try:
        samples = load_test_dataset()
        print(f"테스트 샘플: {len(samples)}건\n")

        y_true = []
        y_pred = []
        inference_times = []

        for i, (img_path, label) in enumerate(samples):
            t_start = time.time()
            pred = predict_single(model, tokenizer, img_path)
            t_elapsed = time.time() - t_start
            inference_times.append(t_elapsed)

            matched = pred
            if pred not in CLASS_NAMES:
                for cls in CLASS_NAMES:
                    if cls in pred:
                        matched = cls
                        break
                else:
                    matched = pred

            y_true.append(label)
            y_pred.append(matched)

            status = "O" if label == matched else "X"
            print(f"  [{i+1}/{len(samples)}] {status}  정답: {label:10s}  예측: {matched:10s}  ({t_elapsed:.2f}s)")
    except Exception as e:
        notify_discord_json(discord_embed(f"❌ [2/3] 추론 중 에러 발생: {e}"))
        raise

    # ════════════════════════════════════════
    # 6개 평가 메트릭
    # ════════════════════════════════════════
    notify_discord_json(discord_embed("📈 [3/3] 평가 결과를 집계합니다."))

    print("\n" + "=" * 60)
    print("평가 결과 (6개 메트릭)")
    print("=" * 60)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    print("\n[1] Confusion Matrix:")
    print(f"{'':15s} {'예측:썩덩나무노린재':>15s} {'예측:정상':>10s}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{'실제:'+cls:15s} {cm[i][0]:15d} {cm[i][1]:10d}")

    # 2. Accuracy
    acc = accuracy_score(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    print(f"\n[2] Accuracy: {acc:.4f} ({correct}/{len(y_true)})")

    # 3. Precision (per class + macro)
    prec_per_class = precision_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    prec_macro = precision_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    print(f"\n[3] Precision:")
    for cls, p in zip(CLASS_NAMES, prec_per_class):
        print(f"    {cls}: {p:.4f}")
    print(f"    Macro: {prec_macro:.4f}")

    # 4. Recall (per class + macro)
    rec_per_class = recall_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    print(f"\n[4] Recall:")
    for cls, r in zip(CLASS_NAMES, rec_per_class):
        print(f"    {cls}: {r:.4f}")
    print(f"    Macro: {rec_macro:.4f}")

    # 5. Macro F1 Score
    f1_per_class = f1_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    print(f"\n[5] Macro F1 Score:")
    for cls, f in zip(CLASS_NAMES, f1_per_class):
        print(f"    {cls}: {f:.4f}")
    print(f"    Macro: {f1_macro:.4f}")

    # 6. 추론 속도
    avg_time = sum(inference_times) / len(inference_times)
    total_time = sum(inference_times)
    print(f"\n[6] 추론 속도:")
    print(f"    총 소요 시간: {total_time:.1f}s")
    print(f"    이미지당 평균: {avg_time:.2f}s")
    print(f"    처리량: {len(samples)/total_time:.1f} img/s")

    # 오답 목록
    wrong = [(t, p, s[0]) for (s, t, p) in zip(samples, y_true, y_pred) if t != p]
    if wrong:
        print(f"\n오답 {len(wrong)}건:")
        for t, p, path in wrong:
            print(f"  정답: {t:10s}  예측: {p:10s}  {os.path.basename(path)}")

    print("=" * 60)

    # 평가 결과를 JSON으로 저장 (모델 디렉토리에 함께 보관)
    from datetime import datetime
    eval_results = {
        "timestamp": datetime.now().isoformat(),
        "test_samples": len(samples),
        "confusion_matrix": cm.tolist(),
        "accuracy": round(acc, 4),
        "precision": {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, prec_per_class)},
        "precision_macro": round(float(prec_macro), 4),
        "recall": {cls: round(float(r), 4) for cls, r in zip(CLASS_NAMES, rec_per_class)},
        "recall_macro": round(float(rec_macro), 4),
        "f1": {cls: round(float(f), 4) for cls, f in zip(CLASS_NAMES, f1_per_class)},
        "f1_macro": round(float(f1_macro), 4),
        "inference_speed": {
            "avg_seconds_per_image": round(avg_time, 2),
            "images_per_second": round(len(samples) / total_time, 1),
            "total_seconds": round(total_time, 1),
        },
        "wrong_count": len(wrong),
    }
    eval_path = os.path.join(model_path, "evaluation_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"\n평가 결과 저장: {eval_path}")

    # Discord 알림
    notify_discord_json(discord_embed(
        f"@everyone\n✅ [3/3] 평가 완료!\n\n"
        f"Accuracy: {acc:.4f} ({correct}/{len(y_true)})\n"
        f"Precision (Macro): {prec_macro:.4f}\n"
        f"Recall (Macro): {rec_macro:.4f}\n"
        f"Macro F1: {f1_macro:.4f}\n"
        f"추론 속도: {avg_time:.2f}s/img ({len(samples)/total_time:.1f} img/s)\n"
        f"오답: {len(wrong)}건"
    ))


def main():
    parser = argparse.ArgumentParser(description="해충 진단 모델 평가")
    parser.add_argument("--model", default="pest-detector-lora", help="LoRA 모델 경로")
    args = parser.parse_args()

    evaluate(args.model)


if __name__ == "__main__":
    main()
