"""
노지 작물 해충 진단 (서브셋) - 평가 스크립트
대상: 썩덩나무노린재 + 정상 (2클래스)
사용법: python evaluate.py --model pest-detector-lora
"""

import argparse
import json
import os

from PIL import Image
from unsloth import FastVisionModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

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

    # 생성된 토큰만 디코딩
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    image.close()
    return prediction


def evaluate(model_path):
    """test 데이터셋 전체에 대해 평가 실행"""
    # 모델 로딩
    print(f"모델 로딩: {model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)

    # 데이터 로딩
    samples = load_test_dataset()
    print(f"테스트 샘플: {len(samples)}건\n")

    # 추론
    y_true = []
    y_pred = []

    for i, (img_path, label) in enumerate(samples):
        pred = predict_single(model, tokenizer, img_path)

        # 예측값이 클래스 이름에 포함되지 않으면 가장 가까운 것으로 매칭
        matched = pred
        if pred not in CLASS_NAMES:
            for cls in CLASS_NAMES:
                if cls in pred:
                    matched = cls
                    break
            else:
                matched = pred  # 매칭 실패 시 원본 유지

        y_true.append(label)
        y_pred.append(matched)

        status = "O" if label == matched else "X"
        print(f"  [{i+1}/{len(samples)}] {status}  정답: {label:10s}  예측: {matched}")

    # 결과 출력
    print("\n" + "=" * 50)
    print("평가 결과")
    print("=" * 50)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{len(y_true)})\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    print(f"{'':15s} {'썩덩나무노린재':>10s} {'정상':>10s}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:15s} {cm[i][0]:10d} {cm[i][1]:10d}")

    # 틀린 샘플 요약
    wrong = [(t, p, s[0]) for (s, t, p) in zip(samples, y_true, y_pred) if t != p]
    if wrong:
        print(f"\n오답 {len(wrong)}건:")
        for t, p, path in wrong:
            print(f"  정답: {t:10s}  예측: {p:10s}  {os.path.basename(path)}")


def main():
    parser = argparse.ArgumentParser(description="해충 진단 모델 평가")
    parser.add_argument("--model", default="pest-detector-lora", help="LoRA 모델 경로")
    args = parser.parse_args()

    evaluate(args.model)


if __name__ == "__main__":
    main()
