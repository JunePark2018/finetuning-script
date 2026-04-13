"""
노지 작물 해충 진단 (서브셋) - 추론 스크립트
대상: 썩덩나무노린재 + 정상 (2클래스)
사용법: python inference.py --image test.jpg --model pest-detector-lora
"""

import argparse

from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer

SYSTEM_MSG = (
    "당신은 작물 해충 식별 전문가입니다. "
    '사진을 보고 해충의 이름만 한국어로 답하세요. '
    '해충이 없으면 "정상"이라고만 답하세요. '
    "부가 설명 없이 이름만 출력하세요."
)

CLASS_NAMES = [
    "썩덩나무노린재", "정상",
]


def predict(image_path: str, model_path: str):
    """단일 이미지에 대해 해충 분류 추론"""
    print(f"모델 로딩: {model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)

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

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    print(f"\n이미지: {image_path}")
    print("예측 결과: ", end="", flush=True)
    model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=20,
        use_cache=True,
        temperature=0.1,
    )
    print()
    image.close()


def main():
    parser = argparse.ArgumentParser(description="해충 진단 추론")
    parser.add_argument("--image", required=True, help="이미지 경로")
    parser.add_argument("--model", default="pest-detector-lora", help="LoRA 모델 경로")
    args = parser.parse_args()

    predict(args.image, args.model)


if __name__ == "__main__":
    main()
