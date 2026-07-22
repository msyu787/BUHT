"""
빈 캡션(training_caption 없음) 270장에 대한 캡션 재생성 스크립트.

배경
----
captions_all.json / caption_map.csv 를 만들 때 일부 이미지(주로 인물 img_2_1,
사물 img_3_1 계열) 270장의 training_caption 이 비어 있었다. 이 스크립트는 그
270장만 골라 캡션을 다시 생성해 caption_map.csv 의 빈 칸을 채운다.

왜 512 전처리본(preprocessed_512)을 쓰나?
---------------------------------------
- 캡션은 "그림에 무엇이 그려졌는지"를 설명하는 텍스트다. 512로 리사이즈해도
  주제/구도/붓질 같은 캡션에 필요한 정보는 원본과 동일하게 보인다.
- 검증(caption_map 생성 시)에서 원본→512 전처리본이 픽셀 단위로 1:1 대응함을
  이미 확인했으므로, 512본으로 만든 캡션을 원본 파일에 그대로 매칭해도 안전하다.
- 512본은 이미 HuggingFace 데이터셋에 올라가 있어 원본 대용량 스캔을 다시
  받을 필요 없이 필요한 270장만 가볍게 내려받을 수 있다.

동작 순서
--------
1. missing_caption.csv 에서 대상 270장의 caption_key(예: 00003) 목록을 읽는다.
2. 각 key 에 해당하는 512 전처리본을 HuggingFace 에서 내려받는다.
   (data/preprocessed_512/{key}.png)
3. 기존 캡션 로직(Luxia Bridge + GPT-4o)으로 캡션을 생성한다.
4. 생성 결과를 caption_map.csv 의 빈 subject/style_tags/training_caption 칸에 채운다.
5. 중단되어도 이어서 되도록: caption_map.csv 에 이미 채워진 key 는 건너뛴다.

사용법
------
  # 상단 상수에 API_KEY(팀원에게 받은 Luxia apikey)와 HF_TOKEN 을 채운 뒤
  python3 src/regenerate_captions.py
"""

import base64
import csv
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

# ============================================================
# 경로 · 설정 상수 (환경에 맞게 여기만 수정)
# ============================================================
# 이 파일(src/regenerate_captions.py) 기준으로 레포 루트(BUHT/)를 잡는다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CAPTION_MAP_CSV = PROJECT_ROOT / "caption_map.csv"      # 채워 넣을 대상(단일 진실원본)
MISSING_CSV     = PROJECT_ROOT / "missing_caption.csv"  # 재생성 대상 270장 목록
IMG_CACHE_DIR   = PROJECT_ROOT / "outputs" / "regen_images"   # HF 다운로드 캐시
REGEN_JSON      = PROJECT_ROOT / "outputs" / "regenerated_captions.json"  # 전체 결과 백업

# --- HuggingFace 데이터셋 (Data/huggingface.py 업로드 설정과 동일) ---
HF_REPO_ID    = "buht-hyu-26/BUHT-sumukwha"
HF_REPO_TYPE  = "dataset"
HF_PATH_PREFIX = "data/preprocessed_512"   # 레포 안 512 전처리본 폴더
HF_TOKEN      = ""   # 데이터셋이 private 면 팀원에게 받은 read 토큰 입력 (public 이면 빈칸 가능)

# --- Luxia Bridge (GPT-4o) API 설정 (BUHT_preprocessing.ipynb 와 동일) ---
API_KEY    = ""   # ★ 팀원에게 받은 Luxia Bridge apikey 를 여기에 입력 ★
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o/create"
MODEL      = "gpt-4o-2024-08-06"
HEADERS    = {"apikey": API_KEY, "Content-Type": "application/json"}

# 캡션 끝에 붙는 LoRA 트리거 워드 (전통 먹 화풍)
TRIGGER_WORD = "traditional ink wash painting"
SLEEP_SEC    = 0.3   # API 호출 사이 대기(레이트리밋 완화)

# SDXL 학습용 캡션 시스템 프롬프트 (기존 파이프라인과 동일하게 유지)
CAPTION_SYSTEM_PROMPT = """You caption images for training Stable Diffusion XL on East Asian ink wash painting (sumukhwa / suiboku / sumi-e).

Write captions optimized for generative model training, not for human gallery descriptions.

Rules:
- English only, 1-2 sentences, 35-70 words
- Lead with subject + composition (what is painted, where placed, viewpoint)
- Describe ink technique: brush strokes, ink density, wash gradients, negative space
- Note sparse color washes only if visible; default to monochrome ink
- Include mood/atmosphere
- Avoid: "image of", "photo", camera/lens terms, "masterpiece", "8k", artist names
- Do NOT invent objects not clearly visible

Return ONLY valid JSON:
{
  "caption": "<full training caption>",
  "subject": "<main subject, few words>",
  "composition": "<brief layout>",
  "style_tags": ["tag1", "tag2", "tag3"]
}"""


# ============================================================
# 캡션 생성 헬퍼 (BUHT_preprocessing.ipynb 로직 그대로 재사용)
# ============================================================
def image_to_data_url(path: Path) -> str:
    """이미지를 base64 data URL 로 변환해 GPT-4o vision 입력으로 넘긴다."""
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def parse_json_response(text: str) -> dict:
    """모델 응답에서 JSON 을 뽑아낸다. ```json 코드펜스나 잡텍스트가 섞여도 복구."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 본문 어딘가에 {...} 가 있으면 그 부분만 다시 시도
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def build_training_caption(parsed: dict, trigger_word: str = TRIGGER_WORD) -> str:
    """caption 본문 + style_tags + 트리거 워드를 합쳐 최종 학습 캡션을 만든다."""
    base = parsed["caption"].strip().rstrip(".")
    tags = parsed.get("style_tags", [])
    tag_str = ", ".join(tags[:4]) if tags else "ink wash painting, sumi-e"
    return f"{base}, {tag_str}, {trigger_word}"


def _post_chat(messages, timeout=90, max_retries=3) -> str:
    """Luxia Bridge chat-completions 호출 (타임아웃/연결오류 시 지수 백오프 재시도)."""
    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "temperature": 0.2,
                "top_p": 1.0,
                "max_tokens": 400,
            }
            r = requests.post(BRIDGE_URL, headers=HEADERS, json=payload, timeout=timeout)
            if r.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise RuntimeError(f"status={r.status_code}, body={r.text[:300]}")
            return r.json()["choices"][0]["message"]["content"].strip()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2 ** attempt) + random.uniform(0, 0.2))
                continue
            raise


def caption_image_with_vlm(image_path: Path, trigger_word: str = TRIGGER_WORD) -> dict:
    """이미지 한 장을 GPT-4o 로 캡셔닝하고 training_caption 까지 만들어 반환."""
    data_url = image_to_data_url(image_path)
    raw = _post_chat([
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Caption this ink wash painting for SDXL fine-tuning. "
                         "Focus on visible content and brushwork."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ])
    parsed = parse_json_response(raw)
    parsed["training_caption"] = build_training_caption(parsed, trigger_word)
    return parsed


# ============================================================
# 데이터 입출력 헬퍼
# ============================================================
def download_preprocessed(key: str) -> Path:
    """
    HuggingFace 데이터셋에서 512 전처리본 한 장을 내려받아 로컬 경로를 반환.
    - 레포 안 경로: data/preprocessed_512/{key}.png
    - hf_hub_download 는 자체 캐시가 있어 이미 받은 파일은 재다운로드하지 않는다.
    """
    return Path(hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=f"{HF_PATH_PREFIX}/{key}.png",
        token=HF_TOKEN or None,          # 빈 문자열이면 익명(public) 접근
        local_dir=IMG_CACHE_DIR,         # 프로젝트 안 캐시 폴더에 모아둠
    ))


def load_csv_rows(path: Path) -> list[dict]:
    """CSV 를 dict 리스트로 읽는다 (BOM 대응: utf-8-sig)."""
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    """dict 리스트를 원래 컬럼 순서 그대로 CSV 로 덮어쓴다 (엑셀 한글 대응)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ============================================================
# 메인
# ============================================================
def main() -> None:
    # --- 0. API 키 확인 (없으면 안내 후 종료) ---
    if not API_KEY:
        print("[!] API_KEY 가 비어 있습니다. 파일 상단 상수에 Luxia Bridge apikey 를 입력하세요.")
        return

    IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REGEN_JSON.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. 재생성 대상(270장) key 목록 로드 ---
    targets = [r["caption_key"] for r in load_csv_rows(MISSING_CSV)]

    # --- 2. caption_map.csv 를 메모리로 읽고 key→row 인덱스 구성 ---
    rows = load_csv_rows(CAPTION_MAP_CSV)
    row_by_key = {r["caption_key"]: r for r in rows}

    # --- 3. 이어서 하기: 이미 training_caption 이 채워진 대상은 건너뜀 ---
    todo = [k for k in targets if not row_by_key[k]["training_caption"].strip()]
    done_already = len(targets) - len(todo)
    print(f"재생성 대상 총 {len(targets)}장 | 이미 완료 {done_already}장 | 이번에 처리 {len(todo)}장")

    # 전체 구조화 결과 백업(JSON) 로드 — 있으면 이어서 누적
    regen = {}
    if REGEN_JSON.exists():
        regen = json.loads(REGEN_JSON.read_text(encoding="utf-8"))

    ok, failed = 0, 0
    for i, key in enumerate(todo, start=1):
        try:
            # (a) 512 전처리본 다운로드
            img_path = download_preprocessed(key)
            # (b) 캡션 생성
            result = caption_image_with_vlm(img_path)

            # (c) caption_map.csv 해당 행의 빈 칸 채우기
            row = row_by_key[key]
            row["subject"] = result.get("subject", "")
            tags = result.get("style_tags", [])
            row["style_tags"] = "|".join(tags) if isinstance(tags, list) else ""
            row["training_caption"] = result.get("training_caption", "")

            # (d) 구조화 결과 백업 + CSV 즉시 저장 (중단 대비: 매 건 저장)
            regen[key] = {
                "caption_key": key,
                "original_filename": row["original_filename"],
                "subject": result.get("subject", ""),
                "composition": result.get("composition", ""),
                "caption": result.get("caption", ""),
                "style_tags": tags,
                "training_caption": result.get("training_caption", ""),
            }
            REGEN_JSON.write_text(json.dumps(regen, ensure_ascii=False, indent=2), encoding="utf-8")
            write_csv_rows(CAPTION_MAP_CSV, rows)

            ok += 1
            print(f"[{i}/{len(todo)}] OK  {key}  {row['original_filename']}")
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(todo)}] FAIL {key}: {e}")

        time.sleep(SLEEP_SEC)

    # --- 4. 최종 요약 ---
    remaining = sum(1 for k in targets if not row_by_key[k]["training_caption"].strip())
    print("=" * 60)
    print(f"완료: 성공 {ok} | 실패 {failed} | 남은 빈 캡션 {remaining}")
    print(f"업데이트: {CAPTION_MAP_CSV}")
    print(f"백업 JSON: {REGEN_JSON}")
    if remaining:
        print("남은 항목은 다시 실행하면 이어서 처리됩니다(이미 된 건 건너뜀).")


if __name__ == "__main__":
    main()
