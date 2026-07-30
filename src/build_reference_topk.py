"""
build_reference_topk.py
=======================
학습 이미지마다 "스타일 참고용 reference 이미지 top-k"를 찾아 매핑 파일로 저장한다.

배경
----
reference 이미지를 조건으로 넣어 학습/생성하려면, 각 이미지마다
"어떤 그림을 참고할지"가 미리 정해져 있어야 한다.
이 스크립트는 그 목록만 만든다. 실제로 모델에 집어넣는 일(IP-Adapter 등)은
학습/추론 코드가 이 매핑 파일을 읽어서 처리한다.

동작
----
1. 학습 이미지 전체를 CLIP 이미지 임베딩으로 변환
2. 이미지끼리 코사인 유사도를 계산
3. 각 이미지에 대해 유사도 상위 k개를 고름
   - 자기 자신 제외
   - 같은 base ID(동일 원본의 다른 편집본) 제외
4. JSON / CSV 두 형태로 저장

제외 규칙이 중요한 이유
----------------------
reference 로 자기 자신이나 같은 원본의 편집본이 뽑히면,
모델에 정답을 미리 보여주는 셈이 되어 학습이 오염된다.
(평가 단계에서 q_copy 가 부풀려지는 것과 같은 문제)

실행
----
python build_reference_topk.py

산출물
------
outputs/reference_topk.json : {"00507": ["00123", "00891", ...], ...}
outputs/reference_topk.csv  : image, rank, reference, similarity

수정 이력
---------
[2026-07-27] 최초 작성
  무엇을: CLIP 이미지 임베딩 기반 top-k reference 선택 스크립트 신규 작성.
  왜:     reference 조건을 학습에 넣기 위해, 이미지별 참고 대상 목록이 필요하다.
  방법:   이미지 임베딩 유사도 상위 k개를 뽑고, 자기 자신과 같은 base ID 는 제외.
          caption_map.csv 의 original_filename 으로 base ID 를 판별한다.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

# =====================================================================
# 설정값 (경로/하이퍼파라미터는 전부 여기서 수정)
# =====================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/BUHT")
DATA_ROOT = PROJECT_ROOT / "data"

# 학습 이미지 폴더 (512 전처리본 + 캡션 .txt 가 함께 있는 폴더)
IMAGE_DIR = DATA_ROOT / "image_txt"

# 원본 파일명 ↔ 전처리 번호 매핑표 (base ID 판별에 사용)
# 없으면 base ID 제외 규칙 없이 자기 자신만 제외한다.
CAPTION_MAP_CSV = Path("caption_map.csv")

# 결과 저장 경로
OUTPUT_DIR = Path("outputs")
JSON_PATH = OUTPUT_DIR / "reference_topk.json"
CSV_PATH = OUTPUT_DIR / "reference_topk.csv"

TOP_K = 5                       # 이미지당 뽑을 reference 개수
BATCH_SIZE = 32
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 유사도가 이 값 이상이면 "사실상 동일 그림"으로 보고 제외한다.
# 같은 원본의 편집본이 caption_map 에 누락된 경우를 잡기 위한 안전장치.
DUPLICATE_THRESHOLD = 0.98


# =====================================================================
# 유틸
# =====================================================================

def list_images(folder: Path) -> list[Path]:
    """폴더 안의 이미지 파일을 파일명 순으로 반환한다."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def extract_base_id(original_filename: str) -> str:
    """
    원본 파일명에서 base ID 를 추출한다.
    예: img_1_1_0001(1232)_Scan_edit.jpg -> img_1_1_0001

    같은 base ID 를 가진 파일들은 동일 원본의 다른 편집본이므로,
    서로의 reference 가 되지 않도록 걸러내는 데 쓴다.
    """
    stem = Path(original_filename).stem
    m = re.match(r"^(.+?)\(\d+\)", stem)
    return m.group(1) if m else stem


def load_base_id_map(csv_path: Path) -> dict[str, str]:
    """
    caption_map.csv 를 읽어 {전처리번호: base ID} 를 만든다.
    파일이 없으면 빈 dict 를 반환하고, 그 경우 base ID 제외 규칙은 생략된다.
    """
    if not csv_path.exists():
        print(f"[경고] {csv_path} 가 없어 base ID 제외 규칙을 건너뜁니다.")
        return {}

    mapping = {}
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = row.get("caption_key") or Path(row.get("preprocessed_file", "")).stem
            original = row.get("original_filename", "")
            if key and original:
                mapping[key] = extract_base_id(original)
    print(f"base ID 매핑 로드: {len(mapping)}건")
    return mapping


_clip_cache: dict = {}


def _get_clip():
    """CLIP 모델/프로세서를 1회만 로드해 재사용한다."""
    if "model" not in _clip_cache:
        from transformers import CLIPModel, CLIPProcessor  # 지연 import
        _clip_cache["model"] = (
            CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
        )
        _clip_cache["processor"] = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _clip_cache["model"], _clip_cache["processor"]


def _as_tensor(feat):
    """
    CLIP 반환값을 텐서로 통일한다.
    transformers 버전에 따라 임베딩 텐서가 아니라
    BaseModelOutputWithPooling 객체가 오는 경우가 있다. (evaluate.py 와 동일한 처리)
    """
    if isinstance(feat, torch.Tensor):
        return feat
    for attr in ("image_embeds", "pooler_output"):
        value = getattr(feat, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"CLIP 임베딩을 텐서로 변환할 수 없습니다: {type(feat)}")


@torch.no_grad()
def compute_embeddings(paths: list[Path]) -> torch.Tensor:
    """이미지 목록을 L2 정규화된 CLIP 임베딩으로 변환한다. shape (N, D)"""
    model, processor = _get_clip()
    chunks = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = paths[i : i + BATCH_SIZE]
        images = [load_rgb(p) for p in batch]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        emb = _as_tensor(model.get_image_features(**inputs))
        chunks.append(emb / emb.norm(dim=-1, keepdim=True))
        print(f"  임베딩 {min(i + BATCH_SIZE, len(paths))}/{len(paths)}", end="\r")
    print()
    return torch.cat(chunks, dim=0)


# =====================================================================
# top-k 선택
# =====================================================================

def build_topk(
    paths: list[Path],
    emb: torch.Tensor,
    base_ids: dict[str, str],
    top_k: int = TOP_K,
) -> dict[str, list[dict]]:
    """
    각 이미지에 대해 유사도 상위 top_k reference 를 고른다.

    제외 규칙
    - 자기 자신
    - 같은 base ID (동일 원본의 다른 편집본)
    - 유사도가 DUPLICATE_THRESHOLD 이상 (매핑 누락된 중복 방어)
    """
    stems = [p.stem for p in paths]
    sim = emb @ emb.T                      # (N, N) 코사인 유사도

    # 제외 표시값. 코사인 유사도 범위(-1~1) 밖의 값을 써서
    # "제외된 쌍"과 "유사도가 낮은 정상 후보"를 확실히 구분한다.
    EXCLUDED = -2.0

    # 자기 자신은 항상 제외
    sim.fill_diagonal_(EXCLUDED)

    # 같은 base ID 끼리 서로 제외
    if base_ids:
        groups = defaultdict(list)
        for idx, stem in enumerate(stems):
            bid = base_ids.get(stem)
            if bid:
                groups[bid].append(idx)
        excluded_pairs = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            idx = torch.tensor(members, device=sim.device)
            sim[idx.unsqueeze(1), idx.unsqueeze(0)] = EXCLUDED
            excluded_pairs += len(members) * (len(members) - 1)
        print(f"같은 base ID 쌍 제외: {excluded_pairs}건")

    # 중복 의심 쌍 제외
    dup_mask = sim >= DUPLICATE_THRESHOLD
    dup_count = int(dup_mask.sum())
    if dup_count:
        sim[dup_mask] = EXCLUDED
        print(f"유사도 {DUPLICATE_THRESHOLD} 이상 쌍 제외: {dup_count}건")

    top_sim, top_idx = sim.topk(top_k, dim=1)

    result = {}
    for i, stem in enumerate(stems):
        refs = []
        for rank in range(top_k):
            j = int(top_idx[i, rank])
            score = float(top_sim[i, rank])
            if score <= EXCLUDED + 0.5:    # 제외 표시된 쌍은 건너뜀
                continue
            refs.append({"reference": stems[j], "similarity": round(score, 4)})
        result[stem] = refs
    return result


def save_results(topk: dict[str, list[dict]]) -> None:
    """JSON(학습 코드용)과 CSV(육안 확인용) 두 형태로 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON: 학습 코드에서 바로 읽기 쉬운 형태
    simple = {stem: [r["reference"] for r in refs] for stem, refs in topk.items()}
    JSON_PATH.write_text(
        json.dumps(simple, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # CSV: 유사도까지 포함해 사람이 검토하기 쉬운 형태
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "rank", "reference", "similarity"])
        for stem, refs in topk.items():
            for rank, r in enumerate(refs, start=1):
                writer.writerow([stem, rank, r["reference"], r["similarity"]])

    print(f"\n저장 완료")
    print(f"  {JSON_PATH}")
    print(f"  {CSV_PATH}")


def main() -> None:
    paths = list_images(IMAGE_DIR)
    if not paths:
        raise RuntimeError(f"이미지가 없습니다: {IMAGE_DIR}")
    print(f"대상 이미지: {len(paths)}장 (device={DEVICE})")

    base_ids = load_base_id_map(CAPTION_MAP_CSV)

    print("CLIP 임베딩 계산 중...")
    emb = compute_embeddings(paths)

    print(f"top-{TOP_K} reference 선택 중...")
    topk = build_topk(paths, emb, base_ids)

    # 결과 미리보기 — 뽑힌 reference 가 말이 되는지 눈으로 확인할 것
    print("\n--- 결과 샘플 ---")
    for stem in list(topk)[:5]:
        refs = ", ".join(f"{r['reference']}({r['similarity']})" for r in topk[stem])
        print(f"  {stem} → {refs}")

    empty = [s for s, r in topk.items() if not r]
    if empty:
        print(f"\n[경고] reference 를 못 찾은 이미지 {len(empty)}장: {empty[:10]}")

    save_results(topk)


if __name__ == "__main__":
    main()
