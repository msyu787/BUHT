"""
build_reference_topk.py
=======================
각 이미지(train / val / test)마다 "스타일 참고용 reference 이미지 top-k"를
train 이미지 중에서 골라 매핑 파일로 저장한다.

배경
----
reference 이미지를 조건으로 넣어 학습/생성하려면, 각 이미지마다
"어떤 그림을 참고할지"가 미리 정해져 있어야 한다.
이 스크립트는 그 목록만 만든다. 실제로 모델에 집어넣는 일(IP-Adapter 등)은
학습/추론 코드가 이 매핑 파일을 읽어서 처리한다.

데이터 구조 (data/split_manifest.csv 기준)
------------------------------------------
data/{train,val,test}/
    00000.png         수묵화 원본(512)
    00000.txt         캡션
    00000_QA.txt      QA (이 스크립트는 사용하지 않음)
    00000_sketch.png  스케치

선택 규칙 — split 마다 다르게 고르는 이유가 핵심
------------------------------------------------
reference 후보(pool)는 항상 train 이미지만 쓴다.
val/test 이미지가 후보에 들어가면 평가 대상을 미리 보여주는 셈이 된다.

1) train 이미지의 reference  (학습용)
   학습 때는 정답 수묵화가 있으므로 그림과 캡션을 모두 본다.
   점수 = IMAGE_WEIGHT * 이미지 유사도 + TEXT_WEIGHT * 캡션 유사도
   (둘 중 한쪽이라도 캡션이 비어 있으면 이미지 유사도만 사용)

2) val / test 이미지의 reference  (추론·평가용)
   생성할 때 모델이 받는 것은 스케치와 캡션뿐이고 정답 그림은 없다.
   정답 그림의 겉모습으로 reference 를 고르면 정답을 보고 힌트를 고르는 셈이라
   평가가 부풀려진다. 그래서 입력으로 실제 주어지는 정보만 쓴다.
   - 캡션이 있으면: 캡션 유사도로 고름 (후보도 캡션 있는 train 이미지로 한정)
   - 캡션이 없으면: 스케치 유사도로 고름 (query 스케치 vs train 스케치)

공통 제외 규칙
- 자기 자신
- 같은 base ID (동일 원본의 다른 편집본) — split 이 달라도 제외
- 이미지 유사도 DUPLICATE_THRESHOLD 이상 (base ID 매핑에서 빠진 중복 방어)
  ※ val/test 에서도 정답 그림은 '순위'에는 쓰지 않고 '중복 제거'에만 쓴다.
     제외는 결과를 보수적으로 만드는 방향이라 누수가 생기지 않는다.

실행
----
레포 루트에서: python src/build_reference_topk.py

산출물
------
outputs/reference_topk.json : {"00507": ["00123", "00891", ...], ...}   (전 split)
outputs/reference_topk.csv  : split, image, rank, reference, score, method,
                              image_sim, text_sim, sketch_sim

수정 이력
---------
[2026-07-27] 최초 작성
  무엇을: CLIP 이미지 임베딩 기반 top-k reference 선택 스크립트 신규 작성.
  왜:     reference 조건을 학습에 넣기 위해, 이미지별 참고 대상 목록이 필요하다.
  방법:   이미지 임베딩 유사도 상위 k개를 뽑고, 자기 자신과 같은 base ID 는 제외.

[2026-07-30] caption_map 경로/누락 처리 강화
  무엇을: caption_map.csv 를 스크립트 위치 기준으로 찾고, 없으면 기본적으로 중단.
  왜:     상대경로로는 실행 위치에 따라 조용히 못 찾아 base ID 제외가 빠질 수 있었다.

[2026-09-30] 캡션 유사도 반영 + train/val/test 분할 구조 대응
  무엇을: (1) 캡션을 CLIP 텍스트 임베딩으로 만들어 유사도에 반영.
          (2) 데이터가 data/{train,val,test} 로 나뉜 구조에 맞춰 경로를 변경하고,
              폴더에 섞여 있는 *_sketch.png 는 원본 목록에서 제외.
          (3) reference 후보를 train 으로 한정하고, val/test 는 캡션(없으면 스케치)으로만 선택.
  왜:     (1) 겉모습만 보면 내용이 다른 그림이 뽑힐 수 있다.
          (2) 기존 경로(data/image_txt)가 사라졌고, 스케치가 원본 후보로 섞이면 안 된다.
          (3) val/test 의 reference 를 정답 그림 기준으로 고르거나 test 끼리 참고하면
              평가 대상 정보가 새어 들어가 성능이 부풀려진다.
  방법:   캡션은 각 split 폴더의 {stem}.txt 를 읽는다(빈 파일은 캡션 없음으로 처리).
          base ID 는 기존처럼 레포 루트의 caption_map.csv 로 판별한다.
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

DATA_ROOT = Path("/content/drive/MyDrive/BUHT/data")
SPLIT_MANIFEST = DATA_ROOT / "split_manifest.csv"

POOL_SPLIT = "train"                      # reference 후보는 이 split 에서만 뽑는다
QUERY_SPLITS = ["train", "val", "test"]   # reference 를 만들어 줄 대상

# 레포 루트 (이 파일: <레포>/src/build_reference_topk.py)
REPO_ROOT = Path(__file__).resolve().parent.parent
CAPTION_MAP_CSV = REPO_ROOT / "caption_map.csv"   # base ID 판별용

# caption_map.csv 가 없을 때: False 면 중단, True 면 base ID 제외 없이 진행
ALLOW_MISSING_CAPTION_MAP = False

OUTPUT_DIR = REPO_ROOT / "outputs"
JSON_PATH = OUTPUT_DIR / "reference_topk.json"
CSV_PATH = OUTPUT_DIR / "reference_topk.csv"

TOP_K = 5
BATCH_SIZE = 32
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 이미지 유사도가 이 값 이상이면 사실상 같은 그림으로 보고 제외
DUPLICATE_THRESHOLD = 0.98

# train 이미지(학습용) reference 점수 = IMAGE_WEIGHT*이미지 + TEXT_WEIGHT*캡션
# TEXT_WEIGHT = 0 이면 이미지만 본다. reference 는 스타일 참고용이라 겉모습 비중을 크게 둠.
IMAGE_WEIGHT = 0.7
TEXT_WEIGHT = 0.3

EXCLUDED = -2.0   # 제외 표시값 (코사인 유사도 범위 -1~1 밖)


# =====================================================================
# 데이터 로드
# =====================================================================

def load_manifest(path: Path) -> list[dict]:
    """split_manifest.csv 를 읽는다. 각 행: split, stem, image_file, caption_file, sketch_file"""
    if not path.exists():
        raise RuntimeError(f"split_manifest.csv 를 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in QUERY_SPLITS or r["split"] == POOL_SPLIT]
    return rows


def read_caption(path: Path) -> str | None:
    """캡션 파일을 읽는다. 파일이 없거나 비어 있으면 None."""
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    text = " ".join(text.split())
    return text or None


def extract_base_id(original_filename: str) -> str:
    """img_1_1_0001(1232)_Scan_edit.jpg -> img_1_1_0001"""
    stem = Path(original_filename).stem
    m = re.match(r"^(.+?)\(\d+\)", stem)
    return m.group(1) if m else stem


def load_base_id_map(csv_path: Path) -> dict[str, str]:
    """caption_map.csv 로 {전처리번호: base ID} 를 만든다."""
    if not csv_path.exists():
        if not ALLOW_MISSING_CAPTION_MAP:
            raise RuntimeError(
                f"caption_map.csv 를 찾을 수 없습니다: {csv_path}\n"
                f"base ID 제외 없이 진행하려면 ALLOW_MISSING_CAPTION_MAP=True 로 명시하세요."
            )
        print(f"[경고] {csv_path} 없음 — base ID 제외 없이 진행합니다.")
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


# =====================================================================
# CLIP 임베딩
# =====================================================================

_clip_cache: dict = {}


def _get_clip():
    if "model" not in _clip_cache:
        from transformers import CLIPModel, CLIPProcessor
        _clip_cache["model"] = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
        _clip_cache["processor"] = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _clip_cache["model"], _clip_cache["processor"]


def _as_tensor(feat):
    """transformers 버전에 따라 텐서가 아닌 객체가 올 때를 대비 (evaluate.py 와 동일)."""
    if isinstance(feat, torch.Tensor):
        return feat
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(feat, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"CLIP 임베딩을 텐서로 변환할 수 없습니다: {type(feat)}")


@torch.no_grad()
def embed_images(paths: list[Path], label: str) -> torch.Tensor:
    """이미지 목록 → L2 정규화된 CLIP 이미지 임베딩 (N, D)"""
    model, processor = _get_clip()
    chunks = []
    for i in range(0, len(paths), BATCH_SIZE):
        images = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE]]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        emb = _as_tensor(model.get_image_features(**inputs))
        chunks.append(emb / emb.norm(dim=-1, keepdim=True))
        print(f"  {label} 임베딩 {min(i + BATCH_SIZE, len(paths))}/{len(paths)}", end="\r")
    print()
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def embed_texts(texts: list[str], label: str) -> torch.Tensor:
    """캡션 목록 → L2 정규화된 CLIP 텍스트 임베딩 (N, D). 77 토큰 초과분은 잘린다."""
    model, processor = _get_clip()
    chunks = []
    for i in range(0, len(texts), BATCH_SIZE):
        inputs = processor(text=texts[i:i + BATCH_SIZE], return_tensors="pt",
                           padding=True, truncation=True).to(DEVICE)
        emb = _as_tensor(model.get_text_features(**inputs))
        chunks.append(emb / emb.norm(dim=-1, keepdim=True))
        print(f"  {label} 임베딩 {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    print()
    return torch.cat(chunks, dim=0)


def embed_optional_texts(stems: list[str], captions: dict[str, str], label: str) -> torch.Tensor:
    """캡션이 있는 것만 임베딩하고, 없는 행은 NaN 으로 채운 (N, D) 텐서를 돌려준다."""
    has = [i for i, s in enumerate(stems) if s in captions]
    if not has:
        return None
    emb = embed_texts([captions[stems[i]] for i in has], label)
    out = torch.full((len(stems), emb.shape[1]), float("nan"), device=emb.device)
    out[torch.tensor(has, device=emb.device)] = emb
    return out


# =====================================================================
# top-k 선택
# =====================================================================

def _nan_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """NaN 행이 있는 임베딩끼리 내적. NaN 이 낀 쌍은 결과도 NaN."""
    return a @ b.T


def build_topk(
    q_stems: list[str], q_splits: list[str],
    p_stems: list[str],
    q_img: torch.Tensor, p_img: torch.Tensor,
    q_txt, p_txt,
    q_sk, p_sk,
    base_ids: dict[str, str],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    query × pool 유사도 행렬을 만들고 split 규칙에 따라 점수를 정한 뒤 top-k 를 고른다.
    q_txt/p_txt 는 캡션 없는 행이 NaN 인 텐서(또는 None).
    q_sk/p_sk 는 스케치 임베딩(또는 None — val/test 에 캡션 없는 query 가 없으면 불필요).
    """
    nq, npool = len(q_stems), len(p_stems)
    img_sim = q_img @ p_img.T
    txt_sim = _nan_matmul(q_txt, p_txt) if q_txt is not None and p_txt is not None \
        else torch.full((nq, npool), float("nan"), device=img_sim.device)
    sk_sim = q_sk @ p_sk.T if q_sk is not None and p_sk is not None \
        else torch.full((nq, npool), float("nan"), device=img_sim.device)

    has_txt = ~torch.isnan(txt_sim)
    is_train = torch.tensor([s == POOL_SPLIT for s in q_splits], device=img_sim.device).unsqueeze(1)
    q_has_cap = torch.tensor([not torch.isnan(q_txt[i]).any() if q_txt is not None else False
                              for i in range(nq)], device=img_sim.device).unsqueeze(1)

    score = torch.full((nq, npool), EXCLUDED, device=img_sim.device)
    method = [None] * nq

    # 1) train query: 이미지+캡션 합산, 캡션 없는 쌍은 이미지만
    blend = torch.where(has_txt, IMAGE_WEIGHT * img_sim + TEXT_WEIGHT * torch.nan_to_num(txt_sim), img_sim)
    score = torch.where(is_train.expand(-1, npool), blend, score)

    # 2) val/test query + 캡션 있음: 캡션 유사도만, 후보도 캡션 있는 train 으로 한정
    eval_cap = (~is_train) & q_has_cap
    score = torch.where(eval_cap.expand(-1, npool) & has_txt, torch.nan_to_num(txt_sim), score)

    # 3) val/test query + 캡션 없음: 스케치 유사도
    eval_nocap = (~is_train) & (~q_has_cap)
    if eval_nocap.any():
        if torch.isnan(sk_sim).all():
            raise RuntimeError("캡션 없는 val/test 이미지가 있는데 스케치 임베딩이 없습니다.")
        score = torch.where(eval_nocap.expand(-1, npool), sk_sim, score)

    for i in range(nq):
        if q_splits[i] == POOL_SPLIT:
            method[i] = "image+text"
        elif bool(q_has_cap[i]):
            method[i] = "text"
        else:
            method[i] = "sketch"

    # --- 제외 규칙 ---
    p_index = {s: j for j, s in enumerate(p_stems)}
    for i, s in enumerate(q_stems):                       # 자기 자신
        j = p_index.get(s)
        if j is not None:
            score[i, j] = EXCLUDED

    excluded_base = 0
    if base_ids:                                          # 같은 base ID
        pool_by_base = defaultdict(list)
        for j, s in enumerate(p_stems):
            if s in base_ids:
                pool_by_base[base_ids[s]].append(j)
        for i, s in enumerate(q_stems):
            js = [j for j in pool_by_base.get(base_ids.get(s), []) if p_stems[j] != s]
            if js:
                score[i, js] = EXCLUDED
                excluded_base += len(js)
    print(f"같은 base ID 쌍 제외: {excluded_base}건")

    dup = (img_sim >= DUPLICATE_THRESHOLD) & (score > EXCLUDED)   # 중복 의심
    if int(dup.sum()):
        score[dup] = EXCLUDED
        print(f"이미지 유사도 {DUPLICATE_THRESHOLD} 이상 쌍 제외: {int(dup.sum())}건")

    top_score, top_idx = score.topk(min(top_k, npool), dim=1)

    def val(t, i, j):
        v = float(t[i, j])
        return None if v != v else round(v, 4)

    results = []
    for i, s in enumerate(q_stems):
        refs = []
        for r in range(top_idx.shape[1]):
            j = int(top_idx[i, r])
            if float(top_score[i, r]) <= EXCLUDED + 0.5:
                continue
            refs.append({
                "reference": p_stems[j],
                "score": round(float(top_score[i, r]), 4),
                "image_sim": val(img_sim, i, j) if q_splits[i] == POOL_SPLIT else None,
                "text_sim": val(txt_sim, i, j),
                "sketch_sim": val(sk_sim, i, j),
            })
        results.append({"split": q_splits[i], "image": s, "method": method[i], "refs": refs})
    return results


def save_results(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    simple = {r["image"]: [x["reference"] for x in r["refs"]] for r in results}
    JSON_PATH.write_text(json.dumps(simple, ensure_ascii=False, indent=2), encoding="utf-8")

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "rank", "reference", "score", "method",
                    "image_sim", "text_sim", "sketch_sim"])
        for r in results:
            for rank, x in enumerate(r["refs"], start=1):
                w.writerow([r["split"], r["image"], rank, x["reference"], x["score"], r["method"],
                            *["" if x[k] is None else x[k] for k in ("image_sim", "text_sim", "sketch_sim")]])
    print(f"\n저장 완료\n  {JSON_PATH}\n  {CSV_PATH}")


def main() -> None:
    rows = load_manifest(SPLIT_MANIFEST)
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    print("split 별 개수:", {k: len(v) for k, v in by_split.items()}, f"(device={DEVICE})")

    def paths(r, key):
        return DATA_ROOT / r["split"] / r[key]

    captions = {}
    for r in rows:
        c = read_caption(paths(r, "caption_file"))
        if c:
            captions[r["stem"]] = c
    print(f"캡션 있음: {len(captions)}/{len(rows)}")

    base_ids = load_base_id_map(CAPTION_MAP_CSV)

    pool = by_split[POOL_SPLIT]
    queries = [r for sp in QUERY_SPLITS for r in by_split[sp]]
    p_stems = [r["stem"] for r in pool]
    q_stems = [r["stem"] for r in queries]
    q_splits = [r["split"] for r in queries]

    print("CLIP 임베딩 계산 중...")
    p_img = embed_images([paths(r, "image_file") for r in pool], "train 이미지")
    q_img = embed_images([paths(r, "image_file") for r in queries], "query 이미지")
    p_txt = embed_optional_texts(p_stems, captions, "train 캡션")
    q_txt = embed_optional_texts(q_stems, captions, "query 캡션")

    need_sketch = [r for r in queries if r["split"] != POOL_SPLIT and r["stem"] not in captions]
    q_sk = p_sk = None
    if need_sketch:
        print(f"캡션 없는 val/test {len(need_sketch)}장 → 스케치 유사도로 선택")
        p_sk = embed_images([paths(r, "sketch_file") for r in pool], "train 스케치")
        q_sk = embed_images([paths(r, "sketch_file") for r in queries], "query 스케치")

    print(f"top-{TOP_K} reference 선택 중...")
    results = build_topk(q_stems, q_splits, p_stems, q_img, p_img,
                         q_txt, p_txt, q_sk, p_sk, base_ids, top_k=TOP_K)

    print("\n--- 선택 방식 요약 ---")
    summary = defaultdict(lambda: defaultdict(int))
    for r in results:
        summary[r["split"]][r["method"]] += 1
    for sp, d in summary.items():
        print(f"  {sp}: {dict(d)}")

    empty = [r["image"] for r in results if not r["refs"]]
    if empty:
        print(f"\n[경고] reference 를 못 찾은 이미지 {len(empty)}장: {empty[:10]}")

    print("\n--- 결과 샘플 ---")
    for r in results[:3] + [x for x in results if x["split"] == "test"][:2]:
        print(f"  [{r['split']}/{r['method']}] {r['image']} → "
              + ", ".join(f"{x['reference']}({x['score']})" for x in r["refs"]))

    save_results(results)


if __name__ == "__main__":
    main()