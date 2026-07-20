# -*- coding: utf-8 -*-
"""
BUHT 수묵화 생성 모델 평가 스크립트
====================================

SD v1.5 + ControlNet + LoRA로 생성한 한국 수묵화를 4가지 지표로 평가하고,
순정(baseline) SD v1.5 생성 결과와 비교하는 표를 출력한다.

지표 요약
---------
- q_sem   : 텍스트-이미지 일치도 (CLIP score).            높을수록 좋음
- q_struct: 입력 스케치의 구조 반영도 (edge IoU + SSIM).   높을수록 좋음
- q_style : 수묵화다움 (실제 수묵화 집단과의 FID).          낮을수록 좋음
- q_copy  : 학습 데이터 복사 여부 (최근접 CLIP 유사도).     낮을수록 좋음

사용 전제
---------
생성 이미지 파일명은 스케치/캡션과 같은 stem을 쓴다고 가정한다.
예: 스케치 data/sketch/painting_001.png 로 생성한 이미지는
    generated/ours/painting_001.png 로 저장.
(노트북의 candidate_seed_42.png 같은 이름은 스케치와 매칭이 안 되므로
 q_sem / q_struct 계산 시 건너뛰고 경고를 출력한다.
 q_style / q_copy 는 파일명과 무관하게 폴더 전체를 사용하므로 항상 계산된다.)

필요 패키지 (Colab T4 기준, 대부분 기본 설치되어 있음)
------------------------------------------------------
# !pip install -q transformers torchmetrics[image]
# (opencv-python, scikit-image, pandas 는 Colab에 기본 포함)

실행
----
python evaluate.py            # OURS_DIR / BASELINE_DIR 를 모두 평가 후 비교표 출력
또는 노트북에서 개별 함수만 import 해서 사용:
    from evaluate import q_sem, q_struct, q_style, q_copy
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

# =====================================================================
# 설정값 (경로/하이퍼파라미터는 전부 여기서 수정)
# =====================================================================

# --- 데이터 경로 (노트북 test_SDv1.5.ipynb 와 동일한 구조) ---
PROJECT_ROOT = Path("/content/drive/MyDrive/BUHT")
DATA_ROOT = PROJECT_ROOT / "data"
SKETCH_DIR = DATA_ROOT / "sketch"        # 입력 스케치 (conditioning_image)
TARGET_DIR = DATA_ROOT / "image_txt"     # 실제 수묵화 원본 + 캡션(.txt)

# --- 생성 이미지 경로 ---
OUTPUT_ROOT = Path("/content/drive/MyDrive/korean_sketch_model")
OURS_DIR = OUTPUT_ROOT / "generated" / "ours"          # 우리 모델(ControlNet+LoRA) 생성 결과
BASELINE_DIR = OUTPUT_ROOT / "generated" / "baseline"  # 순정 SD v1.5 생성 결과

# --- 결과 저장 ---
RESULT_CSV = OUTPUT_ROOT / "evaluation_results.csv"

# --- 캡션 처리 (학습 때와 동일하게 맞춰야 CLIP score가 공정함) ---
STYLE_PREFIX = "Korean traditional painting style, "

# --- 모델/연산 설정 ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # T4에서 가볍게 돌아가는 CLIP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16          # CLIP/FID 배치 크기 (T4 16GB 기준 여유 있음)
RESOLUTION = 512         # 학습/생성 해상도 (q_struct 에서 크기 통일용)

# --- q_struct: edge 추출 설정 ---
CANNY_LOW = 100          # Canny 하한 임계값
CANNY_HIGH = 200         # Canny 상한 임계값
EDGE_DILATE_PX = 3       # edge IoU 계산 전 팽창 커널 크기(픽셀).
                         # 선이 몇 px 어긋나도 겹친 것으로 인정하는 허용 오차.

# --- q_style: FID 설정 ---
FID_MAX_IMAGES = 500     # FID 계산에 사용할 폴더당 최대 이미지 수 (메모리/시간 절약)

# --- 종합 점수 가중치 (합이 1이 되도록) ---
W_SEM = 0.30             # 텍스트 일치도
W_STRUCT = 0.30          # 구조 반영도
W_STYLE = 0.30           # 수묵화다움
W_COPY = 0.10            # 복사 억제 (1 - 복사 유사도)
FID_SCALE = 50.0         # FID를 0~1 점수로 바꿀 때의 스케일: exp(-FID/FID_SCALE)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


# =====================================================================
# 공용 헬퍼
# =====================================================================

def list_images(folder: Path) -> list[Path]:
    """폴더 안의 이미지 파일 목록을 파일명 순으로 반환한다."""
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb(path: Path) -> Image.Image:
    """EXIF 회전을 반영해 RGB로 이미지를 연다."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def read_caption(stem: str) -> str | None:
    """
    TARGET_DIR 에서 {stem}.txt 캡션을 읽어 학습 때와 동일하게
    STYLE_PREFIX 를 붙여 반환한다. 없으면 None.
    """
    txt_path = TARGET_DIR / f"{stem}.txt"
    if not txt_path.exists():
        return None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            text = txt_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = txt_path.read_text(encoding="utf-8", errors="replace")
    compact = " ".join(text.split())
    return f"{STYLE_PREFIX}{compact}".strip()


_clip_cache: dict = {}


def _get_clip():
    """CLIP 모델/프로세서를 1회만 로드해 재사용한다 (q_sem, q_copy 공용)."""
    if "model" not in _clip_cache:
        from transformers import CLIPModel, CLIPProcessor  # 지연 import
        _clip_cache["model"] = (
            CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
        )
        _clip_cache["processor"] = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _clip_cache["model"], _clip_cache["processor"]


@torch.no_grad()
def _clip_image_embeddings(paths: list[Path]) -> torch.Tensor:
    """이미지 목록을 CLIP 이미지 임베딩(L2 정규화됨)으로 변환한다. shape (N, D)"""
    model, processor = _get_clip()
    chunks = []
    for i in range(0, len(paths), BATCH_SIZE):
        images = [load_rgb(p) for p in paths[i : i + BATCH_SIZE]]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        emb = model.get_image_features(**inputs)
        chunks.append(emb / emb.norm(dim=-1, keepdim=True))
    return torch.cat(chunks, dim=0)


# =====================================================================
# q_sem — 텍스트-이미지 일치도 (CLIP score)
# =====================================================================

@torch.no_grad()
def q_sem(gen_dir: Path = OURS_DIR) -> dict:
    """
    생성 이미지와 해당 캡션 사이의 CLIP 코사인 유사도 평균을 계산한다.

    측정 대상: "프롬프트(캡션)가 말하는 내용이 그림에 실제로 담겼는가"
    - 높을수록: 텍스트 조건을 잘 따른 것 (보통 0.25~0.35 정도면 양호)
    - 낮을수록: 프롬프트와 무관한 그림이 나온 것 (조건 무시 / 모드 붕괴 의심)

    캡션은 파일명 stem 으로 TARGET_DIR/{stem}.txt 에서 찾는다.
    """
    model, processor = _get_clip()
    gen_paths = list_images(gen_dir)

    # 캡션이 존재하는 이미지만 평가 대상으로 삼는다.
    pairs = []
    for p in gen_paths:
        caption = read_caption(p.stem)
        if caption is None:
            print(f"[q_sem] 캡션 없음, 건너뜀: {p.name}")
            continue
        pairs.append((p, caption))

    if not pairs:
        raise RuntimeError(f"[q_sem] 캡션과 매칭되는 생성 이미지가 없습니다: {gen_dir}")

    scores = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        images = [load_rgb(p) for p, _ in batch]
        texts = [c for _, c in batch]
        inputs = processor(
            images=images, text=texts, return_tensors="pt",
            padding=True, truncation=True,
        ).to(DEVICE)
        img_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_emb = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        # 같은 인덱스끼리(이미지 i ↔ 캡션 i)의 코사인 유사도
        scores.extend((img_emb * txt_emb).sum(dim=-1).cpu().tolist())

    return {
        "mean": float(np.mean(scores)),   # 종합 점수에 쓰는 대표값
        "std": float(np.std(scores)),
        "n": len(scores),
    }


# =====================================================================
# q_struct — 구조 반영도 (생성 이미지 edge vs 입력 스케치)
# =====================================================================

def q_struct(gen_dir: Path = OURS_DIR, sketch_dir: Path = SKETCH_DIR) -> dict:
    """
    생성 이미지에서 Canny edge 를 추출해 입력 스케치의 edge 와 비교한다.

    측정 대상: "ControlNet 이 스케치의 윤곽/구도를 얼마나 지켰는가"
    - edge IoU 높을수록: 스케치 선을 잘 따라 그린 것 (구조 보존 양호)
    - 낮을수록: 스케치를 무시하고 자유롭게 그린 것 (ControlNet 효과 약함)

    비교 방법:
    1) 두 이미지 모두 RESOLUTION 크기로 통일 후 Canny edge 추출
       (스케치도 같은 Canny 를 거쳐 동일 조건에서 비교)
    2) 선이 몇 px 어긋나는 것은 허용하도록 EDGE_DILATE_PX 만큼 팽창 후 IoU 계산
    3) 참고용으로 edge map 간 SSIM 도 함께 계산
    """
    import cv2  # 지연 import (Colab 기본 포함)
    from skimage.metrics import structural_similarity as ssim

    sketch_index = {p.stem: p for p in list_images(sketch_dir)}
    kernel = np.ones((EDGE_DILATE_PX, EDGE_DILATE_PX), np.uint8)

    def edges(path: Path) -> np.ndarray:
        gray = np.array(load_rgb(path).resize((RESOLUTION, RESOLUTION)).convert("L"))
        return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    ious, ssims = [], []
    for gen_path in list_images(gen_dir):
        sketch_path = sketch_index.get(gen_path.stem)
        if sketch_path is None:
            print(f"[q_struct] 스케치 없음, 건너뜀: {gen_path.name}")
            continue

        gen_edge = edges(gen_path)
        sketch_edge = edges(sketch_path)

        # 팽창(dilate)으로 위치 오차 허용 후 IoU (교집합/합집합)
        g = cv2.dilate(gen_edge, kernel) > 0
        s = cv2.dilate(sketch_edge, kernel) > 0
        union = np.logical_or(g, s).sum()
        iou = float(np.logical_and(g, s).sum() / union) if union > 0 else 0.0
        ious.append(iou)

        # edge map 자체의 구조 유사도 (보조 지표)
        ssims.append(float(ssim(gen_edge, sketch_edge)))

    if not ious:
        raise RuntimeError(f"[q_struct] 스케치와 매칭되는 생성 이미지가 없습니다: {gen_dir}")

    return {
        "edge_iou_mean": float(np.mean(ious)),  # 종합 점수에 쓰는 대표값
        "ssim_mean": float(np.mean(ssims)),
        "n": len(ious),
    }


# =====================================================================
# q_style — 수묵화다움 (실제 수묵화 집단과의 FID)
# =====================================================================

@torch.no_grad()
def q_style(gen_dir: Path = OURS_DIR, real_dir: Path = TARGET_DIR) -> dict:
    """
    생성 이미지 집단과 실제 수묵화(학습 데이터) 집단 사이의 FID 를 계산한다.

    측정 대상: "생성 결과의 전체적인 질감/색감/화풍 분포가 진짜 수묵화와 비슷한가"
    - 낮을수록: 실제 수묵화 분포에 가까움 (수묵화다움 높음)
    - 높을수록: 분포가 동떨어짐 (사진풍/서양화풍 등 스타일 이탈)

    주의: FID 는 이미지 수가 적으면(수십 장 이하) 값이 불안정하고
    과대평가되는 경향이 있다. ours vs baseline 처럼 같은 조건끼리의
    상대 비교 용도로 해석할 것.
    """
    # pip install torchmetrics[image] 필요 (Inception-v3 특징 추출용)
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)

    def feed(paths: list[Path], real: bool):
        for i in range(0, len(paths), BATCH_SIZE):
            batch = [
                np.array(load_rgb(p).resize((299, 299)))  # Inception 입력 크기
                for p in paths[i : i + BATCH_SIZE]
            ]
            tensor = (
                torch.from_numpy(np.stack(batch))
                .permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                .to(DEVICE, dtype=torch.uint8)
            )
            fid.update(tensor, real=real)

    real_paths = list_images(real_dir)[:FID_MAX_IMAGES]
    gen_paths = list_images(gen_dir)[:FID_MAX_IMAGES]
    if not real_paths or not gen_paths:
        raise RuntimeError(f"[q_style] 이미지가 부족합니다: real={len(real_paths)}, gen={len(gen_paths)}")

    feed(real_paths, real=True)
    feed(gen_paths, real=False)

    return {
        "fid": float(fid.compute().item()),  # 종합 점수에 쓰는 대표값 (낮을수록 좋음)
        "n_real": len(real_paths),
        "n_gen": len(gen_paths),
    }


# =====================================================================
# q_copy — 학습 데이터 복사(과적합) 여부
# =====================================================================

@torch.no_grad()
def q_copy(gen_dir: Path = OURS_DIR, real_dir: Path = TARGET_DIR, top_k: int = 5) -> dict:
    """
    각 생성 이미지에 대해 학습 데이터에서 가장 유사한 이미지를 찾아
    CLIP 코사인 유사도를 측정한다.

    측정 대상: "모델이 학습 이미지를 거의 그대로 재현(암기)하고 있지 않은가"
    - 낮을수록: 학습 데이터를 베끼지 않고 새로운 그림을 만든 것 (바람직)
    - 높을수록(특히 0.95 이상): 특정 학습 이미지를 복사했을 가능성 (과적합 의심)
      → top_matches 에 찍힌 (생성 파일, 최근접 학습 파일) 쌍을 눈으로 확인할 것.

    참고: 화풍이 같은 도메인이므로 0.7~0.85 정도의 유사도는 자연스러운 수준이다.
    절대값보다 baseline 대비 얼마나 높은지, 그리고 상위 쌍의 실제 모습이 중요하다.
    """
    gen_paths = list_images(gen_dir)
    real_paths = list_images(real_dir)
    if not gen_paths or not real_paths:
        raise RuntimeError(f"[q_copy] 이미지가 부족합니다: gen={len(gen_paths)}, real={len(real_paths)}")

    gen_emb = _clip_image_embeddings(gen_paths)    # (G, D)
    real_emb = _clip_image_embeddings(real_paths)  # (R, D)

    # 모든 (생성, 학습) 쌍의 코사인 유사도 → 각 생성 이미지의 최근접 학습 이미지
    sim = gen_emb @ real_emb.T                     # (G, R)
    max_sim, max_idx = sim.max(dim=1)

    # 유사도가 높은 순으로 top_k 쌍을 기록 (육안 검증용)
    order = torch.argsort(max_sim, descending=True)[:top_k]
    top_matches = [
        {
            "generated": gen_paths[i].name,
            "nearest_train": real_paths[max_idx[i]].name,
            "similarity": round(float(max_sim[i]), 4),
        }
        for i in order.tolist()
    ]

    return {
        "mean_max_sim": float(max_sim.mean()),  # 종합 점수에 쓰는 대표값
        "max_sim": float(max_sim.max()),        # 최악(가장 복사에 가까운) 케이스
        "top_matches": top_matches,
        "n": len(gen_paths),
    }


# =====================================================================
# 종합 점수 및 비교 리포트
# =====================================================================

def weighted_score(sem: float, struct: float, fid: float, copy_sim: float) -> float:
    """
    4개 지표를 0~1 스케일로 정규화한 뒤 가중 합산한다. 높을수록 좋음.

    - s_sem   : CLIP 유사도는 이미 0~1 (실질 0.2~0.4 대역) → 그대로 사용
    - s_struct: edge IoU 는 이미 0~1 → 그대로 사용
    - s_style : FID 는 낮을수록 좋으므로 exp(-FID/FID_SCALE) 로 변환 (0~1, 높을수록 좋음)
    - s_copy  : 복사 유사도는 낮을수록 좋으므로 1 - 유사도 로 변환

    절대적인 의미보다는 같은 정규화를 거친 ours vs baseline 의
    상대 비교용 점수로 해석해야 한다.
    """
    s_sem = max(0.0, min(1.0, sem))
    s_struct = max(0.0, min(1.0, struct))
    s_style = float(np.exp(-fid / FID_SCALE))
    s_copy = max(0.0, min(1.0, 1.0 - copy_sim))
    return W_SEM * s_sem + W_STRUCT * s_struct + W_STYLE * s_style + W_COPY * s_copy


def evaluate_model(name: str, gen_dir: Path) -> dict:
    """한 모델의 생성 폴더에 대해 4개 지표를 모두 계산해 dict 로 반환한다."""
    print(f"\n===== [{name}] 평가 시작: {gen_dir} =====")

    sem = q_sem(gen_dir)
    print(f"  q_sem   (CLIP score, ↑): {sem['mean']:.4f} (n={sem['n']})")

    struct = q_struct(gen_dir)
    print(f"  q_struct(edge IoU,  ↑): {struct['edge_iou_mean']:.4f} / SSIM {struct['ssim_mean']:.4f} (n={struct['n']})")

    style = q_style(gen_dir)
    print(f"  q_style (FID,       ↓): {style['fid']:.2f} (real={style['n_real']}, gen={style['n_gen']})")

    copy = q_copy(gen_dir)
    print(f"  q_copy  (최근접 유사도, ↓): mean {copy['mean_max_sim']:.4f} / max {copy['max_sim']:.4f}")
    for m in copy["top_matches"]:
        print(f"    - {m['generated']} ↔ {m['nearest_train']} (sim={m['similarity']})")

    total = weighted_score(
        sem["mean"], struct["edge_iou_mean"], style["fid"], copy["mean_max_sim"]
    )
    print(f"  weighted score (↑): {total:.4f}")

    return {
        "model": name,
        "q_sem": round(sem["mean"], 4),
        "q_struct_iou": round(struct["edge_iou_mean"], 4),
        "q_struct_ssim": round(struct["ssim_mean"], 4),
        "q_style_fid": round(style["fid"], 2),
        "q_copy_mean": round(copy["mean_max_sim"], 4),
        "q_copy_max": round(copy["max_sim"], 4),
        "weighted_score": round(total, 4),
    }


def main():
    """
    OURS_DIR / BASELINE_DIR 중 존재하는 폴더를 각각 평가하고,
    비교표를 print + CSV 로 저장한다.
    """
    import csv

    targets = [("ours", OURS_DIR), ("baseline_sd15", BASELINE_DIR)]
    rows = []
    for name, folder in targets:
        if not folder.exists() or not list_images(folder):
            print(f"[main] 폴더가 없거나 비어 있어 건너뜀: {name} ({folder})")
            continue
        rows.append(evaluate_model(name, folder))

    if not rows:
        raise RuntimeError("평가할 생성 이미지 폴더가 없습니다. OURS_DIR/BASELINE_DIR 를 확인하세요.")

    # --- 비교표 출력 ---
    columns = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print("\n===== 최종 비교표 =====")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in columns))
    print("\n(↑: q_sem, q_struct, weighted_score 는 높을수록 좋음 / ↓: q_style_fid, q_copy 는 낮을수록 좋음)")

    # --- CSV 저장 ---
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"결과 CSV 저장: {RESULT_CSV}")


if __name__ == "__main__":
    main()
