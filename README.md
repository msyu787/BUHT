# BUHT
# Sketch-to-Korean Ink Painting Generation

간단한 스케치와 텍스트 프롬프트를 입력하면, 스케치의 구조를 유지하면서 한국 전통 수묵화 스타일의 이미지를 생성하는 프로젝트입니다.

ControlNet으로 스케치의 형태와 구도를 반영하고, Stable Diffusion XL을 기반으로 이미지를 생성합니다. LoRA와 IP-Adapter를 활용해 수묵화 스타일을 강화하며, 여러 생성 결과 중 평가 점수가 가장 높은 이미지를 최종 결과로 선택합니다.

## 주요 기능

- 간단한 선화 또는 스케치 기반 이미지 생성
- 텍스트 프롬프트를 통한 장면 및 세부 요소 제어
- ControlNet을 활용한 스케치 구조 보존
- LoRA를 활용한 한국 전통 수묵화 스타일 학습
- IP-Adapter를 활용한 레퍼런스 이미지 스타일 반영
- VLM 기반 레퍼런스 이미지 자동 선택
- 여러 후보 이미지 생성 후 품질 평가 및 최종 이미지 선정

## Generation Pipeline

```mermaid
flowchart TD
    A["Input<br/>Sketch + Text Prompt + Reference Images"] --> B["Reference Selection<br/>VLM 기반 적합한 레퍼런스 선택"]
    B --> C["Image Generation<br/>SDXL + ControlNet + IP-Adapter + LoRA"]
    C --> D["Candidate Evaluation<br/>의미·구조·스타일·복제 정도 평가"]
    D --> E["Final Selection<br/>가중치 점수가 가장 높은 이미지 선택"]
