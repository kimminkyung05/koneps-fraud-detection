# koneps-fraud-detection

나라장터 공공데이터 기반 불공정 조달행위 이상 탐지 및 예측 모델 — **DACON 236754 나라장터 입찰공고 위반 탐지**

나라장터 입찰공고 1건마다 24개 위반 항목(v1~v24)의 위반 여부(0/1)와 근거 문구를 판정합니다.
고정 모델 **Gemma 4 26B**(`google/gemma-4-26B-A4B-it`)를 학습 없이 쓰고, **RAG(유사 사례 검색)·프롬프트·규칙**으로 성능을 올립니다.

## 구성

| 경로 | 내용 |
|---|---|
| `dacon_final_pipeline.ipynb` | 최종 파이프라인: 사례은행 생성 → dev 평가 → pseudo 사례 → 제출 형식 검사 → `submit.zip` |
| `autolabel.py` | 규칙 기반 자동 라벨러 (`unlabeled.csv`의 자동 라벨 19,600행을 만든 코드) |
| `docs/데이터 라벨링.md` | 라벨링 방법 — 사람 라벨 400행(직접 라벨링·검토 반영)과 자동 라벨 19,600행 |
| `docs/추론.md` | 추론 파이프라인 방법론 — 하이브리드 검색, Contextual Retrieval, 구조 기반 청킹 등과 측정값 |

## 저장소에 없는 파일 (직접 준비)

대회 데이터는 재배포할 수 없어 `.gitignore`로 제외했습니다. 이 폴더에 아래 파일을 두고 실행하세요.

| 파일 | 출처 |
|---|---|
| `train_unlabeled.jsonl.gz`, `dev.jsonl.gz`, `dev_labels.csv`, `sample_submission.csv`, `data/`, `baseline/` | DACON 대회 페이지의 `open.zip` |
| `unlabeled.csv` (2만 행 라벨), `human_labels.csv` (사람 라벨 400행) | 팀 내부 공유(공고 원문 근거 문구가 들어 있어 저장소에서 제외) |

## 실행

**Colab (A100 권장)**
1. `dacon_final_pipeline.ipynb`를 Colab에서 열고 A100 런타임 선택
2. 보안 비밀(Secrets)에 `HF_TOKEN` 등록 (Gemma는 Hugging Face에서 라이선스 동의 필요)
3. 모두 실행 → 업로드 창에서 `open.zip`, `unlabeled.csv`, `human_labels.csv` 선택
4. 결과: Google Drive `pps_dacon/`에 저장되고 `submit.zip`이 자동으로 다운로드됨

**GPU 서버**: 이 폴더에서 노트북을 열고 실행하면 폴더의 데이터를 그대로 읽습니다. VRAM에 따라 프로필이 자동으로 선택됩니다.

| GPU 프로필 | 모델 | 양자화 | 컨텍스트 |
|---|---|---|---|
| `a100` (VRAM ≥ 38GB) | 원본 `gemma-4-26B-A4B-it` | `int8_per_channel_weight_only` (평가 서버와 동일) | 16384 |
| `rtx3090` (24GB) | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | `awq_marlin` | 12288 |

빠른 흐름 점검은 노트북 1번 셀에서 `DEV_LIMIT = 20`, `RUN_PSEUDO = False`, `EVAL_HUMAN_HOLDOUT = False`.

## 대회 제약

- 고정 LLM 사용, 학습·LoRA 없음, 제출 ZIP에 모델 가중치 없음, 평가 시 인터넷 차단
- 제출 `submission.csv` 49열(`id, v1~v24, e1~e24`), 근거는 원문 부분문자열 500자 이하, 부재 탐지 항목(v10·v11·v16·v18·v20)은 근거 빈칸
- dev 점수를 잴 때는 dev 정답 사례를 검색에서 제외(누수 방지)

## 현재 상태

- 노트북 전체 셀은 GPU 없이 가짜 모델로 끝까지 실행해 오류가 없음을 확인했습니다. **실제 Gemma dev 점수는 GPU에서 실행해야 나옵니다.**
- 참고 기준: 규칙만 0.03, 기존 V1 베이스라인 0.1138 (dev Macro F1)
