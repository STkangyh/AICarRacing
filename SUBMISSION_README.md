# AICarRacing (7팀) — 제출물 안내

CarRacing-v3 기반 **2-Action PPO** 강화학습 프로젝트. 붕괴 복구 → 장애물 회피 확장 → 2 vs 3 action 대조.

## 제출물 구성
| 항목 | 파일/폴더 | 설명 |
|---|---|---|
| **설명 문서 (PDF)** | `Team7_Final_Report.pdf` | 채점용 본 보고서 (환경/State/Reward/PPO이론/코드/결과·고찰). **먼저 §0 참조** |
| 설명 문서 (Word 원본) | `COMBINED_REPORT.docx` | 위 PDF의 Word 원본 |
| **실행 환경** | `requirements.txt` | `pip install -r requirements.txt` |
| **코드** | `src/*.py`, `scripts/*.py` | 환경·에이전트·학습·평가 |
| 체크포인트 | `models/ppo_2action4/best_model.pth` (베이스, clean 667)<br>`models/obs_small/best_model.pth` (장애물, clean 365)<br>`BestSavedAgents/evaluated641.pth` (3-action 참고) | 평가용. 그 외 모델은 원격 전용(보고서 §0.4) |
| 그림·영상 | `report_assets/fig_*.png`, `frame_*.png`, **`gif_*.gif`** | 보고서 그림 + 주행 애니메이션 GIF |

## 빠른 시작 (채점자용)
```bash
# 1) 설치 (기존 실습 image에 추가). 자세한 버전은 PDF §0
pip install -r requirements.txt
#  ※ opencv-python(cv2) 필수 — 없으면 import 단계에서 실패

# 2) 베이스 평가 (장애물 없음)
python -m scripts.evaluate_agent_2action --model ./models/ppo_2action4/best_model.pth --episodes 100 --seed 42

# 3) 장애물 평가
python -m scripts.evaluate_agent_2action_obstacles --model ./models/obs_small/best_model.pth \
    --obstacle-size-min 0.25 --obstacle-size-max 0.6 --episodes 50 --seed 42
```
- 모든 스크립트는 **저장소 루트에서 모듈 방식**(`python -m scripts.NAME`)으로 실행.
- 학습 커맨드와 전체 설명은 `Team7_Final_Report.pdf` §0.3 참조.

## 메인 코드 (엔트리포인트)
- `src/car_racing_obstacles.py` — `CarRacingObstacles-v0` 환경(장애물)
- `src/ppo_agent_2.py` — PPO 에이전트(2-action fork, per-minibatch KL early-stop)
- `src/env_wrappers.py` / `src/cnn_model.py` — 관측 전처리 / CNN
- `scripts/train_ppo_2action2.py` · `train_ppo_2action_obstacles.py` · `train_ppo_3action_obstacles.py` — 학습
- `scripts/evaluate_agent_2action*.py` · `evaluate_agent_3action_obstacles.py` — 평가
- `scripts/record_video.py` — mp4 녹화

> 원본 주행 mp4(`videos_*/`)와 일부 원격 체크포인트(`obs_small_p02` 415, `obs_3action_lowent` 229)는 용량 관계로 제외했으며, 결과 수치는 보고서에 기록·재현 가능하다.

## (옵션) Docker 이미지 — 분할본 제출
개발 환경 차이로 코드가 안 도는 것을 막기 위해, 프로젝트 전체를 담은 **Docker 이미지**를 분할 압축해 함께 제출한다(교수님 안내의 옵션 항목).
- 업로드 파일: `aicarracing_team7_image.tar.gz.part_aa` ~ `part_ai` (90MB×8 + 4.5MB×1, **총 9개**)
- 이미지: `aicarracing-team7:latest` (linux/amd64, CPU torch, 평가용 체크포인트 동봉)
- **복원·실행**(채점자):
  ```bash
  cat aicarracing_team7_image.tar.gz.part_* > aicarracing_team7_image.tar.gz
  gunzip -c aicarracing_team7_image.tar.gz | docker load
  docker run --rm aicarracing-team7:latest \
      python -m scripts.evaluate_agent_2action --model ./models/ppo_2action4/best_model.pth --episodes 100 --seed 42
  ```
- 빌드 재현: `bash build_docker.sh` (`Dockerfile`·`build_docker.sh` 포함). 자세한 설명은 `Team7_Final_Report.pdf` §0.
