"""De-duplicate COMBINED_REPORT.md per user instructions:
- 장애물 환경 설계: keep Part I 5, replace Part II 2 with a pointer
- segfault / loop-spawn / logging bugs: move Part II 3 to a back appendix; drop Part I 3.2 & 5.5
- 재현 커맨드 / 핵심 교훈 / 구현 산출물 / 개요: merge to one each
- remove inner per-report titles and the redundant Part I 6 (round-2 plan, superseded)
- delete every § symbol
Operates on level-1/2 heading blocks so whole sections move atomically; data tables untouched.
"""
import re, os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "COMBINED_REPORT.md")
text = open(P, encoding="utf-8").read()
lines = text.split("\n")

# split into blocks at level-1/2 headings (### / #### stay inside their ## block)
# IMPORTANT: ignore '#' lines inside ``` code fences (bash comments look like headings)
hidx = []
in_fence = False
for i, l in enumerate(lines):
    if l.lstrip().startswith("```"):
        in_fence = not in_fence
        continue
    if not in_fence and re.match(r"^#{1,2}\s", l):
        hidx.append(i)
blocks = []
if hidx and hidx[0] > 0:
    blocks.append("\n".join(lines[:hidx[0]]))           # preamble (yaml)
for k, start in enumerate(hidx):
    end = hidx[k + 1] if k + 1 < len(hidx) else len(lines)
    blocks.append("\n".join(lines[start:end]))

def head(b):
    return b.split("\n", 1)[0]

def find(sub, occ=1):
    c = 0
    for i, b in enumerate(blocks):
        if sub in head(b):
            c += 1
            if c == occ:
                return i
    raise ValueError(f"NOT FOUND: {sub!r} #{occ}")

PB = "```{=openxml}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```\n"

# ---- 1) capture Part II 3 (bugs) for the appendix, then drop in place ----
i_bugs = find("## 3. 환경 구축 시 해결한 버그")
bugs_block = blocks[i_bugs]
# relabel its top heading; keep the ### 3.x subsections as-is
bugs_body = bugs_block.split("\n", 1)[1]
bug_appendix = (
    PB + "\n# 부록 A — 엔지니어링 노트 (버그 수정)\n\n"
    "> 환경 구축 중 Linux box2d-py에서 해결한 핵심 버그 3종(SEGFAULT · loop-spawn · 로깅)이다. "
    "본문 흐름과 무관한 엔지니어링 세부라 부록으로 분리한다.\n\n"
    "## A. 환경 구축 시 해결한 버그 (엔지니어링 핵심)\n" + bugs_body
)

# ---- 2) authored merged blocks ----
overview_merged = blocks[find("## 1. 개요 / 목표", 1)]  # keep Part I overview as-is

lessons_merged = """## 핵심 교훈 (종합)

베이스 복구(Part I)와 장애물·3-action 실험(Part II)에서 얻은 교훈을 한데 모은다.

1. **PPO 붕괴는 거의 항상 신뢰영역(KL) 제어 실패** — KL은 epoch가 아니라 **업데이트(minibatch) 단위**로 검사하라.
2. **하나의 증상 숫자가 여러 독립 버그를 가린다** — 붕괴(reward −17) 뒤에 KL·LR·죽은 속도보상 3개가 동시에 있었다.
3. **shaped reward ≠ clean reward** — 스케일업·모델 선택은 반드시 clean 평가로 게이팅하라(shaped 838이 clean 667).
4. **보상 셰이핑 항은 조용한 no-op일 수 있다 — 실제로 발화하는지 검증하라.** 속도 보상(`info['speed']`=0)과 코너 패널티(미구현)가 둘 다 무발화였다.
5. **픽셀 입력 에이전트는 장애물을 관측에 직접 "그려" 넣어야 한다** — 물리 세계에만 두면 보이지 않는다.
6. **Linux box2d-py는 body 파괴 중 contact 콜백에서 segfault** — listener를 먼저 detach하라(macOS가 가려서 로컬에선 안 잡힘).
7. **벡터 env 로깅은 `infos[key][i]`(dict-of-arrays)로 읽어야** 한다 — `infos.get(i)`(정수키)는 무음 실패.
8. **셰이핑을 추가하기 전에 실패 모드를 측정하라.** "급커브 이탈"은 오진이었고, 계측(코너가속·off%·충돌/에피소드)이 진짜 원인(충돌)을 드러냈다.
9. **샘플링 vs 결정론 평가는 std가 클 때 크게 다르다** — 샘플링 노이즈가 장애물 충돌을 유발(seed53 결정론 486 vs 샘플링 −36).
10. **"노이즈가 원인"은 가설로 끝내지 말고 직접 검증하라.** `ent_coef`로 노이즈를 줄여봤더니(entropy 1.3→0.91) 충돌이 안 줄었다 — 일부는 노이즈로 제거 안 되는 **구조적 실패**.
11. **이 태스크는 전반부에 정점** 후 정체 — 코너 패널티·스텝 증가·노이즈 축소 **세 레버 모두 clean ~415 천장을 못 넘음**. 다음 개선은 하이퍼파라미터가 아니라 **표현/아키텍처**(명시적 장애물 채널, recurrent, 더 큰 CNN) 쪽이어야 한다.
12. **action 파라미터화는 공짜가 아니다 — 2-action의 ActionWrapper가 유용한 inductive bias.** native 3-action은 동일 조건에서 clean 229(2-action 415의 ~55%): 독립 gas/brake가 동시입력 퇴화영역·무방비 brake·~2.9배 탐색부피를 만들어 std 발산/저성능을 유발.
13. **하이퍼파라미터는 차원에 따라 재튜닝하라.** 2D에서 안정적이던 `ent_coef 0.01`이 3D에선 entropy를 발산(2.0→5.33)시켰다 — `ent항/policy항` 비율로 균형을 확인하라.
"""

artifacts_merged = """## 구현 산출물 (커밋 안 함)

| 파일 | 상태 | 내용 |
|---|---|---|
| `src/ppo_agent_2.py` | 신규 | 2-action PPO fork — per-minibatch KL early-stop + env-step 코사인 LR |
| `src/car_racing_obstacles.py` | 신규 | `CarRacingObstacles-v0` 환경 (랜덤 크기·픽셀 렌더·충돌 패널티·segfault/loop-spawn 픽스) |
| `src/env_wrappers.py` | 수정 | `ActionWrapper`(2D `[steer,throttle]`→3D `[steer,gas,brake]`) 추가 |
| `scripts/train_ppo_2action2.py` | 신규 | 베이스 2-action 학습 (LR/안정화 config, `RewardShapingWrapper` 인라인) |
| `scripts/train_ppo_2action_obstacles.py` | 신규 | 장애물 회피 학습 fork (코너 패널티·로깅 픽스·CLI 다수) |
| `scripts/train_ppo_3action_obstacles.py` | 신규 | 3-action 대조군 학습 (ActionWrapper 미적용, native 3D, evaluated641서 fine-tune) |
| `scripts/evaluate_agent_2action.py` | 분리(베이스) | 장애물 코드 제거 → 베이스(CarRacing-v3) 전용 clean 평가기 |
| `scripts/evaluate_agent_2action_obstacles.py` | 신규 | 장애물 전용 평가기 (`CarRacingObstacles-v0` 항상 ON) |
| `scripts/evaluate_agent_3action_obstacles.py` | 신규 | 3-action 장애물 평가기 |
| `scripts/record_video.py` | 신규 | 평가 리플레이 → mp4 녹화 |
| `scripts/dump_tb_scalars.py` | 신규 | TensorBoard 이벤트 → 텍스트 표 + PNG 덤프 (다중 run 오버레이) |
| `scripts/make_report_figs.py`, `build_combined_report.py` | 신규 | 보고서 그림·합본 생성 도구 |
| `requirements.txt`, `Dockerfile`, `build_docker.sh` | 신규 | 재현 환경 (pip / 도커 이미지) |

**학습 스크립트 CLI 오버라이드**: `--checkpoint --steps --seed --log-dir --save-dir --n-obstacles --obstacle-size-min --obstacle-size-max --track-penalty --accel-turn-weight --ent-coef`
"""

repro_merged = """## 부록 — 재현 커맨드

> 학습=원격 GPU(conda `teamB_env`), 평가/녹화=로컬(conda `racing`). 스크립트는 저장소 루트에서 `python -m scripts.NAME`으로 실행.

```bash
# ===== 학습 =====
# 베이스 2-action (6M)
python -m scripts.train_ppo_2action2 --steps 6000000
#   스케일업: 6M best 체크포인트에서 total_timesteps만 20M으로 올려 재개 → ppo_2action4/best_model.pth (9.83M)

# 장애물 회피 2-action (corner penalty 0.2, 랜덤 크기)
python -m scripts.train_ppo_2action_obstacles \\
  --accel-turn-weight 0.2 --obstacle-size-min 0.25 --obstacle-size-max 0.6 \\
  --save-dir ./models/obs_p02 --log-dir ./logs/obs_p02

# 3-action 대조군 (ent_coef 0.003 — 3D 발산 방지)
python -m scripts.train_ppo_3action_obstacles \\
  --ent-coef 0.003 --obstacle-size-min 0.25 --obstacle-size-max 0.6 \\
  --save-dir ./models/obs_3action --log-dir ./logs/obs_3action

# 대형 장애물(도로보다 큼) 실험 — off-road 우회가 유일 통과법이라 track_penalty 완화 필요
python -m scripts.train_ppo_2action_obstacles \\
  --obstacle-size-min 2.0 --obstacle-size-max 3.0 --track-penalty 0.2 \\
  --save-dir ./models/obs_big --log-dir ./logs/obs_big

# ===== 평가 (clean) =====
python -m scripts.evaluate_agent_2action \\
  --model ./models/ppo_2action4/best_model.pth --episodes 100 --seed 42            # 베이스
python -m scripts.evaluate_agent_2action_obstacles \\
  --model ./models/obs_small/best_model.pth \\
  --obstacle-size-min 0.25 --obstacle-size-max 0.6 --episodes 50 --seed 42         # 장애물(학습과 동일 크기 분포로!)
python -m scripts.evaluate_agent_3action_obstacles \\
  --model ./models/obs_3action/best_model.pth \\
  --obstacle-size-min 0.25 --obstacle-size-max 0.6 --episodes 50 --seed 42         # 3-action

# ===== 영상 녹화 (mp4) =====
python scripts/record_video.py --model ./models/ppo_2action4/best_model.pth --episodes 5
python scripts/record_video.py --model ./models/obs_small/best_model.pth \\
  --obstacles --obstacle-size-min 0.25 --obstacle-size-max 0.6 --episodes 5
```
"""

part1_6_pointer = """## 6. 코너 감속 개선

코너 감속(accel-turn) 패널티는 Part I 작성 시점엔 *구현만 완료·미학습*이었다. 이후 **실제 학습·진단을 Part II 5절에서 수행**했고, 결론은 "코너 패널티는 비(非)레버"(0.5는 reward만 깎고 이탈은 충돌이 원인)였다. 상세는 Part II 5절 참조.
"""

part2_2_pointer = """## 2. 환경 설계 — `CarRacingObstacles-v0`

장애물 환경의 설계(등록·기본값, 픽셀 가시성, 배치·재현성, 충돌 검출·패널티, 크기 랜덤화, segfault·loop-spawn 픽스)는 **Part I 5절에 코드와 함께 통합**했다. 본 Part II는 그 환경 위에서 수행한 실험·진단을 다룬다. 장애물 크기 변형(소형 vs 대형) 실험은 아래 4절 참조.
"""

# ---- 3) apply edits (order chosen so anchors stay valid) ----
# trim Part I 3.2 (tail of "## 3. 문제 진단 및 해결")
i = find("## 3. 문제 진단 및 해결")
blocks[i] = blocks[i].split("### 3.2 장애물 환경 도입")[0].rstrip() + "\n\n---\n"
# trim Part I 5.5 (tail of "## 5. 장애물 환경 설계")
i = find("## 5. 장애물 환경 설계")
blocks[i] = blocks[i].split("### 5.5 재현성 segfault")[0].rstrip() + "\n\n---\n"
# trim Part I 9.3 from "## 9. 부록" (keep 9.1, 9.2)
i = find("## 9. 부록")
blocks[i] = blocks[i].split("### 9.3 재현 커맨드")[0].rstrip() + "\n"
# Part I 6 -> pointer
blocks[find("## 6. 코너 감속")] = part1_6_pointer
# Part II 2 -> pointer
blocks[find("## 2. 환경 설계 —")] = part2_2_pointer
# Part II 9/10/11 -> merged
blocks[find("## 9. 핵심 교훈")] = lessons_merged
blocks[find("## 10. 구현 산출물")] = artifacts_merged
blocks[find("## 11. 부록 — 재현 커맨드")] = repro_merged

# drop blocks (by exact head match); collect indices then delete descending
drop_heads = [
    ("# AICarRacing 기술 보고서 —", 1),       # Part I inner title
    ("# AICarRacing 장애물 회피 실험 보고서 —", 1),  # Part II inner title
    ("## 1. 개요 / 목표", 2),                  # Part II overview (dup)
    ("## 7. 구현 산출물", 1),                  # Part I artifacts (merged into II)
    ("## 8. 핵심 교훈 / 다음 단계", 1),         # Part I lessons (merged into II)
]
drop_idx = sorted({find(s, o) for s, o in drop_heads}, reverse=True)
for di in drop_idx:
    del blocks[di]

# remove the bugs block (Part II 3) and re-insert as appendix before "부록 Z"
del blocks[find("## 3. 환경 구축 시 해결한 버그")]
zi = find("# 부록 Z")
blocks.insert(zi, bug_appendix)

out = "\n".join(blocks)
# normalize >3 blank lines, delete all § (and any following space)
out = re.sub(r"§\s*", "", out)
out = re.sub(r"\n{4,}", "\n\n\n", out)
open(P, "w", encoding="utf-8").write(out)
print("OK. new length:", len(out.split(chr(10))), "lines")
