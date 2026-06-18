"""Reorganize COMBINED_REPORT.md per user: Part I = 2-action 구현·복구 + 3-action 비교 (무장애물),
Part II = 장애물 전부. Move obstacle content (Part I 5 환경설계, 4.2 round1) into Part II.
Condense all bug fixes to one paragraph each (base-collapse -> Part I 3; obstacle-env -> Part II env section).
Trim Part I 2 시스템 구성 (drop 환경/로컬셋업 dup with Part 0; condense pipeline). Remove appendix A.
Part I numbered inline 1..5; Part II left for the renumber pass (positional).
"""
import re, os
P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "COMBINED_REPORT.md")
text = open(P, encoding="utf-8").read()
lines = text.split("\n")

hidx, inf = [], False
for i, l in enumerate(lines):
    if l.lstrip().startswith("```"):
        inf = not inf; continue
    if not inf and re.match(r"^#{1,2}\s", l):
        hidx.append(i)
blocks = []
if hidx and hidx[0] > 0:
    blocks.append("\n".join(lines[:hidx[0]]))
for k, s in enumerate(hidx):
    e = hidx[k + 1] if k + 1 < len(hidx) else len(lines)
    blocks.append("\n".join(lines[s:e]))

def head(b): return b.split("\n", 1)[0]
def B(sub, occ=1):
    c = 0
    for b in blocks:
        if sub in head(b):
            c += 1
            if c == occ: return b
    raise ValueError("NOT FOUND: " + sub)

# ---------- extract reusable pieces ----------
s2 = B("## 2. 시스템 구성")
segs = s2.split("\n### ")
def seg(key):
    for x in segs:
        if x.startswith(key): return x.split("\n", 1)[1]   # body after title line
    raise ValueError("seg " + key)
action_body = seg("2.2 2-action")
cnn_body = seg("2.3 에이전트")

s4 = B("## 4. 실험 결과")
CUT = "### 4.2 장애물 회피 (round 1)"
s4_keep = s4.split(CUT)[0].rstrip() + "\n"
round1 = CUT + s4.split(CUT)[1]
round1 = re.sub(r"> 참고: 본 절의 결과는 모두 .*?아직 학습되지 않았다\.\n+", "", round1, flags=re.S)
round1 = "## 2. 장애물 회피 학습 (round 1)\n" + round1.split("\n", 1)[1]

s5 = B("## 5. 장애물 환경 설계")
OBS_BUGS = (
"### 1.5 환경 구축 중 해결한 버그 (요약)\n\n"
"Linux box2d-py에서 세 버그를 수정했다. **(1) SEGFAULT** — 차가 장애물에 닿은 채 에피소드가 끝나 `reset()`의 `_destroy()`가 body를 파괴하면 Box2D가 반쯤 파괴된 body에 EndContact 콜백을 재진입시켜 죽었다 → 파괴 **전에** contact listener를 detach. **(2) loop-spawn** — 루프 트랙의 끝 타일이 시작선 바로 뒤라 차 위에 장애물이 스폰됨 → 후보 타일에서 **양 끝을 모두 제외**(`range(start_clear_tiles, n_tiles-start_clear_tiles)`). **(3) 로깅 무력화** — 벡터 env의 `infos`를 `infos.get(i)`(정수키)로 읽어 셰이핑 지표가 전부 미기록 → `infos[key][i]`(dict-of-arrays)로 수정. (상세 코드는 `src/car_racing_obstacles.py`·학습 스크립트 참조.)\n"
)
envdesign = "## 1. 환경 설계 — `CarRacingObstacles-v0`\n" + s5.split("\n", 1)[1].rstrip() + "\n\n" + OBS_BUGS

# ---------- authored Part I blocks ----------
PI_TITLE = "# Part I — 2-action 구현·복구 + 3-action 비교 (무장애물 트랙)\n"

PI1 = """## 1. 개요 / 목표

Part I은 무장애물 `CarRacing-v3` 트랙에서 **2-action PPO 에이전트를 구현·복구**하고, 같은 트랙에서 **native 3-action(gas/brake 독립)과 비교**하는 것을 다룬다. (도로 위 장애물을 입력에 추가한 확장은 Part II.)

작업 흐름: (1) 다운로드한 2-action 학습 라인이 PPO 붕괴(collapse)로 망가져 있어 진단·복구하고, (2) 6M→20M으로 스케일업해 clean 평가 **667**을 달성, (3) 3-action 저장 모델과 비교했다. 핵심 결과 — 복구된 2-action best(`ppo_2action4/best_model.pth`, 9.83M, shaped 837.99)가 clean 평균 **667.49**로 3-action 베이스(임베디드 shaped 674.55 / 637.95)와 동급 이상이다.
"""

PI2 = ("""## 2. 시스템 구성

> 실행 환경·버전(Python/torch/gymnasium 등)과 관측(State)·보상(Reward) 정의는 Part 0을 참조한다. 본 절은 Part I 고유의 **행동 공간(2 vs 3-action)과 신경망 구조**만 다룬다.

### 2.1 2-action vs 3-action 과 ActionWrapper
""" + action_body.rstrip() + """

### 2.2 에이전트 아키텍처 (CNN / Actor / Critic)
""" + cnn_body.rstrip() + """

### 2.3 학습 파이프라인 (요약)

래퍼 순서(내부→외부): `RewardShapingWrapper`(각 학습 스크립트에 인라인 정의) → `ActionWrapper`(2D→3D) → `GrayScaleObservation` → `TimeLimit(1000)` → `FrameStack(4)`. 64개 비동기 환경(`AsyncVectorEnv`)에서 `RolloutBuffer`(32768 = 64×512)를 채우고 minibatch 2048로 PPO를 갱신하며 per-minibatch KL early-stop으로 안정화한다(혼합정밀 `learn_mixed_precision`). PPO 목적함수·하이퍼파라미터 상세는 Part 0 2절, 최종 값 표는 5절 참조.
""")

PI3 = """## 3. 베이스 모델 붕괴 진단 및 복구

다운로드한 `models/ppo_2action2/best_model.pth`는 임베디드 shaped `mean_reward = -17.06`으로 망가져 있었다(`approx_kl`이 43.9까지 폭발, `value_loss` 753 스파이크, LR이 1e-5에 고정 — 전형적 PPO 신뢰영역 붕괴). 원인은 **세 개의 독립 버그**였다. **(1) KL early-stop이 epoch 단위로만 동작** — 큰 버퍼에서 한 epoch(약 16 갱신)를 다 돌린 *뒤에야* KL을 검사해 그 사이 정책이 신뢰영역을 한참 벗어났다 → KL 검사를 **per-minibatch**로 옮긴 2-action 전용 fork `src/ppo_agent_2.py`를 만들었다. **(2) LR 스케줄러 오호출** — `update_learning_rate`에 `progress_remaining`을 잘못 넘겨 progress가 항상 ~0이라 LR이 1e-5에 고정(+ 마지막 스텝 `ZeroDivisionError`) → **env-step 기반**으로 재작성해 코사인 1e-4→1e-5를 실제로 스윕. **(3) 죽은 속도 보상** — `info['speed']`가 항상 0이라 전진 유도 보상이 한 번도 발화하지 않음 → `car.hull.linearVelocity`로 직접 계산(weight 0.003). 추가 안정화로 LR 3e-4→1e-4, clip 0.2→0.15, target_kl 0.015→0.03(per-minibatch), std 0.4→0.5, envs 128→64를 적용했다. (코드 상세는 `src/ppo_agent_2.py` 참조.)
"""

# Part I 5 부록 (was 7) — rename headings 7->5
PI5 = B("## 7. 부록").replace("## 7. 부록", "## 5. 부록 — 하이퍼파라미터 · 버전", 1).replace("### 7.1 ", "### 5.1 ").replace("### 7.2 ", "### 5.2 ")

# rename kept §4 results heading (stays 4)
PI4 = s4_keep  # "## 4. 실험 결과 ..." already; keep

PII_TITLE = ("""# Part II — 장애물(obstacle) 입력 추가

> Part I의 무장애물 베이스 위에, 도로에 **무작위 정적 장애물**을 올려 픽셀 관측에 그려 넣고 회피를 학습한다. 환경 구축 → round-1 학습 → 크기 실험 → 코너-이탈 오진 진단 → 행동 노이즈(ent_coef) → native 3-action 대조까지 다룬다.
""")

# ---------- reassemble ----------
new = [
    blocks[0],                                   # preamble (yaml)
    B("# AICarRacing 종합 기술 보고서 {-}"),       # master title block
    B("# Part 0"),
    B("## 0. 실행 환경"),
    B("## 1. 문제 및 환경 정의"),
    B("## 1-A. State"),
    B("## 1-B. Reward"),
    B("## 2. 강화학습 알고리즘"),
    PI_TITLE, PI1, PI2, PI3, PI4, PI5,
    PII_TITLE, envdesign, round1,
    B("## 2. 장애물 크기 실험"),
    B("## 3. 진단 — "),
    B("## 4. 실험 결과 종합"),
    B("## 5. 마지막 실험 — "),
    B("## 6. 3-action 대조 실험"),
    B("## 7. 핵심 교훈"),
    B("## 8. 구현 산출물"),
    B("## 9. 부록 — 재현 커맨드"),
    B("# 부록 Z"),
]
out = "\n".join(b.rstrip("\n") + "\n" for b in new)
out = re.sub(r"\n{4,}", "\n\n\n", out)
open(P, "w", encoding="utf-8").write(out)
print("restructure done. lines:", len(out.split(chr(10))))
