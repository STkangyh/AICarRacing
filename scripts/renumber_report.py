"""Renumber COMBINED_REPORT.md sections to be contiguous (fix gaps from dedup), and
remap in-text cross-references accordingly. Part 0 left as-is (no gaps).
- Part I:  9 부록 -> 7 부록 (9.1/9.2 -> 7.1/7.2)
- Part II: positional 1..9 for the ## sections (old 2,4,5,6,7,8 + 3 trailing), with ###
           subsections renumbered to <parent>.<seq>.
"""
import re, os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "COMBINED_REPORT.md")
text = open(P, encoding="utf-8").read()

# ---- 1) Part I 9 부록 -> 7 (do first, while "## 9." is unique to Part I) ----
text = text.replace("## 9. 부록\n", "## 7. 부록\n")
text = text.replace("### 9.1 최종 안정화 하이퍼파라미터", "### 7.1 최종 안정화 하이퍼파라미터")
text = text.replace("### 9.2 환경 버전 매트릭스 (재명시)", "### 7.2 환경 버전 매트릭스 (재명시)")

# ---- 2) Part II positional renumber (## -> 1..N, ### -> parent.seq) ----
lines = text.split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("# Part II — 장애물 심화"))
end = next(i for i, l in enumerate(lines) if l.startswith("# 부록 A"))
in_fence = False
h2 = 0
h3 = 0
for i in range(start, end):
    l = lines[i]
    if l.lstrip().startswith("```"):
        in_fence = not in_fence
        continue
    if in_fence:
        continue
    if l.startswith("## "):                      # level-2 section
        h2 += 1
        h3 = 0
        title = re.sub(r"^##\s+(?:\d+\.\s*)?", "", l)
        lines[i] = f"## {h2}. {title}"
    elif l.startswith("### ") and re.match(r"^###\s+\d+\.\d+", l):  # numbered subsection
        h3 += 1
        title = re.sub(r"^###\s+\d+\.\d+\s*", "", l)
        lines[i] = f"### {h2}.{h3} {title}"
text = "\n".join(lines)

# ---- 3) cross-reference fixes (explicit; new Part II map: 진단5->3, 결과6->4, ent7->5, 3action8->6) ----
refs = [
    # Part 0 0.4: "8" (3-action) -> Part II 6
    ("이 수치들은 8·장애물 보고서에 기록", "이 수치들은 Part II 6절·장애물 보고서에 기록"),
    # Part 0 3 Reward accel_turn note -> Part II 진단(3)
    ("상세는 장애물 보고서 5 진단 참조", "상세는 Part II 3절(진단) 참조"),
    # Part I 2.2 table dangling 6.3 (accel-turn impl moved) -> Part 0 Reward
    ("`action[1]=gas` (6.3 참조)", "`action[1]=gas` (Part 0 3절 Reward 참조)"),
    ("보상 셰이핑 래퍼는 그 3D를 받는다(6.3에서 역참조).",
     "보상 셰이핑 래퍼는 그 3D를 받는다(Part 0 3절 Reward에서 다룸)."),
    # Part I 6 pointer -> Part II 진단 now 3
    ("**실제 학습·진단을 Part II 5절에서 수행**", "**실제 학습·진단을 Part II 3절에서 수행**"),
    ("결론은 \"코너 패널티는 비(非)레버\"(0.5는 reward만 깎고 이탈은 충돌이 원인)였다. 상세는 Part II 5절 참조.",
     "결론은 \"코너 패널티는 비(非)레버\"(0.5는 reward만 깎고 이탈은 충돌이 원인)였다. 상세는 Part II 3절 참조."),
    # Part II 1 (env pointer): "아래 4절" (크기실험) -> 2절
    ("장애물 크기 변형(소형 vs 대형) 실험은 아래 4절 참조.", "장애물 크기 변형(소형 vs 대형) 실험은 아래 2절 참조."),
    # Part II 5 (ent_coef) intro: 진단(5절) -> 3절
    ("진단(5절)에 따르면 이 모델의 천장", "진단(3절)에 따르면 이 모델의 천장"),
    # Part II 5.3 판정: levers -> 진단3, 결과4.1, ent5
    ("코너 패널티(5절)·스텝 증가(6.1절)·노이즈 축소(7절)", "코너 패널티(3절)·스텝 증가(4.1절)·노이즈 축소(5절)"),
    # 부록 A logging note: "했다(5절)." (진단) -> Part II 3
    ("실제 진단은 **모델을 직접 계측**해서 했다(5절).", "실제 진단은 **모델을 직접 계측**해서 했다(Part II 3절)."),
]
miss = []
for a, b in refs:
    if a in text:
        text = text.replace(a, b)
    else:
        miss.append(a[:40])

open(P, "w", encoding="utf-8").write(text)
print("renumber done. cross-ref misses:", miss if miss else "none")
