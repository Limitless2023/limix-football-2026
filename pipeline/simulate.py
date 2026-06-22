# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 results/pred_matrix.json（LimiX 预测）+ tournament_2026（赛制/积分）
[OUTPUT]: 蒙特卡洛 N 次模拟整届 2026 世界杯 → results/title_odds.json（晋级/夺冠概率）
[POS]: limix-football预测 的赛事推演层，把单场预测放大为整届冠军概率（对标 Prior Labs"模拟数千次"）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

机制：用 LimiX 预测比分(gh,ga)作泊松强度采样每场进球——小组赛得积分与净胜球、
淘汰赛定胜负(平局按强弱加权点球)。统一一套进球机制，前后自洽。
赛制说明：小组前2 + 8个最佳第三名晋级32强；淘汰赛按当前 Elo 种子化排布（官方具体卡位做了简化，不影响夺冠概率量级）。
"""
import os
import json
import numpy as np
from collections import defaultdict

from tournament_2026 import GROUPS, TEAMS, compute_standings, remaining_group_fixtures, CN_NAME

_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_DIR, "results")
_rng = np.random.default_rng(42)


def _load():
    with open(f"{_OUT}/pred_matrix.json") as f:
        return json.load(f)


def _lams(matrix, a, b):
    """取 a(主) vs b(客) 的预测比分作泊松强度，下限 0.15 防零。"""
    p = matrix[f"{a}|{b}"]
    return max(0.15, p["gh"]), max(0.15, p["ga"])


def _play(matrix, a, b, knockout=False):
    """模拟一场：返回 (a进球, b进球, a是否晋级[仅淘汰赛])。"""
    la, lb = _lams(matrix, a, b)
    ga, gb = int(_rng.poisson(la)), int(_rng.poisson(lb))
    if not knockout:
        return ga, gb, None
    if ga == gb:                                    # 点球：按胜负概率加权
        p = matrix[f"{a}|{b}"]
        pa = p["p_home"] / max(1e-9, p["p_home"] + p["p_away"])
        return ga, gb, (_rng.random() < pa)
    return ga, gb, (ga > gb)


def _rank(teams, tab):
    """组内/第三名排序：积分 → 净胜球 → 进球 → 随机。"""
    return sorted(teams, key=lambda t: (tab[t]["pts"], tab[t]["gd"], tab[t]["gf"], _rng.random()),
                  reverse=True)


def simulate_once(matrix, base, rem):
    # ---- 小组赛：从当前积分出发，补完剩余对阵 ----
    tab = {t: dict(base[t]) for t in TEAMS}
    for g, a, b in rem:
        ga, gb, _ = _play(matrix, a, b)
        tab[a]["gf"] += ga; tab[a]["ga"] += gb; tab[b]["gf"] += gb; tab[b]["ga"] += ga
        if ga > gb:   tab[a]["pts"] += 3
        elif gb > ga: tab[b]["pts"] += 3
        else:         tab[a]["pts"] += 1; tab[b]["pts"] += 1
    for t in tab:
        tab[t]["gd"] = tab[t]["gf"] - tab[t]["ga"]

    # ---- 排名：各组前2直接晋级，第三名取最佳8 ----
    advanced, thirds = [], []
    for g, ts in GROUPS.items():
        r = _rank(ts, tab)
        advanced += r[:2]; thirds.append(r[2])
    best_thirds = _rank(thirds, tab)[:8]
    advanced += best_thirds
    qualified = set(advanced)

    # ---- 淘汰赛：32强按 Elo 种子化，标准对阵（强弱分置两端）----
    seeds = sorted(advanced, key=lambda t: matrix and _ELO[t], reverse=True)
    cur = seeds
    reached = {t: "R32" for t in advanced}
    round_names = ["R16", "QF", "SF", "Final", "Champion"]
    ri = 0
    while len(cur) > 1:
        nxt = []
        n = len(cur)
        for i in range(n // 2):
            a, b = cur[i], cur[n - 1 - i]
            _, _, a_win = _play(matrix, a, b, knockout=True)
            w = a if a_win else b
            nxt.append(w)
            reached[w] = round_names[ri]
        cur = nxt
        ri += 1
    champion = cur[0]
    return qualified, reached, champion


def main(n=20000):
    data = _load()
    matrix = data["pairs"]
    global _ELO
    _ELO = data["ratings"]
    base = compute_standings()
    rem = remaining_group_fixtures()

    cnt = {t: dict(adv=0, sf=0, final=0, champ=0) for t in TEAMS}
    for _ in range(n):
        qualified, reached, champ = simulate_once(matrix, base, rem)
        for t in qualified:
            cnt[t]["adv"] += 1
        for t, r in reached.items():
            if r in ("SF", "Final", "Champion"):
                cnt[t]["sf"] += 1
            if r in ("Final", "Champion"):
                cnt[t]["final"] += 1
        cnt[champ]["champ"] += 1

    odds = []
    for t in TEAMS:
        c = cnt[t]
        odds.append({"team": t, "cn": CN_NAME[t], "elo": _ELO[t],
                     "advance": round(c["adv"] / n, 4),
                     "semifinal": round(c["sf"] / n, 4),
                     "final": round(c["final"] / n, 4),
                     "champion": round(c["champ"] / n, 4)})
    odds.sort(key=lambda x: -x["champion"])

    with open(f"{_OUT}/title_odds.json", "w") as f:
        json.dump({"n_sims": n, "odds": odds}, f, ensure_ascii=False, indent=2)

    print(f"蒙特卡洛 {n:,} 次模拟 — 夺冠概率 Top 12：")
    print(f"{'队伍':<10}{'Elo':>7}{'晋级32强':>10}{'进四强':>9}{'进决赛':>9}{'夺冠':>8}")
    for o in odds[:12]:
        print(f"{o['cn']:<10}{o['elo']:>7.0f}{o['advance']*100:>9.0f}%{o['semifinal']*100:>8.0f}%"
              f"{o['final']*100:>8.0f}%{o['champion']*100:>7.1f}%")
    print(f"\n[saved] results/title_odds.json")


if __name__ == "__main__":
    main()
