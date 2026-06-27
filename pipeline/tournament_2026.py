# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 data_loader.load_results 取 2026 世界杯已踢比赛
[OUTPUT]: 对外提供 GROUPS(官方12组) / TEAMS / compute_standings / played_matches / CN_NAME
[POS]: limix-football预测 的赛制配置层，蒙特卡洛与网页的赛程真相源
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

分组数据：2025-12-05 华盛顿终抽签结果（队名已对齐数据集口径）。
来源交叉核对：Wikipedia 2026 FIFA World Cup draw + FIFA 官方 + ESPN/NBC。
"""
from collections import defaultdict
import pandas as pd
from data_loader import load_results, load_shootouts, load_fixtures

# ============================================================
# 淘汰赛：北京时间 6/29 起进入淘汰赛 —— 无平局，只有胜/负（加时+点球必分胜负）
# ============================================================
KO_START = pd.Timestamp("2026-06-29")
_SHOOT = None


def is_knockout(date) -> bool:
    return pd.Timestamp(date) >= KO_START


def _shoot_map():
    global _SHOOT
    if _SHOOT is None:
        s = load_shootouts()
        _SHOOT = {(pd.Timestamp(r.date), r.home_team, r.away_team): r.winner for r in s.itertuples()}
    return _SHOOT


def resolve_outcome(date, home, away, hs, as_):
    """最终胜负 home_win/away_win/draw；平分则按点球胜者定（淘汰赛绝不为平）。"""
    if hs > as_:
        return "home_win"
    if hs < as_:
        return "away_win"
    w = _shoot_map().get((pd.Timestamp(date), home, away))
    if w == home:
        return "home_win"
    if w == away:
        return "away_win"
    return "draw"


def unplayed_fixtures():
    """2026 世界杯未踢赛程（含淘汰赛），来自数据 NA 比分行：返回 (date, home, away, neutral, ko)。"""
    fx = load_fixtures()
    wc = fx[(fx.tournament == "FIFA World Cup") & (fx.date.dt.year == 2026)]
    return [(r.date, r.home_team, r.away_team, bool(r.neutral), is_knockout(r.date))
            for r in wc.itertuples()]

# ============================================================
# 官方 12 组（队名用数据集口径：Czech Republic / Turkey / DR Congo）
# ============================================================
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
TEAMS = [t for g in GROUPS.values() for t in g]
TEAM_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}

# ============================================================
# 中文队名（网页展示用）
# ============================================================
CN_NAME = {
    "Mexico": "墨西哥", "South Africa": "南非", "South Korea": "韩国", "Czech Republic": "捷克",
    "Canada": "加拿大", "Bosnia and Herzegovina": "波黑", "Qatar": "卡塔尔", "Switzerland": "瑞士",
    "Brazil": "巴西", "Morocco": "摩洛哥", "Haiti": "海地", "Scotland": "苏格兰",
    "United States": "美国", "Paraguay": "巴拉圭", "Australia": "澳大利亚", "Turkey": "土耳其",
    "Germany": "德国", "Curaçao": "库拉索", "Ivory Coast": "科特迪瓦", "Ecuador": "厄瓜多尔",
    "Netherlands": "荷兰", "Japan": "日本", "Sweden": "瑞典", "Tunisia": "突尼斯",
    "Belgium": "比利时", "Egypt": "埃及", "Iran": "伊朗", "New Zealand": "新西兰",
    "Spain": "西班牙", "Cape Verde": "佛得角", "Saudi Arabia": "沙特", "Uruguay": "乌拉圭",
    "France": "法国", "Senegal": "塞内加尔", "Iraq": "伊拉克", "Norway": "挪威",
    "Argentina": "阿根廷", "Algeria": "阿尔及利亚", "Austria": "奥地利", "Jordan": "约旦",
    "Portugal": "葡萄牙", "DR Congo": "刚果(金)", "Uzbekistan": "乌兹别克斯坦", "Colombia": "哥伦比亚",
    "England": "英格兰", "Croatia": "克罗地亚", "Ghana": "加纳", "Panama": "巴拿马",
}


def played_matches() -> pd.DataFrame:
    """返回 2026 世界杯已踢比赛（含比分）。"""
    res = load_results()
    wc = res[(res.tournament == "FIFA World Cup") & (res.date.dt.year == 2026)].copy()
    return wc.sort_values("date").reset_index(drop=True)


def remaining_group_fixtures():
    """推断每组尚未踢的小组赛对阵（全循环 6 场减去已踢）。"""
    wc = played_matches()
    done = {frozenset((r.home_team, r.away_team)) for r in wc.itertuples()}
    rem = []
    for g, ts in GROUPS.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                pair = frozenset((ts[i], ts[j]))
                if pair not in done:
                    rem.append((g, ts[i], ts[j]))
    return rem


def compute_standings():
    """从已踢比赛算各组当前积分榜：积分/胜平负/进失球/净胜球。"""
    wc = played_matches()
    st = {t: dict(pts=0, w=0, d=0, l=0, gf=0, ga=0, played=0) for t in TEAMS}
    for r in wc.itertuples():
        h, a, hs, as_ = r.home_team, r.away_team, int(r.home_score), int(r.away_score)
        if h not in st or a not in st:
            continue
        st[h]["gf"] += hs; st[h]["ga"] += as_; st[h]["played"] += 1
        st[a]["gf"] += as_; st[a]["ga"] += hs; st[a]["played"] += 1
        if hs > as_:
            st[h]["pts"] += 3; st[h]["w"] += 1; st[a]["l"] += 1
        elif hs < as_:
            st[a]["pts"] += 3; st[a]["w"] += 1; st[h]["l"] += 1
        else:
            st[h]["pts"] += 1; st[a]["pts"] += 1; st[h]["d"] += 1; st[a]["d"] += 1
    for t in st:
        st[t]["gd"] = st[t]["gf"] - st[t]["ga"]
    return st


if __name__ == "__main__":
    st = compute_standings()
    rem = remaining_group_fixtures()
    print(f"48 队 / 12 组；已踢 {len(played_matches())} 场，剩余小组赛 {len(rem)} 场\n")
    for g, ts in GROUPS.items():
        rows = sorted(ts, key=lambda t: (-st[t]["pts"], -st[t]["gd"], -st[t]["gf"]))
        print(f"组{g}: " + " | ".join(
            f"{CN_NAME[t]}({st[t]['pts']}分 {st[t]['w']}-{st[t]['d']}-{st[t]['l']} GD{st[t]['gd']:+d})"
            for t in rows))
