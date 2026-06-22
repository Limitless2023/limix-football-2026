# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 pandas/numpy 与 data_loader.load_results 产出的比赛主表
[OUTPUT]: 对外提供 FeatureEngine（fit_transform / transform_fixtures）、FEATURE_COLS
[POS]: limix-football预测 的特征核心，对标 Prior Labs 六大类特征，严守"无未来泄漏"
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

设计铁律 —— 时间单向单遍扫描：
    对每场比赛，先用"此前已发生"的状态算出特征行，再用本场结果更新状态。
    任何特征在开球时刻只见过去，不见未来。这是整套预测可信度的基石。
"""
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# ============================================================
# 特征列清单（喂给 LimiX 的输入，label/比分单独处理）
# ============================================================
FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff", "home_adv",
    "ppg5_home", "ppg5_away", "ppg10_home", "ppg10_away",
    "gf5_home", "ga5_home", "gf5_away", "ga5_away",
    "gd10_home", "gd10_away",
    "streak_home", "streak_away",
    "rest_home", "rest_away",
    "h2h_winrate_home", "h2h_drawrate", "h2h_gd_home", "h2h_games",
    "importance",
]

# ------------------------------------------------------------
# Elo 参数（World Football Elo 风格）
# ------------------------------------------------------------
_ELO_INIT = 1500.0
_HOME_ADV = 65.0          # 非中立场主队加成（对齐 Prior Labs）
_REST_CAP = 90            # 距上场天数上限


def _importance(tournament: str) -> float:
    """赛事重要性 → Elo 的 K 因子基数。世界杯最高，友谊赛最低。"""
    t = (tournament or "").lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "world cup qual" in t or "uefa euro" in t or "copa am" in t or "african cup" in t \
            or "asian cup" in t or "confederations" in t or "nations league" in t:
        return 50.0
    if "qual" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def _goal_mult(margin: int) -> float:
    """净胜球放大因子：大胜对 Elo 影响更大（World Football Elo 公式）。"""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def _result_points(hs: int, as_: int):
    """返回 (主队积分, 客队积分, 主队Elo实际得分)。"""
    if hs > as_:
        return 3, 0, 1.0
    if hs < as_:
        return 0, 3, 0.0
    return 1, 1, 0.5


class FeatureEngine:
    """单遍扫描的有状态特征引擎，可复用于历史回测与未来赛程预测。"""

    def __init__(self):
        self.elo = defaultdict(lambda: _ELO_INIT)
        self.recent = defaultdict(lambda: deque(maxlen=10))   # 每队近10场 {pts,gf,ga}
        self.last_date = {}                                   # 上场日期
        self.streak = defaultdict(int)                        # 连胜场次（连败为负）
        self.h2h = defaultdict(lambda: {"g": 0, "w": 0, "d": 0, "gd": 0})  # (team,opp)→记录

    # -------- 只读：根据当前状态算一场比赛的特征行（不改状态） --------
    def _row(self, home, away, neutral, date, tournament) -> dict:
        rec_h, rec_a = self.recent[home], self.recent[away]

        def _ppg(rec, n):
            xs = list(rec)[-n:]
            return float(np.mean([r["pts"] for r in xs])) if xs else 1.0

        def _gf(rec, n):
            xs = list(rec)[-n:]
            return float(np.mean([r["gf"] for r in xs])) if xs else 1.0

        def _ga(rec, n):
            xs = list(rec)[-n:]
            return float(np.mean([r["ga"] for r in xs])) if xs else 1.0

        def _gd(rec, n):
            xs = list(rec)[-n:]
            return float(np.mean([r["gf"] - r["ga"] for r in xs])) if xs else 0.0

        def _rest(team):
            ld = self.last_date.get(team)
            if ld is None:
                return _REST_CAP
            return min((date - ld).days, _REST_CAP)

        ha = 0.0 if neutral else _HOME_ADV
        h2h = self.h2h[(home, away)]
        g = h2h["g"]

        return {
            "elo_home": self.elo[home],
            "elo_away": self.elo[away],
            "elo_diff": self.elo[home] + ha - self.elo[away],
            "home_adv": 0 if neutral else 1,
            "ppg5_home": _ppg(rec_h, 5), "ppg5_away": _ppg(rec_a, 5),
            "ppg10_home": _ppg(rec_h, 10), "ppg10_away": _ppg(rec_a, 10),
            "gf5_home": _gf(rec_h, 5), "ga5_home": _ga(rec_h, 5),
            "gf5_away": _gf(rec_a, 5), "ga5_away": _ga(rec_a, 5),
            "gd10_home": _gd(rec_h, 10), "gd10_away": _gd(rec_a, 10),
            "streak_home": self.streak[home], "streak_away": self.streak[away],
            "rest_home": _rest(home), "rest_away": _rest(away),
            "h2h_winrate_home": (h2h["w"] / g) if g else 0.5,
            "h2h_drawrate": (h2h["d"] / g) if g else 0.25,
            "h2h_gd_home": (h2h["gd"] / g) if g else 0.0,
            "h2h_games": g,
            "importance": _importance(tournament),
        }

    # -------- 写：用一场真实结果更新状态 --------
    def _update(self, home, away, hs, as_, neutral, tournament):
        hp, ap, sh = _result_points(hs, as_)
        ha = 0.0 if neutral else _HOME_ADV

        # Elo 更新
        dr = self.elo[home] + ha - self.elo[away]
        e_home = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        k = _importance(tournament) * _goal_mult(abs(hs - as_))
        delta = k * (sh - e_home)
        self.elo[home] += delta
        self.elo[away] -= delta

        # 近况
        self.recent[home].append({"pts": hp, "gf": hs, "ga": as_})
        self.recent[away].append({"pts": ap, "gf": as_, "ga": hs})

        # 连胜（连败记为负）
        if hs > as_:
            self.streak[home] = max(1, self.streak[home] + 1)
            self.streak[away] = min(-1, self.streak[away] - 1)
        elif hs < as_:
            self.streak[away] = max(1, self.streak[away] + 1)
            self.streak[home] = min(-1, self.streak[home] - 1)
        else:
            self.streak[home] = 0
            self.streak[away] = 0

        # 交锋史（双向各记一条，视角对应）
        gd = hs - as_
        rh = self.h2h[(home, away)]
        ra = self.h2h[(away, home)]
        rh["g"] += 1; ra["g"] += 1
        rh["d"] += (hs == as_); ra["d"] += (hs == as_)
        rh["w"] += (hs > as_); ra["w"] += (as_ > hs)
        rh["gd"] += gd; ra["gd"] += -gd

    # -------- 历史全表：逐场产出特征行 + label + 比分 --------
    def fit_transform(self, results: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for r in results.itertuples(index=False):
            date = r.date
            feats = self._row(r.home_team, r.away_team, bool(r.neutral), date, r.tournament)
            # 标签 + 回归目标 + 元信息
            if r.home_score > r.away_score:
                label = "home_win"
            elif r.home_score < r.away_score:
                label = "away_win"
            else:
                label = "draw"
            feats.update({
                "date": date, "home_team": r.home_team, "away_team": r.away_team,
                "tournament": r.tournament, "neutral": bool(r.neutral),
                "label": label, "home_score": int(r.home_score), "away_score": int(r.away_score),
            })
            rows.append(feats)
            # 先记特征、后更新状态 —— 无泄漏
            self._update(r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                         bool(r.neutral), r.tournament)
            self.last_date[r.home_team] = date
            self.last_date[r.away_team] = date
        return pd.DataFrame(rows)

    # -------- 未来赛程：只读产出特征行（用于 2026 在打赛事预测） --------
    def transform_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for r in fixtures.itertuples(index=False):
            feats = self._row(r.home_team, r.away_team, bool(r.neutral), r.date, r.tournament)
            feats.update({
                "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
                "tournament": r.tournament, "neutral": bool(r.neutral),
            })
            rows.append(feats)
        return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import load_results
    res = load_results()
    eng = FeatureEngine()
    feat = eng.fit_transform(res)
    print(f"特征表：{feat.shape[0]:,} 行 × {len(FEATURE_COLS)} 特征")
    print(f"标签分布：\n{feat['label'].value_counts(normalize=True).round(3)}")
    # 抽查 2026 世界杯的一场，看特征是否合理
    wc = feat[(feat.tournament == "FIFA World Cup") & (feat.date >= "2026-06-01")]
    print(f"\n2026世界杯已录入 {len(wc)} 场，样例：")
    cols = ["date", "home_team", "away_team", "elo_home", "elo_away", "elo_diff", "label", "home_score", "away_score"]
    print(wc[cols].head(6).to_string(index=False))
