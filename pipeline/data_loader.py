# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 requests/pandas；从 martj42/international_results GitHub raw 拉取
[OUTPUT]: 对外提供 load_results / load_shootouts / load_goalscorers / load_all
[POS]: limix-football预测 的数据接入层，所有特征工程的唯一数据源头
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import os
import pandas as pd
import requests

# ============================================================
# 数据源 —— GitHub raw，无需 Kaggle 登录，与 Kaggle 数据集同源
# ============================================================
_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
_FILES = {
    "results": "results.csv",
    "shootouts": "shootouts.csv",
    "goalscorers": "goalscorers.csv",
}
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _cached(name: str) -> pd.DataFrame:
    """拉取并缓存单张表；本地已有则直接读，避免重复下载。"""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, _FILES[name])
    if not os.path.exists(path):
        url = f"{_BASE}/{_FILES[name]}"
        print(f"[data] downloading {name} <- {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_results() -> pd.DataFrame:
    """主表：date/home_team/away_team/home_score/away_score/tournament/city/country/neutral"""
    df = _cached("results").dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def load_fixtures() -> pd.DataFrame:
    """未踢赛程：home_score/away_score 为空的比赛（含未来淘汰赛对阵）。"""
    df = _cached("results")
    df = df[df["home_score"].isna() | df["away_score"].isna()].copy()
    return df.sort_values("date").reset_index(drop=True)


def load_shootouts() -> pd.DataFrame:
    """点球大战：date/home_team/away_team/winner/first_shooter（淘汰赛平局判定用）"""
    return _cached("shootouts")


def load_goalscorers() -> pd.DataFrame:
    """进球明细：date/home_team/away_team/team/scorer/own_goal/penalty"""
    return _cached("goalscorers")


def load_all():
    return load_results(), load_shootouts(), load_goalscorers()


if __name__ == "__main__":
    r = load_results()
    print(f"results : {len(r):,} 场，{r['date'].min().date()} ~ {r['date'].max().date()}")
    print(f"          球队数 {pd.concat([r.home_team, r.away_team]).nunique()}，赛事数 {r.tournament.nunique()}")
    s = load_shootouts()
    g = load_goalscorers()
    print(f"shootouts: {len(s):,} 场点球大战")
    print(f"goalscorers: {len(g):,} 条进球记录")
    print("\n[results 样例]")
    print(r.tail(3).to_string())
