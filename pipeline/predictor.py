# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 data_loader/features/limix_client/tournament_2026
[OUTPUT]: 生成 48强全配对预测矩阵 results/pred_matrix.json（胜平负概率+预测比分+当前Elo）
[POS]: limix-football预测 的预测核心，把"当前态特征"经 LimiX 固化为离线矩阵，供网页/蒙特卡洛即时取用
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

设计：网页交互若每次现调 LimiX 会卡延迟。改为一次性算全 2256 directed pairs，
固化成矩阵 → 网页与模拟都零延迟读取。WC 多为中立场，统一按 neutral=True 处理。
"""
import os
import json
import itertools
import numpy as np
import pandas as pd

from data_loader import load_results
from features import FeatureEngine, FEATURE_COLS
from limix_client import LimiXClient
from tournament_2026 import TEAMS

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_TRAIN_START = "2006-01-01"
_TODAY = pd.Timestamp("2026-06-22")


def build_matrix(model_type="LIMIX_16M"):
    os.makedirs(_OUT, exist_ok=True)
    res = load_results()
    eng = FeatureEngine()
    hist = eng.fit_transform(res)                      # 喂入全部历史，引擎停在当前态
    train = hist[hist.date >= _TRAIN_START]
    ratings = {t: round(float(eng.elo[t]), 1) for t in TEAMS}

    # ---- 构造全 directed pairs 作为待预测赛程 ----
    pairs = [(a, b) for a, b in itertools.product(TEAMS, TEAMS) if a != b]
    fixtures = pd.DataFrame([{
        "date": _TODAY, "home_team": a, "away_team": b,
        "neutral": True, "tournament": "FIFA World Cup",
    } for a, b in pairs])
    feat = eng.transform_fixtures(fixtures)

    client = LimiXClient(model_type=model_type)
    cls = client.predict_classification(
        train[FEATURE_COLS + ["label"]], feat[FEATURE_COLS], target="label", tag="mx_cls")
    probs = cls[["pred_probs_0", "pred_probs_1", "pred_probs_2"]].values
    probs = probs / probs.sum(axis=1, keepdims=True)         # away/draw/home 归一
    rh = client.predict_regression(
        train[FEATURE_COLS + ["home_score"]], feat[FEATURE_COLS], target="home_score", tag="mx_rh")
    ra = client.predict_regression(
        train[FEATURE_COLS + ["away_score"]], feat[FEATURE_COLS], target="away_score", tag="mx_ra")

    matrix = {}
    for k, (a, b) in enumerate(pairs):
        matrix[f"{a}|{b}"] = {
            "p_home": round(float(probs[k, 2]), 4),
            "p_draw": round(float(probs[k, 1]), 4),
            "p_away": round(float(probs[k, 0]), 4),
            "gh": round(float(rh["pred_label"].values[k]), 2),
            "ga": round(float(ra["pred_label"].values[k]), 2),
        }

    out = {"generated": str(_TODAY.date()), "model": model_type,
           "teams": TEAMS, "ratings": ratings, "pairs": matrix}
    with open(f"{_OUT}/pred_matrix.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[saved] results/pred_matrix.json：{len(pairs)} directed pairs，Elo 最高 "
          f"{sorted(ratings.items(), key=lambda kv:-kv[1])[:3]}")
    return out


if __name__ == "__main__":
    build_matrix()
