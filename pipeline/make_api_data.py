# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 data_loader/features/tournament_2026
[OUTPUT]: 为 Vercel Serverless 函数生成 api/_data/{train_cls,train_rh,train_ra}.csv + pair_features.json
[POS]: limix-football预测 的在线推理数据导出层，让 Node 函数无需重放历史即可现场调 LimiX
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import os
import json
import itertools
import pandas as pd
from data_loader import load_results
from features import FeatureEngine, FEATURE_COLS
from tournament_2026 import TEAMS

_DEPLOY = "/Users/limitless/Desktop/Projects/stable-ai/05-演示产品/limix-football-2026"
_OUT = os.path.join(_DEPLOY, "api", "_data")
_CTX = 6000
_TODAY = pd.Timestamp("2026-06-22")


def main():
    os.makedirs(_OUT, exist_ok=True)
    eng = FeatureEngine()
    hist = eng.fit_transform(load_results())     # 引擎停在当前态
    ctx = hist[hist.date < _TODAY].tail(_CTX)

    # 三份训练上下文（特征相同，目标列不同）
    ctx[FEATURE_COLS + ["label"]].to_csv(f"{_OUT}/train_cls.csv", index=False)
    ctx[FEATURE_COLS + ["home_score"]].to_csv(f"{_OUT}/train_rh.csv", index=False)
    ctx[FEATURE_COLS + ["away_score"]].to_csv(f"{_OUT}/train_ra.csv", index=False)

    # 48强全 2256 directed pairs 的 23 维特征行（中立场，今日）
    pairs = [(a, b) for a, b in itertools.product(TEAMS, TEAMS) if a != b]
    fixtures = pd.DataFrame([{
        "date": _TODAY, "home_team": a, "away_team": b,
        "neutral": True, "tournament": "FIFA World Cup",
    } for a, b in pairs])
    feat = eng.transform_fixtures(fixtures)[FEATURE_COLS]
    mtx = {f"{a}|{b}": [round(float(v), 4) for v in feat.iloc[k].values]
           for k, (a, b) in enumerate(pairs)}
    with open(f"{_OUT}/pair_features.json", "w") as f:
        json.dump({"cols": FEATURE_COLS, "pairs": mtx}, f, ensure_ascii=False)

    sizes = {n: f"{os.path.getsize(f'{_OUT}/{n}'):,}B" for n in
             ["train_cls.csv", "train_rh.csv", "train_ra.csv", "pair_features.json"]}
    print(f"[saved] -> {_OUT}")
    print(f"  上下文 {len(ctx)} 行 / 配对 {len(pairs)} 组 / 文件大小 {sizes}")


if __name__ == "__main__":
    main()
