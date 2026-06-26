# -*- coding: utf-8 -*-
"""
[INPUT]: data_loader/features/limix_client/tournament_2026
[OUTPUT]: 2026世界杯逐日滚动预测：每个比赛日用"之前历史"训练，出 默认版 + best_f1阈值版 → results/daily_walkforward.json
[POS]: limix-football预测 的逐日 walk-forward 层，无未来泄漏；LLM 基线另由 build_llm_baseline 合并
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

机制：对每个比赛日 D，LimiX 上下文 = D 之前最近 5000 场(特征本身已含全史 Elo，无泄漏)；
      预测当日全部比赛。默认=三类 argmax；阈值版=各类概率 / best_f1阈值 取最大(让低阈值的平局有机会被选中)。
"""
import os
import json
import time
import numpy as np
import pandas as pd
from data_loader import load_results
from features import FeatureEngine, FEATURE_COLS
from tournament_2026 import CN_NAME

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CLASSES = ["away_win", "draw", "home_win"]          # 概率列 0/1/2 顺序
CN = {"home_win": "主胜", "draw": "平", "away_win": "客胜"}
# best_f1 阈值(来自 OOF 实测，LIMIX_16M)
THR = {"home_win": 0.36, "away_win": 0.25, "draw": 0.18}
_CTX = 5000


def main():
    from limix_client import LimiXClient
    os.makedirs(_OUT, exist_ok=True)
    feat = FeatureEngine().fit_transform(load_results())
    wc = feat[(feat.tournament == "FIFA World Cup") & (feat.date.dt.year == 2026)].sort_values("date")
    days = sorted(wc["date"].unique())
    c = LimiXClient(model_type="LIMIX_16M", timeout=400)
    print(f"逐日滚动：{len(days)} 个比赛日 / {len(wc)} 场")

    rows = []
    for D in days:
        day = wc[wc.date == D]
        train = feat[feat.date < D].tail(_CTX)
        res = c.predict_classification(train[FEATURE_COLS + ["label"]], day[FEATURE_COLS],
                                       target="label", tag=f"wf_{pd.Timestamp(D).strftime('%m%d')}")
        P = res[["pred_probs_0", "pred_probs_1", "pred_probs_2"]].values
        P = P / P.sum(1, keepdims=True)
        for k, (_, m) in enumerate(day.iterrows()):
            prob = {"away_win": float(P[k, 0]), "draw": float(P[k, 1]), "home_win": float(P[k, 2])}
            default = max(prob, key=prob.get)
            thr_pick = max(CLASSES, key=lambda cl: prob[cl] / THR[cl])
            rows.append({
                "date": pd.Timestamp(D).strftime("%m-%d"),
                "home": m["home_team"], "away": m["away_team"],
                "match": f"{CN_NAME.get(m['home_team'], m['home_team'])} vs {CN_NAME.get(m['away_team'], m['away_team'])}",
                "score": f"{int(m['home_score'])}:{int(m['away_score'])}",
                "actual": m["label"],
                "p_home": round(prob["home_win"], 3), "p_draw": round(prob["draw"], 3), "p_away": round(prob["away_win"], 3),
                "limix_default": default, "limix_thr": thr_pick,
            })
        print(f"  {pd.Timestamp(D).strftime('%m-%d')}: {len(day)}场 (训练{len(train)})")

    # 汇总命中
    def hits(key):
        return sum(1 for r in rows if r[key] == r["actual"])
    summ = {"n": len(rows), "thresholds_best_f1": THR,
            "hits_default": hits("limix_default"), "hits_thr": hits("limix_thr"),
            "draw_total": sum(1 for r in rows if r["actual"] == "draw"),
            "draw_recall_default": sum(1 for r in rows if r["actual"] == "draw" and r["limix_default"] == "draw"),
            "draw_recall_thr": sum(1 for r in rows if r["actual"] == "draw" and r["limix_thr"] == "draw")}
    with open(f"{_OUT}/daily_walkforward.json", "w") as f:
        json.dump({"summary": summ, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n默认命中 {summ['hits_default']}/{summ['n']}  阈值版命中 {summ['hits_thr']}/{summ['n']}")
    print(f"平局召回：默认 {summ['draw_recall_default']}/{summ['draw_total']}  阈值版 {summ['draw_recall_thr']}/{summ['draw_total']}")
    print(f"[saved] results/daily_walkforward.json")


if __name__ == "__main__":
    main()
