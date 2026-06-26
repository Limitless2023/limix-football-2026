# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 data_loader/features/limix_client/tournament_2026
[OUTPUT]: 调 LimiX local 解释接口对未开踢比赛算特征归因 → results/explanations.json
[POS]: limix-football预测 的可解释层，回答"LimiX 为什么这样预测"（真实 API 重要性 + 值推方向）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

实测要点：local 解释要求 target_column 为【整数编码】标签，否则服务端 4501；
LimiX 返回 feature_scores（重要性，direction=unknown），方向由特征实际值透明推导。
"""
import os
import json
import time
import requests
import numpy as np
import pandas as pd

from data_loader import load_results
from features import FeatureEngine, FEATURE_COLS
from limix_client import LimiXClient
from tournament_2026 import remaining_group_fixtures, CN_NAME, TEAM_GROUP

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_LABEL_MAP = {"away_win": 0, "draw": 1, "home_win": 2}

# ------------------------------------------------------------
# 23 原始特征 → 8 个人类可读概念（聚合重要性 + 给方向函数）
# dir(row) > 0 → 利好主队；< 0 → 利好客队；0 → 中性
# ------------------------------------------------------------
_CONCEPTS = [
    ("实力评分 Elo", ["elo_home", "elo_away", "elo_diff", "home_adv"],
     lambda r: r["elo_diff"]),
    ("近期状态", ["ppg5_home", "ppg5_away", "ppg10_home", "ppg10_away"],
     lambda r: (r["ppg5_home"] + r["ppg10_home"]) - (r["ppg5_away"] + r["ppg10_away"])),
    ("进攻火力", ["gf5_home", "gf5_away"], lambda r: r["gf5_home"] - r["gf5_away"]),
    ("防守稳健", ["ga5_home", "ga5_away"], lambda r: r["ga5_away"] - r["ga5_home"]),
    ("近期净胜球", ["gd10_home", "gd10_away"], lambda r: r["gd10_home"] - r["gd10_away"]),
    ("连胜势头", ["streak_home", "streak_away"], lambda r: r["streak_home"] - r["streak_away"]),
    ("体能休整", ["rest_home", "rest_away"], lambda r: r["rest_home"] - r["rest_away"]),
    ("历史交锋", ["h2h_winrate_home", "h2h_drawrate", "h2h_gd_home", "h2h_games"],
     lambda r: (r["h2h_winrate_home"] - 0.5) if r["h2h_games"] > 0 else 0.0),
]


def _explain(client, train, ex_feat):
    tr = client.upload(train, "exp_train")
    ev = client.upload(ex_feat, "exp_explain")
    body = {"explanation_mode": "local", "train_data_version_id": tr,
            "explain_data_version_id": ev, "target_column": "label",
            "local_task_type": "classification", "model_type": client.model_type,
            "fuse_shap": False}
    t0 = time.time()
    r = requests.post(client.base + "/v1/extensions/explanations/tasks",
                      headers=client.h, json=body, timeout=600)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"explain code={j.get('code')} msg={j.get('message')}")
    d = j["data"]
    result = d.get("result") or {}
    # 接口已改异步：若结果未就绪(无 feature_names)，按 result_id 轮询直至完成
    if "feature_names" not in result:
        rid = d["result_id"]
        for _ in range(60):                          # 最多约 5 分钟
            time.sleep(5)
            q = requests.get(f"{client.base}/v1/extensions/results/{rid}",
                             headers=client.h, timeout=60).json()
            qd = q.get("data", {})
            st = qd.get("task_status")
            if st == "succeeded":
                result = qd.get("result", {})
                break
            if st in ("failed", "timeout", "cancelled"):
                raise RuntimeError(f"explain {st}: {qd.get('error_message')}")
    if "feature_names" not in result:
        raise RuntimeError("explain 轮询超时，结果仍未就绪")
    print(f"[explain] {len(ex_feat)} 场解释完成，耗时 {time.time()-t0:.0f}s")
    return result


def main():
    os.makedirs(_OUT, exist_ok=True)
    res = load_results()
    eng = FeatureEngine()
    hist = eng.fit_transform(res)
    # 训练上下文用整数标签（接口要求），控制规模换速度
    train = hist[hist.date >= "2018-01-01"][FEATURE_COLS + ["label"]].tail(1500).copy()
    train["label"] = train["label"].map(_LABEL_MAP)

    # 待解释 = 32 场未开踢小组赛
    rem = remaining_group_fixtures()
    fixtures = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-26"), "home_team": a, "away_team": b,
        "neutral": True, "tournament": "FIFA World Cup",
    } for g, a, b in rem])
    ex = eng.transform_fixtures(fixtures)
    ex_feat = ex[FEATURE_COLS]

    client = LimiXClient(model_type="LIMIX_2M", timeout=600)
    result = _explain(client, train, ex_feat)

    names = result["feature_names"]
    scores = np.array(result["feature_scores"])            # [n_samples × 23]
    probs = result.get("probabilities")                     # [n × 3] away/draw/home
    idx = {n: i for i, n in enumerate(names)}

    # 逐场聚合到概念
    out = {}
    for k, (g, a, b) in enumerate(rem):
        row = ex.iloc[k]
        concept_imp = []
        for cname, cols, dfn in _CONCEPTS:
            imp = float(sum(scores[k, idx[c]] for c in cols if c in idx))
            d = dfn(row)
            side = "home" if d > 1e-6 else ("away" if d < -1e-6 else "even")
            concept_imp.append({"name": cname, "importance": round(imp, 4), "side": side})
        concept_imp.sort(key=lambda x: -x["importance"])
        tot = sum(c["importance"] for c in concept_imp) or 1.0
        for c in concept_imp:
            c["pct"] = round(c["importance"] / tot, 4)
        entry = {"home": a, "away": b, "group": g,
                 "hcn": CN_NAME[a], "acn": CN_NAME[b], "concepts": concept_imp}
        if probs:
            entry["p_away"], entry["p_draw"], entry["p_home"] = \
                round(probs[k][0], 4), round(probs[k][1], 4), round(probs[k][2], 4)
        out[f"{a}|{b}"] = entry

    # 逐场方差自检：确认不是一份全局重要性套在所有比赛上
    var = float(np.mean(np.var(scores, axis=0)))
    with open(f"{_OUT}/explanations.json", "w") as f:
        json.dump({"model": client.model_type, "n": len(rem),
                   "per_sample_var": round(var, 6), "explanations": out}, f, ensure_ascii=False)
    print(f"[saved] results/explanations.json：{len(rem)} 场，逐场重要性方差={var:.5f}")
    # 抽样展示
    sample = out[list(out)[0]]
    print(f"样例 {sample['hcn']} vs {sample['acn']}：top3 概念 "
          + " / ".join(f"{c['name']}({c['pct']*100:.0f}%,{c['side']})" for c in sample["concepts"][:3]))


if __name__ == "__main__":
    main()
