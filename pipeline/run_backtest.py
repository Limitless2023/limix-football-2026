# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 data_loader/features/limix_client + sklearn/xgboost
[OUTPUT]: 对某届世界杯的回测：LimiX 胜平负+比分预测、准确率/logloss、PK 传统ML；产出 results/*.json|csv
[POS]: limix-football预测 的价值证明入口，把"无泄漏特征 + LimiX 基础模型"的实战成绩诚实摊开
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

跑法：
    python3 run_backtest.py 2022      # 单届世界杯回测（讲故事：名场面对照）
    python3 run_backtest.py broad     # 大样本回测（讲实力：数千场，噪声平均）
"""
import sys
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loader import load_results
from features import FeatureEngine, FEATURE_COLS
from limix_client import LimiXClient

CLASSES = ["away_win", "draw", "home_win"]      # 概率列 0/1/2 的固定顺序
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_TRAIN_START = "2006-01-01"                       # 训练上下文起点（平衡丰富度与时延）


# ============================================================
# 数据准备：切出某届世界杯作为测试集，之前的比赛作为训练上下文
# ============================================================
def prepare(year: int):
    res = load_results()
    feat = FeatureEngine().fit_transform(res)

    mask_wc = (feat.tournament == "FIFA World Cup") & (feat.date.dt.year == year)
    test = feat[mask_wc].copy()
    if test.empty:
        raise SystemExit(f"未找到 {year} 年世界杯比赛")
    cutoff = test.date.min()
    train = feat[(feat.date < cutoff) & (feat.date >= _TRAIN_START)].copy()
    print(f"[{year}WC] 训练上下文 {len(train):,} 场（{_TRAIN_START}~{cutoff.date()}），测试 {len(test)} 场")
    return train, test


def prepare_broad(test_start="2024-01-01"):
    """大样本回测：test_start 之后所有比赛为测试集，之前为训练上下文。"""
    res = load_results()
    feat = FeatureEngine().fit_transform(res)
    test = feat[feat.date >= test_start].copy()
    train = feat[(feat.date < test_start) & (feat.date >= _TRAIN_START)].copy()
    print(f"[broad] 训练上下文 {len(train):,} 场，测试 {len(test):,} 场（{test_start} 起）")
    return train, test


# ============================================================
# LimiX 预测：胜平负分类 + 主/客比分回归
# ============================================================
def limix_predict(client, train, test):
    cls = client.predict_classification(
        train[FEATURE_COLS + ["label"]], test[FEATURE_COLS], target="label", tag="wc_cls")
    probs = cls[["pred_probs_0", "pred_probs_1", "pred_probs_2"]].values
    probs = probs / probs.sum(axis=1, keepdims=True)      # 行归一，确保概率和为1
    pred_label = cls["pred_label"].values

    reg_h = client.predict_regression(
        train[FEATURE_COLS + ["home_score"]], test[FEATURE_COLS], target="home_score", tag="wc_rh")
    reg_a = client.predict_regression(
        train[FEATURE_COLS + ["away_score"]], test[FEATURE_COLS], target="away_score", tag="wc_ra")
    return probs, pred_label, reg_h["pred_label"].values, reg_a["pred_label"].values


# ============================================================
# 传统 ML 基准：同一份特征，PK XGBoost / RandomForest / LogReg
# ============================================================
def baselines(train, test):
    Xtr, ytr = train[FEATURE_COLS].values, train["label"].values
    Xte = test[FEATURE_COLS].values
    models = {
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, eval_metric="mlogloss", n_jobs=4,
                                 random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=12, n_jobs=4,
                                               random_state=42),
        "LogisticReg": make_pipeline(StandardScaler(),
                                     LogisticRegression(max_iter=2000, C=0.5)),
    }
    out = {}
    for name, m in models.items():
        # XGBoost 需要数值标签
        if name == "XGBoost":
            ymap = {c: i for i, c in enumerate(CLASSES)}
            m.fit(Xtr, np.array([ymap[y] for y in ytr]))
            proba = m.predict_proba(Xte)
            order = list(m.classes_)                      # 数值类顺序
            proba = proba[:, [order.index(ymap[c]) for c in CLASSES]]
        else:
            m.fit(Xtr, ytr)
            order = list(m.classes_)
            proba = m.predict_proba(Xte)[:, [order.index(c) for c in CLASSES]]
        pred = [CLASSES[i] for i in proba.argmax(1)]
        out[name] = (proba, np.array(pred))
    return out


# ============================================================
# 指标
# ============================================================
def score(y_true, probs, pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "logloss": round(float(log_loss(y_true, probs, labels=CLASSES)), 4),
    }


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "2022"
    os.makedirs(_OUT, exist_ok=True)
    if arg == "broad":
        year = "broad"
        train, test = prepare_broad()
    else:
        year = int(arg)
        train, test = prepare(year)
    y_true = test["label"].values

    client = LimiXClient(model_type="LIMIX_16M")
    probs, pred_label, ph, pa = limix_predict(client, train, test)

    # ---- 比分回归 → 派生胜平负 + 比分误差 ----
    ph_r, pa_r = np.rint(ph).astype(int).clip(0), np.rint(pa).astype(int).clip(0)
    score_outcome = np.where(ph_r > pa_r, "home_win", np.where(ph_r < pa_r, "away_win", "draw"))
    mae = float(np.mean(np.abs(ph - test.home_score.values)) +
                np.mean(np.abs(pa - test.away_score.values))) / 2
    exact = float(np.mean((ph_r == test.home_score.values) & (pa_r == test.away_score.values)))

    # ---- 汇总各模型 ----
    leaderboard = {"LimiX": score(y_true, probs, pred_label)}
    for name, (pb, pd_) in baselines(train, test).items():
        leaderboard[name] = score(y_true, pb, pd_)

    title = "大样本回测" if year == "broad" else f"{year} 世界杯回测"
    print(f"\n{'='*52}\n{title}成绩单（{len(test)} 场，真值可对照）\n{'='*52}")
    print(f"{'模型':<14}{'胜平负准确率':>14}{'LogLoss':>12}")
    for name, s in sorted(leaderboard.items(), key=lambda kv: -kv[1]["accuracy"]):
        star = " 🏆" if name == max(leaderboard, key=lambda k: leaderboard[k]["accuracy"]) else ""
        print(f"{name:<14}{s['accuracy']*100:>12.1f}%{s['logloss']:>12.3f}{star}")
    print(f"\nLimiX 比分回归：场均比分 MAE {mae:.2f} 球，精确比分命中率 {exact*100:.1f}%")
    print(f"比分派生胜平负准确率：{accuracy_score(y_true, score_outcome)*100:.1f}%")

    # ---- 落盘明细（供 Web 与报告复用）----
    detail = test[["date", "home_team", "away_team", "neutral",
                   "home_score", "away_score", "label"]].copy()
    detail["p_away"], detail["p_draw"], detail["p_home"] = probs[:, 0], probs[:, 1], probs[:, 2]
    detail["pred_label"] = pred_label
    detail["pred_home_score"], detail["pred_away_score"] = ph_r, pa_r
    detail["pred_home_raw"], detail["pred_away_raw"] = ph.round(2), pa.round(2)
    detail["date"] = detail["date"].dt.strftime("%Y-%m-%d")
    detail.to_csv(f"{_OUT}/backtest_{year}_detail.csv", index=False)

    summary = {"year": year, "n_matches": int(len(test)),
               "train_context": int(len(train)),
               "leaderboard": leaderboard,
               "scoreline": {"mae": round(mae, 3), "exact_rate": round(exact, 4),
                             "outcome_acc": round(float(accuracy_score(y_true, score_outcome)), 4)}}
    with open(f"{_OUT}/backtest_{year}_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] results/backtest_{year}_detail.csv + summary.json")


if __name__ == "__main__":
    main()
