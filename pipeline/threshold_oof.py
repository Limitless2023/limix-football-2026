# -*- coding: utf-8 -*-
"""
[INPUT]: data_loader/features/limix_client + sklearn(KFold/metrics)
[OUTPUT]: 自跑 OOF 二分类阈值检索：3任务(主胜/客胜/平为正)× LimiX，5折OOF搜阈值→60场测试集验证
          → results/oof_threshold_{model}.json
[POS]: limix-football预测 的阈值检索实证层，回应"自己跑OOF阈值"需求，对照 best_f1/precision/recall
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

跑法: python3 threshold_oof.py LIMIX_16M   /   python3 threshold_oof.py LIMIX_64M
OOF 机制: 训练池做 5 折分层；每折用其余 4 折作 LimiX 上下文预测本折 → 得全池 OOF 正类概率；
          在 OOF 概率上按 f1/precision/recall 各搜最优阈值；再用全池作上下文预测 60 场测试集验证。
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score

from data_loader import load_results
from features import FeatureEngine, FEATURE_COLS
from limix_client import LimiXClient

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_CUTOFF = "2026-06-11"          # 2026世界杯开赛日；测试=之后已踢，训练池=之前
_POOL = 1500                    # OOF 训练池规模（控制调用量）
_KFOLD = 5
_GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)
# 三个底层二分类任务（左队视角命名；右队视角=胜/负标签对调）
TASKS = [("胜_主胜为正", "home_win"), ("负_客胜为正", "away_win"), ("平_平局为正", "draw")]


def pos_prob(res):
    """二分类结果取正类('pos')概率列。"""
    cols = [c for c in res.columns if c.startswith("pred_probs")]
    # 类名字典序 neg<pos → 末列为 pos；用 pred_label 校验
    p = res[cols[-1]].values
    return p


def search(y, p):
    """在 OOF 概率上按三指标各搜最优阈值。"""
    out = {}
    for name, fn in [("f1", f1_score), ("precision", precision_score), ("recall", recall_score)]:
        best_t, best_v = 0.5, -1
        for t in _GRID:
            v = fn(y, (p >= t).astype(int), zero_division=0)
            if v > best_v:
                best_v, best_t = v, float(t)
        out[name] = {"threshold": best_t, "oof_score": round(float(best_v), 4)}
    return out


def test_metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "n_pred_pos": int(pred.sum())}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "LIMIX_16M"
    os.makedirs(_OUT, exist_ok=True)
    feat = FeatureEngine().fit_transform(load_results())
    test = feat[(feat.tournament == "FIFA World Cup") & (feat.date.dt.year == 2026)].reset_index(drop=True)
    pool = feat[feat.date < _CUTOFF].tail(_POOL).reset_index(drop=True)
    print(f"[{model}] 训练池 {len(pool)} / 测试 {len(test)} 场 (2026已踢)")
    c = LimiXClient(model_type=model, timeout=400)

    report = {"model": model, "pool": len(pool), "n_test": int(len(test)),
              "cutoff": _CUTOFF, "kfold": _KFOLD, "tasks": {}}

    for tname, pos in TASKS:
        t0 = time.time()
        ytr = (pool["label"] == pos).astype(int).values
        yte = (test["label"] == pos).astype(int).values

        # ---- OOF：5折，每折用其余折作上下文预测本折 ----
        oof = np.zeros(len(pool))
        skf = StratifiedKFold(n_splits=_KFOLD, shuffle=True, random_state=42)
        for k, (tr_idx, va_idx) in enumerate(skf.split(pool[FEATURE_COLS], ytr)):
            tr = pool.iloc[tr_idx].copy()
            tr["label"] = np.where(ytr[tr_idx] == 1, "pos", "neg")
            va = pool.iloc[va_idx]
            res = c.predict_classification(tr[FEATURE_COLS + ["label"]], va[FEATURE_COLS],
                                           target="label", tag=f"oof_{model}_{pos}_f{k}")
            oof[va_idx] = pos_prob(res)
        thr = search(ytr, oof)

        # ---- 测试：全池作上下文预测 60 场 ----
        tr_all = pool.copy(); tr_all["label"] = np.where(ytr == 1, "pos", "neg")
        res_te = c.predict_classification(tr_all[FEATURE_COLS + ["label"]], test[FEATURE_COLS],
                                          target="label", tag=f"test_{model}_{pos}")
        pte = pos_prob(res_te)
        auc = round(float(roc_auc_score(yte, pte)), 4) if len(set(yte)) > 1 else None
        prauc = round(float(average_precision_score(yte, pte)), 4) if len(set(yte)) > 1 else None

        for m in thr:                       # 每个指标选出的阈值，到测试集上评估
            thr[m]["test"] = test_metrics(yte, pte, thr[m]["threshold"])
        report["tasks"][tname] = {"positive": pos, "pos_rate_test": round(float(yte.mean()), 3),
                                  "thresholds": thr, "test_auc": auc, "test_pr_auc": prauc,
                                  "test_prob_max": round(float(pte.max()), 3),
                                  "test_prob_mean": round(float(pte.mean()), 3)}
        print(f"  {tname}: f1阈={thr['f1']['threshold']} prec阈={thr['precision']['threshold']} "
              f"AUC={auc} 测试集最大正类概率={pte.max():.3f}  ({time.time()-t0:.0f}s)")

    out = f"{_OUT}/oof_threshold_{model}.json"
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
