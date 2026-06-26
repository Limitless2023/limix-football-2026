# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 results/{pred_matrix,title_odds,backtest_*}.json|csv + tournament_2026
[OUTPUT]: 汇总为 web/snapshot.json（网页唯一数据源：矩阵/夺冠概率/积分/实时成绩单/回测）
[POS]: limix-football预测 的网页数据装配层，把离线产物收口成单一快照，支撑纯静态前端
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import os
import json
import pandas as pd
from tournament_2026 import (GROUPS, TEAMS, CN_NAME, compute_standings,
                             played_matches, remaining_group_fixtures)

_DIR = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_DIR, "results")
_WEB = os.path.join(_DIR, "web")

# 国旗 emoji（区域指示符）
_FLAG = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}


def _load(name):
    with open(os.path.join(_RES, name)) as f:
        return json.load(f)


def _backtest(tag):
    p = os.path.join(_RES, f"backtest_{tag}_summary.json")
    return _load(f"backtest_{tag}_summary.json") if os.path.exists(p) else None


def main():
    os.makedirs(_WEB, exist_ok=True)
    matrix = _load("pred_matrix.json")
    odds = _load("title_odds.json")

    # ---- 当前小组积分榜 ----
    st = compute_standings()
    groups_view = {}
    for g, ts in GROUPS.items():
        ranked = sorted(ts, key=lambda t: (-st[t]["pts"], -st[t]["gd"], -st[t]["gf"]))
        groups_view[g] = [{"team": t, "cn": CN_NAME[t], "flag": _FLAG.get(t, "🏳️"),
                           **st[t]} for t in ranked]

    # ---- 2026 实时成绩单（已踢40场 + LimiX 预测对照）----
    det = pd.read_csv(os.path.join(_RES, "backtest_2026_detail.csv"))
    live, correct = [], 0
    for r in det.itertuples():
        ok = (r.pred_label == r.label)
        correct += int(ok)
        live.append({
            "date": r.date, "home": r.home_team, "away": r.away_team,
            "hcn": CN_NAME.get(r.home_team, r.home_team), "acn": CN_NAME.get(r.away_team, r.away_team),
            "hflag": _FLAG.get(r.home_team, "🏳️"), "aflag": _FLAG.get(r.away_team, "🏳️"),
            "hs": int(r.home_score), "as": int(r.away_score),
            "p_home": round(r.p_home, 3), "p_draw": round(r.p_draw, 3), "p_away": round(r.p_away, 3),
            "pred": r.pred_label, "pred_hs": int(r.pred_home_score), "pred_as": int(r.pred_away_score),
            "ok": bool(ok),
        })
    live_acc = round(correct / len(live), 4) if live else 0

    # ---- 前瞻预测：32 场未开踢小组赛（预测取大矩阵，解释取 LimiX local）----
    pairs = matrix["pairs"]
    expl = _load("explanations.json")["explanations"] if os.path.exists(os.path.join(_RES, "explanations.json")) else {}
    upcoming = []
    for g, a, b in remaining_group_fixtures():
        p = pairs.get(f"{a}|{b}")
        if not p:
            continue
        ex = expl.get(f"{a}|{b}", {})
        upcoming.append({
            "group": g, "home": a, "away": b,
            "hcn": CN_NAME[a], "acn": CN_NAME[b],
            "hflag": _FLAG.get(a, "🏳️"), "aflag": _FLAG.get(b, "🏳️"),
            "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
            "gh": round(p["gh"]), "ga": round(p["ga"]),
            "concepts": ex.get("concepts", [])[:5],
        })
    upcoming.sort(key=lambda m: (m["group"], -max(m["p_home"], m["p_away"])))

    # ---- 逐日滚动三方对照(默认/阈值/LLM)----
    wf = None
    if os.path.exists(os.path.join(_RES, "daily_walkforward.json")):
        wfd = _load("daily_walkforward.json")
        CNL = {"home_win": "主胜", "draw": "平", "away_win": "客胜"}
        s = wfd["summary"]
        n = s["n"]

        def _cell(p, actual):
            return {"pred": CNL[p], "ok": p == actual}
        wf_rows = [{
            "date": r["date"], "score": r["score"], "actual": CNL[r["actual"]],
            "hcn": CN_NAME.get(r["home"], r["home"]), "acn": CN_NAME.get(r["away"], r["away"]),
            "hflag": _FLAG.get(r["home"], "🏳️"), "aflag": _FLAG.get(r["away"], "🏳️"),
            "default": _cell(r["limix_default"], r["actual"]),
            "thr": _cell(r["limix_thr"], r["actual"]),
            "llm": _cell(r["llm"], r["actual"]),
        } for r in wfd["rows"]]
        wf = {"n": n, "draw_total": s["draw_total"], "rows": wf_rows, "methods": {
            "default": {"name": "LimiX 默认", "acc": round(s["hits_default"] / n, 3), "draw": s["draw_recall_default"]},
            "thr": {"name": "LimiX 阈值 best_f1", "acc": round(s["hits_thr"] / n, 3), "draw": s["draw_recall_thr"]},
            "llm": {"name": "LLM 基线", "acc": round(s["llm_hits"] / n, 3), "draw": s["llm_draw_recall"]},
        }}

    snapshot = {
        "generated": matrix["generated"],
        "model": matrix["model"],
        "teams": TEAMS,
        "cn": CN_NAME,
        "flags": _FLAG,
        "ratings": matrix["ratings"],
        "pairs": matrix["pairs"],
        "groups": groups_view,
        "title_odds": odds["odds"],
        "n_sims": odds["n_sims"],
        "live": {"matches": live, "accuracy": live_acc, "n": len(live)},
        "upcoming": upcoming,
        "walkforward": wf,
        "backtests": {k: _backtest(k) for k in ("broad",) if _backtest(k)},
    }
    out = os.path.join(_WEB, "snapshot.json")
    with open(out, "w") as f:
        json.dump(snapshot, f, ensure_ascii=False)
    kb = os.path.getsize(out) / 1024
    print(f"[saved] web/snapshot.json ({kb:.0f} KB)：{len(TEAMS)}队 "
          f"{len(matrix['pairs'])}配对 实时{len(live)}场(准确率{live_acc*100:.1f}%) "
          f"前瞻{len(upcoming)}场 夺冠Top1={odds['odds'][0]['cn']}({odds['odds'][0]['champion']*100:.1f}%)")


if __name__ == "__main__":
    main()
