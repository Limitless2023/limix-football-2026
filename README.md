# LimiX · 2026 World Cup Prediction ⚽

用 **LimiX 结构化数据基础模型** 复刻并超越 Prior Labs 足球预测 demo（[ux.priorlabs.ai/football](https://ux.priorlabs.ai/football)）。
读完 1872–2026、近 5 万场国际比赛，对每场给出**胜平负概率 + 精确比分 + 逐场可解释**，并蒙特卡洛模拟整届 2026 世界杯推演冠军。

> **Prior Labs 告诉你谁会赢，LimiX 告诉你赢几比几、为什么、凭什么信。**

## 🌐 在线演示

本仓库根目录即静态站点（`index.html` + `snapshot.json`），Vercel 零配置可部署。

## ✨ 六大板块

1. **夺冠概率** — 蒙特卡洛 2 万次模拟（西班牙 21.8% / 阿根廷 19.3% / 法国 13.7%）
2. **对阵预测器** — 任选两队，即时出胜平负概率 + 预测比分（读 2256 配对离线矩阵）
3. **实时成绩单** — 2026 世界杯已踢 40 场，赛前预测 vs 真值，命中率 57.5%
4. **前瞻预测** — 剩余 32 场未开踢比赛的赛前预测 + LimiX 局部解释
5. **回测成绩单** — 大样本 2584 场 LimiX **60.1% 居首**，零调参反超调参后的 XGBoost/RF/LogReg
6. **方法论** — 六大类无泄漏特征 + 与 Prior Labs 逐条对比

## 📊 核心成绩（真值可对照）

| 场景 | 样本 | LimiX 胜平负准确率 |
|---|---|---|
| 大样本回测 | 2584 场 | **60.1% 🏆** |
| 2026 世界杯（在打） | 40 场 | **57.5% 🏆** |
| 比分回归 | — | 场均误差 0.94 球 |

## 🔧 复现管道

完整 Python 管道在 [`pipeline/`](pipeline/)：数据接入 → 无泄漏特征 → LimiX 推理 → 回测/蒙特卡洛/解释 → 网页快照。
设 `LIMIX_API_KEY` 环境变量后运行，详见 [pipeline/README.md](pipeline/README.md)。

```
data_loader → features → {run_backtest | predictor → simulate | explain} → build_web_data → web
```

## 数据与对标
- 数据集：[martj42 International Football Results 1872–2026](https://github.com/martj42/international_results)
- 模型：LimiX 结构化数据基础模型（分类 / 回归 / 时序 / 解释）
- 对标：[Prior Labs / TabPFN](https://ux.priorlabs.ai/football)（已被 SAP 收购）

---
🤖 Built with LimiX · 预测仅供演示，足球魅力恰在不可预测。
