# LimiX 足球预测演示 ⚽

用 **LimiX 新版 API** 复刻并超越 Prior Labs 足球预测 demo（`ux.priorlabs.ai/football`）。
读完 1872–2026、近 5 万场国际比赛，给出每场**胜平负概率 + 精确比分**，模拟整届 2026 世界杯推演冠军，并与传统机器学习诚实对比。

> 一句话定位：**Prior Labs 告诉你谁会赢，LimiX 告诉你赢几比几、为什么、凭什么信。**

## 核心成绩（真值可对照）

| 场景 | 样本 | LimiX 胜平负准确率 | 备注 |
|---|---|---|---|
| 大样本回测 | 2584 场 | **60.1% 🏆** | 零调参一次推理，超过调参后的 XGBoost/RF/LogReg |
| 2026 世界杯（在打） | 40 场 | **57.5% 🏆** | 真实结果持续验证 |
| 2022 世界杯 | 64 场 | 50.0% | 爆冷如云，四模型同处一线 |

比分回归场均误差仅 **0.94 球**——传统分类模型做不到。
夺冠概率 Top：西班牙 21.8% / 阿根廷 19.3% / 法国 13.7% / 英格兰 11.8%。

## 快速开始

```bash
# 1) 数据（首次自动从 GitHub 拉取并缓存到 data/）
python3 data_loader.py

# 2) 回测验证（命令行，直接看 LimiX vs 传统ML 成绩单）
python3 run_backtest.py broad      # 大样本（最强论据）
python3 run_backtest.py 2026       # 在打的 2026 世界杯
python3 run_backtest.py 2022       # 单届故事

# 3) 生成 2026 全量预测 + 蒙特卡洛 + 解释 + 网页快照
python3 predictor.py               # 48强全配对预测矩阵 → results/pred_matrix.json
python3 simulate.py                # 蒙特卡洛夺冠概率 → results/title_odds.json
python3 explain.py                 # 32场未开踢比赛的 LimiX 局部解释 → results/explanations.json
python3 build_web_data.py          # 汇总 → web/snapshot.json

# 4) 看网页（纯静态，零后端）
cd web && python3 -m http.server 8077
# 浏览器打开 http://127.0.0.1:8077/index.html
```

## 架构

```
data_loader.py      数据接入：GitHub raw 拉 results/shootouts/goalscorers
features.py         特征工程：23特征单遍扫描，严守无未来泄漏（FeatureEngine）
limix_client.py     模型推理：上传→分类/回归 predict→下载（LimiXClient）
tournament_2026.py  赛制配置：官方12组 + 当前积分 + 剩余赛程
run_backtest.py     价值证明：胜平负+比分+PK传统ML+回测（年份 / broad）
predictor.py        预测核心：48强全配对矩阵（网页与模拟共享）
simulate.py         赛事推演：蒙特卡洛模拟整届 → 夺冠概率
explain.py          可解释：LimiX local 解释接口算未开踢比赛特征归因（标签须整数编码）
build_web_data.py   网页装配：聚合所有产物 → snapshot.json
web/index.html      前端：纯静态单页，6大板块（夺冠概率/对阵预测/实时成绩单/前瞻预测+解释/回测/方法论）
```

数据流：`data_loader → features → {run_backtest, predictor} → {simulate} → build_web_data → web`

## 设计要点

- **无未来泄漏**：特征按时间单遍扫描，开球时刻只用此前比赛——可信度基石。
- **预计算矩阵**：48强全 2256 配对一次推理固化成矩阵，网页交互零延迟、离线可跑、永不挂。
- **诚实**：回测真值对照，LimiX 微弱但稳定领先，不夸大碾压——对标 Prior Labs"让价值自己说话"。

## 数据与对标
- 数据集：[martj42 International Football Results 1872–2026](https://github.com/martj42/international_results)
- 对标 demo：[ux.priorlabs.ai/football](https://ux.priorlabs.ai/football)
- LimiX API：`test001-limix.stable-ai.cn`（见 `../../07-api合集/`）
