// /api/predict — 在线调 LimiX 现场推理（密钥在服务端，浏览器只发两队名）
// 流程：查内置特征行 → 上传1行 → 并行 分类+主/客比分回归 → 返回概率+比分
// [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
const fs = require("fs");
const path = require("path");

const BASE = (process.env.LIMIX_BASE || "https://test001-limix.stable-ai.cn").replace(/\/$/, "");
const KEY = process.env.LIMIX_API_KEY || "";
const MODEL = process.env.LIMIX_MODEL || "LIMIX_16M";
const DATA = path.join(__dirname, "_data");

// 内置特征矩阵（2256 配对），模块级只读一次
const PF = JSON.parse(fs.readFileSync(path.join(DATA, "pair_features.json"), "utf8"));
// 训练上下文 vid 缓存（冷启动首次上传，暖实例复用）
let VID = null;

async function upload(csv, name) {
  const fd = new FormData();
  fd.append("file", new Blob([csv], { type: "text/csv" }), `${name}.csv`);
  fd.append("encoding", "utf-8-sig");
  fd.append("delimiter", ",");
  const r = await fetch(`${BASE}/v1/data/files`, { method: "POST", headers: { "X-API-KEY": KEY }, body: fd });
  const j = await r.json();
  if (j.code !== 0) throw new Error(`upload ${name}: ${j.code} ${j.message}`);
  return j.data.data_version_id;
}

async function ensureTrains() {
  if (VID) return VID;
  const [cls, rh, ra] = await Promise.all([
    upload(fs.readFileSync(path.join(DATA, "train_cls.csv"), "utf8"), "train_cls"),
    upload(fs.readFileSync(path.join(DATA, "train_rh.csv"), "utf8"), "train_rh"),
    upload(fs.readFileSync(path.join(DATA, "train_ra.csv"), "utf8"), "train_ra"),
  ]);
  VID = { cls, rh, ra };
  return VID;
}

async function infer(kind, trainVid, predictVid, target) {
  const ep = kind === "cls" ? "classification" : "regression";
  const r = await fetch(`${BASE}/v1/inference/${ep}/predict`, {
    method: "POST",
    headers: { "X-API-KEY": KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ train_data_version_id: trainVid, predict_data_version_id: predictVid,
      target_column: target, model_type: MODEL }),
  });
  const j = await r.json();
  if (j.code !== 0) { const e = new Error(`${ep}: ${j.code} ${j.message}`); e.code = j.code; throw e; }
  return j.data.result_version_id;
}

async function download(vid) {
  const r = await fetch(`${BASE}/v1/data/download/${vid}`, { headers: { "X-API-KEY": KEY } });
  const txt = (await r.text()).replace(/^﻿/, "").trim();
  const [head, row] = txt.split(/\r?\n/);
  const cols = head.split(","), vals = row.split(",");
  const o = {}; cols.forEach((c, i) => (o[c.trim()] = vals[i]));
  return o;
}

module.exports = async (req, res) => {
  const t0 = Date.now();
  try {
    if (!KEY) return res.status(500).json({ error: "LIMIX_API_KEY 未配置" });
    const url = new URL(req.url, "http://x");
    const home = url.searchParams.get("home"), away = url.searchParams.get("away");
    if (!home || !away || home === away) return res.status(400).json({ error: "需要不同的 home / away" });
    const feats = PF.pairs[`${home}|${away}`];
    if (!feats) return res.status(404).json({ error: `无此配对：${home} vs ${away}` });

    let trains = await ensureTrains();
    const predictCsv = PF.cols.join(",") + "\n" + feats.join(",") + "\n";

    const run = async () => {
      const pv = await upload(predictCsv, "predict");
      const [cv, hv, av] = await Promise.all([
        infer("cls", trains.cls, pv, "label"),
        infer("reg", trains.rh, pv, "home_score"),
        infer("reg", trains.ra, pv, "away_score"),
      ]);
      return Promise.all([download(cv), download(hv), download(av)]);
    };

    let cls, rh, ra;
    try { [cls, rh, ra] = await run(); }
    catch (e) { if (e.code === 4004) { VID = null; trains = await ensureTrains(); [cls, rh, ra] = await run(); } else throw e; }

    let pa = +cls.pred_probs_0, pd = +cls.pred_probs_1, ph = +cls.pred_probs_2;
    const s = pa + pd + ph; pa /= s; pd /= s; ph /= s;
    return res.status(200).json({
      home, away, model: MODEL,
      p_home: +ph.toFixed(4), p_draw: +pd.toFixed(4), p_away: +pa.toFixed(4),
      gh: +(+rh.pred_label).toFixed(2), ga: +(+ra.pred_label).toFixed(2),
      pred: cls.pred_label, secs: +((Date.now() - t0) / 1000).toFixed(1),
    });
  } catch (e) {
    return res.status(500).json({ error: String(e.message || e), secs: +((Date.now() - t0) / 1000).toFixed(1) });
  }
};

module.exports.config = { maxDuration: 60 };
