# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 requests/pandas；对接 test001-limix.stable-ai.cn 新版 API
[OUTPUT]: 对外提供 LimiXClient（upload/classify/regress/download + predict_* 高阶封装）
[POS]: limix-football预测 的模型推理层，连接特征表与 LimiX 基础模型
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import os
import io
import time
import tempfile
import requests
import pandas as pd

_BASE = "https://test001-limix.stable-ai.cn"
_API_KEY = os.environ.get("LIMIX_API_KEY", "")   # 从环境变量读取，勿硬编码


class LimiXError(RuntimeError):
    pass


class LimiXClient:
    def __init__(self, base=_BASE, api_key=_API_KEY, model_type="LIMIX_16M", timeout=300):
        self.base = base.rstrip("/")
        self.h = {"X-API-KEY": api_key}
        self.model_type = model_type
        self.timeout = timeout

    # ---------------- 基础 ----------------
    def health(self) -> dict:
        r = requests.get(f"{self.base}/v1/health", headers=self.h, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path, json) -> dict:
        r = requests.post(f"{self.base}{path}", headers=self.h, json=json, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if j.get("code") != 0:
            raise LimiXError(f"{path} -> code={j.get('code')} msg={j.get('message')}")
        return j["data"]

    # ---------------- 数据接入 ----------------
    def upload(self, df: pd.DataFrame, name: str) -> str:
        """上传 DataFrame，返回 data_version_id。"""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=True,
                                         encoding="utf-8-sig", newline="") as f:
            df.to_csv(f, index=False)
            f.flush()
            f.seek(0)
            files = {"file": (f"{name}.csv", open(f.name, "rb"), "text/csv")}
            data = {"encoding": "utf-8-sig", "delimiter": ","}
            r = requests.post(f"{self.base}/v1/data/files", headers=self.h,
                              files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if j.get("code") != 0:
            raise LimiXError(f"upload {name} -> code={j.get('code')} msg={j.get('message')}")
        d = j["data"]
        print(f"[limix] upload {name}: {d['row_count']}行×{d['column_count']}列 "
              f"-> {d['data_version_id']} ({d['task_status']})")
        return d["data_version_id"]

    def download(self, version_id: str) -> pd.DataFrame:
        """按 data_version_id 或 result_version_id 下载文件流，解析为 DataFrame。"""
        r = requests.get(f"{self.base}/v1/data/download/{version_id}",
                         headers=self.h, timeout=self.timeout)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content))

    # ---------------- 同步推理 ----------------
    def classify(self, train_vid, predict_vid, target="label", config_name=None) -> str:
        body = {"train_data_version_id": train_vid, "predict_data_version_id": predict_vid,
                "target_column": target, "model_type": self.model_type}
        if config_name:
            body["config_name"] = config_name
        d = self._post("/v1/inference/classification/predict", body)
        return d["result_version_id"]

    def regress(self, train_vid, predict_vid, target, config_name=None) -> str:
        body = {"train_data_version_id": train_vid, "predict_data_version_id": predict_vid,
                "target_column": target, "model_type": self.model_type}
        if config_name:
            body["config_name"] = config_name
        d = self._post("/v1/inference/regression/predict", body)
        return d["result_version_id"]

    # ---------------- 高阶：DataFrame 进，DataFrame 出 ----------------
    def predict_classification(self, train_df, predict_df, target="label", tag="cls") -> pd.DataFrame:
        """train_df 含 target 列；predict_df 不含 target。返回含 pred_probs_* + pred_label 的结果。"""
        t0 = time.time()
        tr = self.upload(train_df, f"{tag}_train")
        pr = self.upload(predict_df, f"{tag}_predict")
        rv = self.classify(tr, pr, target=target)
        res = self.download(rv)
        print(f"[limix] {tag} 分类完成：{len(res)}行，耗时 {time.time()-t0:.1f}s")
        return res

    def predict_regression(self, train_df, predict_df, target, tag="reg") -> pd.DataFrame:
        t0 = time.time()
        tr = self.upload(train_df, f"{tag}_train")
        pr = self.upload(predict_df, f"{tag}_predict")
        rv = self.regress(tr, pr, target=target)
        res = self.download(rv)
        print(f"[limix] {tag} 回归完成({target})：{len(res)}行，耗时 {time.time()-t0:.1f}s")
        return res


if __name__ == "__main__":
    c = LimiXClient()
    print("health:", c.health())
