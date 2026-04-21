#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import re
from itertools import combinations
import json
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    mean_squared_error, silhouette_score
)
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier, LGBMRegressor
from openai import OpenAI
import time
import random
import datetime
import os
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def sanitize_feature_name(name: str) -> str:
    name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


# In[3]:


def operator_and_template_prompt(
    columns,
    task_desc,
    feat_desc,
    task_type,
    base_ops,
    base_templates
):
    return f"""
You are an expert in Automatic Feature Engineering (CAAFE Phase 2).

You will receive:
- Dataset feature names
- Task description and feature semantics
- Task type: "{task_type}"
- A small base operator library and base template library

Your goal:
- Expand the operator whitelist and template whitelist based on the provided bases.
- Keep templates python-like and executable.
- Templates must use only x, y, z as placeholders.
- Templates should be generally useful and numerically stable (e.g., divisions use +1e-6).

Return ONLY a JSON object with EXACT keys:

{{
  "operator_whitelist": {{
    "unary": [...],
    "binary": [...],
    "ternary": [...]
  }},
  "template_whitelist": {{
    "unary": [...],
    "binary": [...],
    "ternary": [...]
  }},
  "meta": {{
    "task_type": "{task_type}",
    "expanded_from_base": true
  }}
}}

Guidelines for expansion:
- Keep it concise: add ~5-15 templates per arity.
- Prefer robust transforms: abs/log1p/sqrt, ratios with eps, interactions, normalized differences.
- Avoid overly redundant variants.

----------------------------
### Feature Names
{columns}

### Task Description
{task_desc}

### Feature Semantics
{feat_desc}

### Base Operator Whitelist
{json.dumps(base_ops, ensure_ascii=False)}

### Base Template Whitelist
{json.dumps(base_templates, ensure_ascii=False)}

----------------------------
Return ONLY JSON. No extra text.
"""


# In[4]:


def expand_operators_and_templates(
    columns,
    base_operator_whitelist,
    base_template_whitelist,
    task_type,
    llm_client=None,
    task_desc=None,
    feat_desc=None,
):
    """
    Stage 0: Operator & Template Expansion

    - 若 llm_client=None → 直接返回 base
    - 否则：调用 LLM 扩充，并与 base 合并（去重、保序）
    """

    # =============================
    # 0. fallback：不使用 LLM
    # =============================
    if llm_client is None:
        return base_operator_whitelist, base_template_whitelist

    if task_desc is None or feat_desc is None:
        raise ValueError(
            "[Stage0] 使用 LLM 扩充时，task_desc 与 feat_desc 不能为空"
        )

    # =============================
    # 1. 构造 Prompt
    # =============================
    prompt = operator_and_template_prompt(
        columns=columns,
        task_desc=task_desc,
        feat_desc=feat_desc,
        task_type=task_type,
        base_ops=base_operator_whitelist,
        base_templates=base_template_whitelist,
    )

    # =============================
    # 2. 调用 LLM
    # =============================
    resp = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()

    # 兼容 ```json ... ``` 包裹
    text = re.sub(r"^```json\s*|\s*```$", "", text).strip()

    llm_debug = {
        "raw_text": text,
        "parsed_json": None,
    }

    try:
        data = json.loads(text)
        llm_debug["parsed_json"] = data
    except Exception as e:
        raise RuntimeError(
            "[Stage0] LLM JSON 解析失败\n"
            f"Error: {e}\n"
            f"Raw output:\n{text}"
        )

    # =============================
    # 3. 结构校验
    # =============================
    for k in ["operator_whitelist", "template_whitelist"]:
        if k not in data:
            raise ValueError(f"[Stage0] LLM 输出缺少 ключ: {k}")

    # =============================
    # 4. 合并（去重 + 保序）
    # =============================
    def merge_keep_order(base_list, new_list):
        seen = set()
        out = []
        for x in base_list + new_list:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    operator_whitelist = {
        "unary": merge_keep_order(
            base_operator_whitelist.get("unary", []),
            data["operator_whitelist"].get("unary", []),
        ),
        "binary": merge_keep_order(
            base_operator_whitelist.get("binary", []),
            data["operator_whitelist"].get("binary", []),
        ),
        "ternary": merge_keep_order(
            base_operator_whitelist.get("ternary", []),
            data["operator_whitelist"].get("ternary", []),
        ),
    }

    template_whitelist = {
        "unary": merge_keep_order(
            base_template_whitelist.get("unary", []),
            data["template_whitelist"].get("unary", []),
        ),
        "binary": merge_keep_order(
            base_template_whitelist.get("binary", []),
            data["template_whitelist"].get("binary", []),
        ),
        "ternary": merge_keep_order(
            base_template_whitelist.get("ternary", []),
            data["template_whitelist"].get("ternary", []),
        ),
    }

    return operator_whitelist, template_whitelist, llm_debug


# ### Stage 1

# In[6]:


def fast_feature_screening(X, y, task_type, config):
    """
    单变量快评：MI + GBDT proxy
    """

    if task_type in ["classification", "multiclassification"]:
        mi = mutual_info_classif(X, y, discrete_features="auto")
    elif task_type == "regression":
        mi = mutual_info_regression(X, y, discrete_features="auto")
    else:  # cluster
        mi = X.var().values

    mi = pd.Series(mi, index=X.columns)

    gbdt_scores = []
    for col in X.columns:
        Xi = X[[col]]
        try:
            if task_type == "classification":
                model = LGBMClassifier(n_estimators=20, verbose=-1)
                model.fit(Xi, y)
                g = model.booster_.feature_importance("gain")[0]
            elif task_type == "multiclassification":
                model = LGBMClassifier(
                    n_estimators=20, num_class=len(np.unique(y)), verbose=-1
                )
                model.fit(Xi, y)
                g = model.booster_.feature_importance("gain")[0]
            elif task_type == "regression":
                model = LGBMRegressor(n_estimators=20, verbose=-1)
                model.fit(Xi, y)
                g = model.booster_.feature_importance("gain")[0]
            else:
                g = -KMeans(n_clusters=3).fit(Xi).inertia_
        except:
            g = 0.0
        gbdt_scores.append(g)

    gbdt = pd.Series(gbdt_scores, index=X.columns)

    mi_n = (mi - mi.min()) / (mi.max() - mi.min() + 1e-6)
    g_n = (gbdt - gbdt.min()) / (gbdt.max() - gbdt.min() + 1e-6)

    score = mi_n + 0.2 * g_n

    P = config["P"]
    return list(score.sort_values(ascending=False).head(P).index)


# ### Stage 2

# In[ ]:


def build_feature_graph(X_S, task_type, config, llm_client=None):
    """
    Stage 2: Feature Graph Construction

    W = alpha * numeric_similarity + (1 - alpha) * semantic_similarity
    G = kNN graph over W

    Additionally:
    - Sample a small set of feature pairs
    - Query LLM for semantic similarity + explanation
    - Return explanation for logging
    """

    cols = list(X_S.columns)
    n = len(cols)

    # =========================
    # 2.1 数值相似矩阵 Corr
    # =========================
    if task_type == "cluster":
        Corr = X_S.corr(method="spearman").abs()
    else:
        Corr = X_S.corr(method="pearson").abs()

    Corr = Corr.fillna(0.0)

    # =========================
    # 2.2 是否使用语义相似度（关闭后每轮仅 Stage0 一次 LLM，避免 O(n^2) 调用）
    # =========================
    use_semantic = (
        llm_client is not None
        and bool(config.get("use_llm_semantic_graph", True))
    )

    semantic_log = None   # ⭐ 用于返回写入日志

    if use_semantic:
        Sim_sem = pd.DataFrame(
            np.zeros((n, n)),
            index=cols,
            columns=cols
        )

        # -------------------------------------------------
        # 2.2.1 抽样一组特征对（仅用于解释）
        # -------------------------------------------------
        sample_k = config.get("semantic_explain_k", 3)
        rng = np.random.default_rng(config.get("random_seed", 42))

        all_pairs = [
            (cols[i], cols[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]

        sampled_pairs = (
            rng.choice(len(all_pairs), size=min(sample_k, len(all_pairs)), replace=False)
            if len(all_pairs) > 0 else []
        )

        semantic_log = {
            "sampled_pairs": [],
        }

        # -------------------------------------------------
        # 2.2.2 询问 LLM：相似度 + 逻辑（仅抽样）
        # -------------------------------------------------
        for idx in sampled_pairs:
            f1, f2 = all_pairs[idx]
            try:
                r = llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": semantic_similarity_with_explanation_prompt(f1, f2)
                        }
                    ],
                    temperature=0.2,
                )

                content = r.choices[0].message.content.strip()
                content = re.sub(r"^```json\s*|\s*```$", "", content)

                parsed = json.loads(content)

                semantic_log["sampled_pairs"].append({
                    "feature1": f1,
                    "feature2": f2,
                    "similarity": parsed.get("similarity"),
                    "logic": parsed.get("logic"),
                })

            except Exception as e:
                semantic_log["sampled_pairs"].append({
                    "feature1": f1,
                    "feature2": f2,
                    "error": str(e),
                })

        # -------------------------------------------------
        # 2.2.3 正式 pairwise 语义相似度（无解释）
        # -------------------------------------------------
        def semantic_prompt(f1, f2):
            return f"""
You are an expert in feature semantics.
Estimate the semantic similarity between:

Feature1: {f1}
Feature2: {f2}

Return ONLY a float number in [0,1]
"""

        def get_similarity(f1, f2):
            try:
                r = llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": semantic_prompt(f1, f2)}],
                )
                sim = float(r.choices[0].message.content.strip())
                return max(0.0, min(1.0, sim))
            except:
                return 0.3

        for i in range(n):
            for j in range(i + 1, n):
                sim = get_similarity(cols[i], cols[j])
                Sim_sem.iloc[i, j] = sim
                Sim_sem.iloc[j, i] = sim

        np.fill_diagonal(Sim_sem.values, 1.0)

    else:
        Sim_sem = Corr.copy()

    # =========================
    # 2.3 融合权重矩阵 W
    # =========================
    alpha = config.get("alpha", 1.0)

    W = alpha * Corr.values + (1.0 - alpha) * Sim_sem.values
    W = pd.DataFrame(W, index=cols, columns=cols)

    # =========================
    # 2.4 kNN 邻接图
    # =========================
    k = config.get("graph_k", 8)

    G = {}
    for col in cols:
        neigh = (
            W.loc[col]
            .drop(col)
            .sort_values(ascending=False)
            .head(k)
            .index
            .tolist()
        )
        G[col] = neigh

    return G, semantic_log


# ### Stage 3 + 4

# In[10]:


def generate_second_order_features(df, y, S, G, template_whitelist, task_type, config):
    """
    Stage 3 + 4:
    - Generate second-order candidate features
    - Beam prune to top-B by score
    """

    binary_templates = template_whitelist["binary"]
    C2 = []

    # =========================
    # safe eval（与原脚本一致）
    # =========================
    # =========================
    def safe_eval(expr, x, y):
        try:
        # ---------------------------------
        # 0. 转 numpy，保证一致性
        # ---------------------------------
            x = np.asarray(x)
            y = np.asarray(y)

        # 长度检查
            if x.ndim != 1 or y.ndim != 1:
                return None
            if len(x) != len(y):
                return None

        # ---------------------------------
        # 1. 安全处理 0（避免除 0）
        # ---------------------------------
            y_safe = np.where(y == 0, 1e-6, y)

        # ---------------------------------
        # 2. eval 环境（关键！）
        # ---------------------------------
            local_env = {
                "x": x,
                "y": y_safe,
                "np": np,
                "log": np.log,
                "log1p": np.log1p,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "sign": np.sign,
            }

            out = eval(expr, local_env, {})
            out = np.asarray(out)

        # ---------------------------------
        # 3. 数值合法性检查
        # ---------------------------------
            if out.ndim != 1:
                return None

            if np.any(np.isnan(out)) or np.any(np.isinf(out)):
                return None

        # 常数 / 近常数直接丢
            if np.nanstd(out) < 1e-8:
                return None

            return out

        except Exception:
            return None




    # =========================
    # scoring（四任务统一）
    # =========================
    def compute_single_score(values):
        df_tmp = pd.DataFrame({"tmp": values})

        if task_type in ["classification", "multiclassification"]:
            return float(
                mutual_info_classif(df_tmp, y, discrete_features="auto")[0]
            )

        if task_type == "regression":
            return float(
                mutual_info_regression(df_tmp, y, discrete_features="auto")[0]
            )

        if task_type == "cluster":
            try:
                X_tmp = values.reshape(-1, 1)
                km = KMeans(n_clusters=3, n_init=5)
                labels = km.fit_predict(X_tmp)
                sc = silhouette_score(X_tmp, labels)
                return float(max(sc, 0))
            except Exception:
                return 0.0

        raise ValueError(f"Unknown task_type: {task_type}")

    # =========================
    # Stage 3: 二阶候选生成
    # =========================
    for i in S:
        for j in G.get(i, []):
            if i == j:
                continue

            x = df[i]
            y2 = df[j]

            for template in binary_templates:
                values = safe_eval(template, x, y2)
                if values is None:
                    continue

                score = compute_single_score(values)
                name = sanitize_feature_name(f"{i}_{template}_{j}")

                C2.append({
                    "name": name,
                    "columns": (i, j),
                    "template": template,
                    "expr": template,
                    "values": values,
                    "score": score,
                })

    # =========================
    # Stage 4: Beam 精筛
    # =========================
    B = config["B"]
    if len(C2) == 0:
        return []

    C2_sorted = sorted(C2, key=lambda x: x["score"], reverse=True)
    C2_top = C2_sorted[:min(B, len(C2_sorted))]

    return C2_top



# ### Stage 5

# In[12]:


def generate_third_order_features(df, y, C2, G, template_whitelist, task_type, config):
    """
    Stage 5: Third-order feature expansion
    """

    ternary_templates = template_whitelist.get("ternary", [])
    T = config["T"]

    if len(ternary_templates) == 0 or len(C2) == 0:
        return []

    C3 = []

    # =========================
    # safe eval（三元）
    # =========================
    # =========================
    # safe eval（三元）
    # =========================
    def safe_eval_ternary(expr, x, y, z):
        try:
            # ---------------------------------
            # 0. 转 numpy，保证一致性
            # ---------------------------------
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)

            if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
                return None
            if not (len(x) == len(y) == len(z)):
                return None

            # ---------------------------------
            # 1. 安全处理 0
            # ---------------------------------
            y_safe = np.where(y == 0, 1e-6, y)
            z_safe = np.where(z == 0, 1e-6, z)

            # ---------------------------------
            # 2. eval 环境（必须齐全）
            # ---------------------------------
            local_env = {
                "x": x,
                "y": y_safe,
                "z": z_safe,
                "np": np,
                "log": np.log,
                "log1p": np.log1p,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "sign": np.sign,
            }

            out = eval(expr, local_env, {})
            out = np.asarray(out)

            # ---------------------------------
            # 3. 数值合法性检查
            # ---------------------------------
            if out.ndim != 1:
                return None

            if np.any(np.isnan(out)) or np.any(np.isinf(out)):
                return None

            # 常数 / 近常数直接丢
            if np.nanstd(out) < 1e-8:
                return None

            return out

        except Exception:
            return None



    # =========================
    # scoring（与 Stage 3 统一）
    # =========================
# =========================
# scoring（与 Stage 3 统一）
# =========================
    def compute_single_score(values):

    # ---------------------------------
    # 0. None / 类型安全
    # ---------------------------------
        if values is None:
            return None

        values = np.asarray(values)

    # 必须是一维
        if values.ndim != 1:
            return None

    # 必须与 y 对齐
        if task_type != "cluster" and len(values) != len(y):
            return None

    # 排除 NaN / Inf
        if np.any(np.isnan(values)) or np.any(np.isinf(values)):
            return None

    # 排除近似常数
        if np.nanstd(values) < 1e-8:
            return None

    # ---------------------------------
    # 1. 分类 / 回归（MI）
    # ---------------------------------
        if task_type in ["classification", "multiclassification"]:
            df_tmp = pd.DataFrame({"tmp": values})
            return float(
                mutual_info_classif(
                    df_tmp,
                    y,
                    discrete_features="auto"
                )[0]
            )

        if task_type == "regression":
            df_tmp = pd.DataFrame({"tmp": values})
            return float(
                mutual_info_regression(
                    df_tmp,
                    y,
                    discrete_features="auto"
                )[0]
            )

    # ---------------------------------
    # 2. 聚类（silhouette）
    # ---------------------------------
        if task_type == "cluster":
            try:
                arr = values.reshape(-1, 1)
                km = KMeans(
                    n_clusters=3,
                    n_init=5
                )
                labels = km.fit_predict(arr)
                sc = silhouette_score(arr, labels)
                return float(max(sc, 0))
            except Exception:
                return 0.0

        raise ValueError(f"Unknown task_type: {task_type}")


    # =========================
    # Stage 5: 三阶扩展
    # =========================
    for item in C2:
        p, q = item["columns"]

        # r 来自 p / q 的邻居
        candidate_r = (set(G.get(p, [])) | set(G.get(q, []))) - {p, q}

        x = df[p]
        y2 = df[q]

        for r in sorted(candidate_r):
            z = df[r]

            for template in ternary_templates:
                values = safe_eval_ternary(template, x, y2, z)
                if values is None:
                    continue

                score = compute_single_score(values)
                name = sanitize_feature_name(f"{p}_{q}_{r}_{template}")

                C3.append({
                    "name": name,
                    "columns": (p, q, r),
                    "template": template,
                    "expr": template,
                    "values": values,
                    "score": score,
                })

                if len(C3) >= T:
                    return C3

    return C3


# ### Stage 6

# In[14]:


def stability_selection(df, y, C_candidates, task_type, config):
    """
    Stage 6: Stability selection
    """

    if len(C_candidates) == 0:
        return []

    # =========================
    # 构造候选特征矩阵
    # =========================
    X_new = pd.DataFrame({c["name"]: c["values"] for c in C_candidates})

    K = config.get("K", 20)
    tau = config.get("tau", 0.5)
    N = config.get("N", 20)

    stability_count = pd.Series(0, index=X_new.columns)

    # =========================
    # 重采样循环
    # =========================
    for k in range(K):

        # ---------- supervised ----------
        if task_type != "cluster":
            X_tr, _, y_tr, _ = train_test_split(
                X_new, y, test_size=0.3, random_state=42 + k
            )

        # ---------- classification ----------
        if task_type == "classification":
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)

            try:
                model = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=500
                )
                model.fit(X_tr_s, y_tr)
                coef = np.abs(model.coef_).flatten()
            except Exception:
                model = LGBMClassifier(n_estimators=60)
                model.fit(X_tr, y_tr)
                coef = model.feature_importances_

            stability_count[coef > 0] += 1

        # ---------- multiclass ----------
        elif task_type == "multiclassification":
            model = LGBMClassifier(n_estimators=60)
            model.fit(X_tr, y_tr)
            coef = model.feature_importances_
            stability_count[coef > 0] += 1

        # ---------- regression ----------
        elif task_type == "regression":
            model = LGBMRegressor(n_estimators=60)
            model.fit(X_tr, y_tr)
            coef = model.feature_importances_
            stability_count[coef > 0] += 1

        # ---------- cluster ----------
        elif task_type == "cluster":
            for col in X_new.columns:
                v = X_new[col].values.reshape(-1, 1)
                try:
                    km = KMeans(n_clusters=3, n_init=5)
                    labels = km.fit_predict(v)
                    sil = silhouette_score(v, labels)
                except Exception:
                    sil = -1

                if sil > 0:
                    stability_count[col] += 1

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    # =========================
    # 稳定性评分
    # =========================
    stability_score = stability_count / K

    # =========================
    # 筛选规则
    # =========================
    if task_type != "cluster":
        selected = list(stability_score[stability_score >= tau].index)
    else:
        selected = list(
            stability_score.sort_values(ascending=False).head(N).index
        )

    # ---------- 不足 N 则补齐 ----------
    if len(selected) < N:
        selected = list(
            stability_score.sort_values(ascending=False).head(N).index
        )

    return selected


# ### Stage 7

# In[16]:


def redundancy_removal(selected_names, C_candidates, config):
    """
    Stage 7: Redundancy Removal (on generated features)

    Parameters
    ----------
    selected_names : List[str]
        特征名（来自稳定性选择）
    C_candidates : List[dict]
        C2 + C3，每个 dict 至少包含:
        {
            "name": str,
            "values": np.ndarray
        }
    config : dict
        包含 rho_max

    Returns
    -------
    List[str]
        去冗余后的特征名
    """

    if len(selected_names) <= 1:
        return selected_names

    rho = config["rho_max"]

    # 1️⃣ 构造 feature value 表
    feat_value_map = {
        item["name"]: item["values"]
        for item in C_candidates
        if item["name"] in selected_names
    }

    if len(feat_value_map) <= 1:
        return list(feat_value_map.keys())

    X = pd.DataFrame(feat_value_map)

    # 2️⃣ 相关矩阵
    corr = X.corr().abs().fillna(0.0)

    kept = []
    removed = set()

    names = list(X.columns)

    # 3️⃣ 去冗余
    for i, f in enumerate(names):
        if f in removed:
            continue

        kept.append(f)

        for g in names[i + 1:]:
            if g in removed:
                continue

            if corr.loc[f, g] > rho:
                removed.add(g)

    return kept


# ### Stage 8 + 9

# In[18]:


def cv_forward_selection(df, y, feature_names, C_candidates, task_type, config):
    """
    Stage 8–9: CV-based greedy forward selection on generated features
    """

    if len(feature_names) == 0:
        return []

    N = config.get("N", 20)
    epsilon = config.get("epsilon", 1e-4)
    kfold = config.get("kfold", 5)
    task = task_type.lower()

    # -------------------------
    # 原始特征
    # -------------------------
    if task != "cluster":
        X_orig = df.drop(columns=[y.name])
    else:
        X_orig = df.copy()

    # -------------------------
    # 构造新特征矩阵（关键）
    # -------------------------
    feat_value_map = {
        item["name"]: item["values"]
        for item in C_candidates
        if item["name"] in feature_names
    }

    if len(feat_value_map) == 0:
        return []

    X_new = pd.DataFrame(feat_value_map)

    # -------------------------
    # CV 评分函数
    # -------------------------
    def cv_score(X):

        scores = []

        if task in ["classification", "multiclassification"]:
            splitter = StratifiedKFold(
                n_splits=kfold, shuffle=True, random_state=2024
            )
            split_iter = splitter.split(X, y)
        else:
            splitter = KFold(
                n_splits=kfold, shuffle=True, random_state=2024
            )
            split_iter = splitter.split(X)

        for tr, va in split_iter:
            X_tr, X_va = X.iloc[tr], X.iloc[va]

            if task != "cluster":
                y_tr, y_va = y.iloc[tr], y.iloc[va]

            try:
                if task == "classification":
                    model = LGBMClassifier(n_estimators=80, learning_rate=0.05)
                    model.fit(X_tr, y_tr)
                    pred = model.predict_proba(X_va)[:, 1]
                    scores.append(roc_auc_score(y_va, pred))

                elif task == "multiclassification":
                    Kc = len(np.unique(y))
                    model = LGBMClassifier(
                        n_estimators=80,
                        learning_rate=0.05,
                        num_class=Kc
                    )
                    model.fit(X_tr, y_tr)
                    pred = model.predict_proba(X_va)
                    scores.append(
                        f1_score(y_va, pred.argmax(1), average="macro")
                    )

                elif task == "regression":
                    model = LGBMRegressor(n_estimators=80, learning_rate=0.05)
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_va)
                    scores.append(-mean_squared_error(y_va, pred))

                elif task == "cluster":
                    km = KMeans(n_clusters=3, random_state=42)
                    labels = km.fit_predict(X_va)
                    scores.append(silhouette_score(X_va, labels))

            except:
                continue

        if len(scores) == 0:
            return -np.inf

        return float(np.mean(scores))

    # -------------------------
    # baseline
    # -------------------------
    current_score = cv_score(X_orig)

    selected = []
    remaining = list(X_new.columns)

    # -------------------------
    # 贪心前向搜索
    # -------------------------
    while remaining and len(selected) < N:

        best_gain = 0.0
        best_feat = None

        for f in remaining:
            X_tmp = pd.concat(
                [X_orig, X_new[selected + [f]]],
                axis=1
            )
            score = cv_score(X_tmp)
            gain = score - current_score

            if gain > best_gain:
                best_gain = gain
                best_feat = f

        if best_feat is None or best_gain < epsilon:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        current_score += best_gain

    return selected



# In[ ]:





# In[19]:


# cafee_pipeline.py
# =========================================================
# CAAFE Phase-2 Pipeline (Method Layer)
# =========================================================


# =========================================================
# 统一的结果数据结构（非常重要）
# =========================================================

class FS_method_result:
    """
    单次 AutoFE 运行的标准输出
    """
    def __init__(
        self,
        selected_features: List[str],
        feature_values: Dict[str, np.ndarray],
        feature_formulas: Dict[str, Dict[str, Any]],
        operator_whitelist: Dict[str, List[str]],
        template_whitelist: Dict[str, List[str]],
        debug_info: Dict[str, Any] = None
    ):
        self.selected_features = selected_features
        self.feature_values = feature_values
        self.feature_formulas = feature_formulas
        self.operator_whitelist = operator_whitelist
        self.template_whitelist = template_whitelist
        self.debug_info = debug_info or {}


def FS_method_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    config: Dict[str, Any],
    operator_base: Dict[str, List[str]],
    template_base: Dict[str, List[str]],
    llm_client=None,
    task_desc: str = None,
    feat_desc: str = None,
) -> FS_method_result:
    """
    单次 AutoFE Pipeline 执行入口（支持每轮内部使用 LLM）
    """

    # =====================================================
    # Step 0. 数据拆分
    # =====================================================
    has_target = target_col is not None and target_col in df.columns

    X = df.drop(columns=[target_col]) if has_target else df.copy()

    if task_type != "cluster" and has_target:
        y = df[target_col]
    else:
        y = None

    # =====================================================
    # Step 1. 算子 / 模板扩展（LLM）
    # =====================================================
    if llm_client is not None:
        if task_desc is None or feat_desc is None:
            raise ValueError(
                "[FS_method_pipeline] 使用 LLM 时，task_desc 与 feat_desc 必须显式传入"
            )

        # ⚠️ 你前面已改：返回 3 个值
        operator_whitelist, template_whitelist, stage0_llm_debug = \
            expand_operators_and_templates(
                X.columns.tolist(),
                operator_base,
                template_base,
                task_type,
                llm_client,
                task_desc=task_desc,
                feat_desc=feat_desc,
            )
    else:
        operator_whitelist = operator_base
        template_whitelist = template_base
        stage0_llm_debug = None

    # =====================================================
    # Step 2. 单变量快评
    # =====================================================
    S = fast_feature_screening(X, y, task_type, config)

    # =====================================================
    # Step 3. 构建特征图（⭐返回 G + semantic_log）
    # =====================================================
    G, semantic_log = build_feature_graph(
        X[S],
        task_type,
        config,
        llm_client=llm_client
    )

    # =====================================================
    # Step 4. 二阶候选
    # =====================================================
    C2 = generate_second_order_features(
        df, y, S, G, template_whitelist, task_type, config
    )

    # =====================================================
    # Step 5. 三阶候选
    # =====================================================
    C3 = generate_third_order_features(
        df, y, C2, G, template_whitelist, task_type, config
    )

    # =====================================================
    # Step 6. 稳定性选择
    # =====================================================
    S_stable = stability_selection(
        df, y, C2 + C3, task_type, config
    )

    # =====================================================
    # Step 7. 去冗余
    # =====================================================
    S_final = redundancy_removal(
        S_stable,
        C2 + C3,
        config
    )

    # =====================================================
    # Step 8. CV 一致性前向选择
    # =====================================================
    selected = cv_forward_selection(
        df,
        y,
        S_final,
        C2 + C3,
        task_type,
        config
    )

    # =====================================================
    # Step 9. 输出整理
    # =====================================================
    candidate_map = {item["name"]: item for item in (C2 + C3)}

    feature_values = {}
    feature_formulas = {}

    for name in selected:
        if name not in candidate_map:
            continue
        item = candidate_map[name]
        feature_values[name] = item["values"]
        feature_formulas[name] = {
            "columns": item["columns"],
            "template": item["template"],
        }

    # =====================================================
    # Step 10. 统一返回（⭐语义解释写入 debug_info）
    # =====================================================
    return FS_method_result(
        selected_features=list(feature_values.keys()),
        feature_values=feature_values,
        feature_formulas=feature_formulas,
        operator_whitelist=operator_whitelist,
        template_whitelist=template_whitelist,
        debug_info={
            "num_C2": len(C2),
            "num_C3": len(C3),
            "num_selected": len(feature_values),

            # ⭐ LLM 相关调试信息
            "stage0_llm": stage0_llm_debug,
            "semantic_similarity_log": semantic_log,
        }
    )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




