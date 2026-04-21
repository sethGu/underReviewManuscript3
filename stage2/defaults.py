"""Operator/template bases and ``CONFIG`` presets."""

from __future__ import annotations

from typing import Any, Dict

OPERATOR_BASE: Dict[str, list] = {
    "unary": ["abs", "log1p", "sqrt", "square", "clip", "sign"],
    "binary": ["add", "sub", "mul", "div", "ratio", "diff_ratio"],
    "ternary": ["mul3", "add_mul", "sub_div", "mean3", "weighted_sum"],
}

TEMPLATE_BASE: Dict[str, list] = {
    "unary": [
        "x",
        "np.abs(x)",
        "np.log1p(np.abs(x))",
        "np.sqrt(np.abs(x) + 1e-6)",
        "x ** 2",
        "np.sqrt(np.abs(x))",
        "np.sign(x) * np.log1p(np.abs(x))",
        "x / (np.abs(x).mean() + 1e-6)",
    ],
    "binary": [
        "x + y",
        "x - y",
        "(x + y) / 2",
        "x * y",
        "x / (y + 1e-6)",
        "y / (x + 1e-6)",
        "(x - y) / (np.abs(x) + np.abs(y) + 1e-6)",
        "np.log1p(np.abs(x * y))",
        "x / (np.abs(y).mean() + 1e-6)",
        "y / (np.abs(x).mean() + 1e-6)",
    ],
    "ternary": [
        "x * y * z",
        "(x + y) * z",
        "(x - y) / (z + 1e-6)",
        "(x + y + z) / 3",
        "(x * y) / (z + 1e-6)",
        "(x + y) / (np.abs(z) + 1e-6)",
    ],
}


def paper_config() -> Dict[str, Any]:
    return {
        "P": 500,
        "E": 100,
        "B": 100,
        "T": 50,
        "N": 10,
        "alpha": 0.05,
        "rho_max": 0.9,
        "tau": 0.5,
        "epsilon": 1e-4,
        "kfold": 10,
        "use_llm_semantic_graph": True,
    }


def smoke_config() -> Dict[str, Any]:
    return {
        "P": 40,
        "E": 100,
        "B": 35,
        "T": 18,
        "N": 6,
        "alpha": 0.05,
        "rho_max": 0.9,
        "tau": 0.5,
        "epsilon": 1e-4,
        "kfold": 3,
        "use_llm_semantic_graph": False,
        "semantic_explain_k": 0,
    }
