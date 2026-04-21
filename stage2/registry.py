"""Registered Stage-2 datasets (task, folder name, augmented CSV stem)."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import List

from stage2.paths import project_root


@dataclass(frozen=True)
class Stage2Dataset:
    task: str
    path_segment: str
    dataset_id: str


def all_stage2_datasets() -> List[Stage2Dataset]:
    return [
        Stage2Dataset("classification", "cc1", "cc1_augmented"),
        Stage2Dataset("classification", "cc2", "cc2_augmented"),
        Stage2Dataset("classification", "cd1", "cd1_augmented"),
        Stage2Dataset("classification", "cd2", "cd2_augmented"),
        Stage2Dataset("classification", "cf1", "cf1_augmented"),
        Stage2Dataset("classification", "ld1", "ld1_augmented"),
        Stage2Dataset("multiclassification", "balance-scale", "balance-scale_augmented"),
        Stage2Dataset("multiclassification", "cmc", "cmc_augmented"),
        Stage2Dataset("multiclassification", "eucalyptus", "eucalyptus_augmented"),
        Stage2Dataset("multiclassification", "jungle_chess", "jungle_chess_augmented"),
        Stage2Dataset("regression", "boston", "boston_augmented"),
        Stage2Dataset("regression", "california", "california_augmented"),
        Stage2Dataset("regression", "concrete", "concrete_augmented"),
        Stage2Dataset("regression", "insurance", "insurance_augmented"),
        Stage2Dataset("regression", "winequality", "winequality_augmented"),
        Stage2Dataset("cluster", "breast", "breast_augmented"),
        Stage2Dataset("cluster", "glass", "glass_augmented"),
        Stage2Dataset("cluster", "iris", "iris_augmented"),
        Stage2Dataset("cluster", "seeds", "seeds_augmented"),
        Stage2Dataset("cluster", "students", "students_augmented"),
    ]


def dataset_base_dir(entry: Stage2Dataset) -> pathlib.Path:
    return project_root() / entry.task / entry.path_segment
