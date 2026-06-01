"""
analysis/buckets.py
-------------------
Classify each simulation run into an outcome bucket for comparison against
the paper's distribution (Bookstaber-Paddrik-Tivnan):

    no_default  : no HF or BD defaulted by the end of the run
    hf0_only    : only HF0 defaulted, no other HF, no BD
    all_default : every HF and every BD defaulted
    partial     : anything else (some entities defaulted but not all)

Also reports qDemand flags (did each entity ever fire-sell during the run)
and the step of first default per entity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def classify_run(history: list[dict]) -> dict:
    """
    Classify a single run.

    Parameters
    ----------
    history : list of per-step snapshot dicts produced by Simulation.run().

    Returns
    -------
    dict with keys:
        qdemand_hf      : list[bool]  — did HF n ever fire-sell?
        qdemand_bd      : list[bool]  — did BD k ever fire-sell?
        default_hf      : list[bool]  — HF n inactive at final step
        default_bd      : list[bool]  — BD k inactive at final step
        default_step_hf : list[int|None] — step of first capital ≤ 0
        default_step_bd : list[int|None]
        bucket          : one of {"no_default", "hf0_only", "all_default", "partial"}
    """
    final = history[-1]
    n_hf = len(final["hf_active"])
    n_bd = len(final["bd_active"])

    qdemand_hf = [
        any(snap["hf_in_fire_sale"][n] for snap in history) for n in range(n_hf)
    ]
    qdemand_bd = [
        any(snap["bd_in_fire_sale"][k] for snap in history) for k in range(n_bd)
    ]

    default_hf = [not final["hf_active"][n] for n in range(n_hf)]
    default_bd = [not final["bd_active"][k] for k in range(n_bd)]

    def first_step(values):
        for i, v in enumerate(values):
            if v <= 0:
                return i
        return None

    default_step_hf = [
        first_step([snap["hf_capitals"][n] for snap in history]) for n in range(n_hf)
    ]
    default_step_bd = [
        first_step([snap["bd_capitals"][k] for snap in history]) for k in range(n_bd)
    ]

    any_hf = any(default_hf)
    any_bd = any(default_bd)
    all_hf = all(default_hf)
    all_bd = all(default_bd)

    if not any_hf and not any_bd:
        bucket = "no_default"
    elif all_hf and all_bd:
        bucket = "all_default"
    elif (default_hf[0] and sum(default_hf) == 1 and not any_bd
          and not any(qdemand_hf[1:]) and not any(qdemand_bd)):
        # Paper's strict definition: HF0 fire-sold and defaulted; no other HF
        # or BD even had a qDemand event. Looser "only HF0 defaulted" runs
        # where HF1 briefly fire-sold and survived now fall into `partial`.
        bucket = "hf0_only"
    else:
        bucket = "partial"

    return {
        "qdemand_hf": qdemand_hf,
        "qdemand_bd": qdemand_bd,
        "default_hf": default_hf,
        "default_bd": default_bd,
        "default_step_hf": default_step_hf,
        "default_step_bd": default_step_bd,
        "bucket": bucket,
    }


def summarize_runs(histories: list[list[dict]], shock_size: float) -> pd.DataFrame:
    """
    Apply classify_run to every history and return a flat DataFrame —
    one row per run, with bucket label and per-entity columns.
    """
    rows = []
    for r, h in enumerate(histories):
        c = classify_run(h)
        n_hf = len(c["qdemand_hf"])
        n_bd = len(c["qdemand_bd"])
        row = {"run": r, "shock_size": shock_size, "bucket": c["bucket"]}
        for n in range(n_hf):
            row[f"qdemand_hf{n}"] = c["qdemand_hf"][n]
            row[f"default_hf{n}"] = c["default_hf"][n]
            row[f"default_step_hf{n}"] = c["default_step_hf"][n]
        for k in range(n_bd):
            row[f"qdemand_bd{k}"] = c["qdemand_bd"][k]
            row[f"default_bd{k}"] = c["default_bd"][k]
            row[f"default_step_bd{k}"] = c["default_step_bd"][k]
        rows.append(row)
    return pd.DataFrame(rows)


def bucket_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return counts for each canonical bucket (zero-filled if missing)."""
    canonical = ["no_default", "hf0_only", "partial", "all_default"]
    counts = df["bucket"].value_counts().to_dict()
    return {b: int(counts.get(b, 0)) for b in canonical}
