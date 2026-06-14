import json
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2:latest"
STREAM = False

ACTIVE_CAUSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
FAILURE_MODES = ["F1", "F2", "F3", "F4", "F5", "F6"]

CAUSE_DESC = {
    "C1": "Terrain irregularities",
    "C2": "Slipping hazards",
    "C3": "Communication failure",
    "C4": "Dynamic obstacles",
    "C5": "Static obstacles",
    "C6": "Amplified hazard zones",
}

MODE_DESC = {
    "F1": "Veers Off",
    "F2": "Gets Stuck",
    "F3": "Rollover",
    "F4": "Stops Functioning",
    "F5": "Deadlock",
    "F6": "Path Conflict",
}

# ============================================================
# FAILURE EFFECTS (TABLE b)
# ============================================================
FAILURE_EFFECTS = [f"E{i}" for i in range(1, 9)]

FAILURE_EFFECT_DESC = {
    "E1": "Task completed with minor delay and no robot damage",
    "E2": "Task completed, but minor repair is required",
    "E3": "Task completed, but the robot becomes unrecoverable",
    "E4": "Partial coverage loss with no robot damage",
    "E5": "Partial coverage achieved, but the robot is damaged",
    "E6": "Entire task aborted and the robot is destroyed",
    "E7": "Logic failure causes task abortion, while hardware remains intact",
    "E8": "Human intervention is required; otherwise the task fails",
}

FAILURE_EFFECT_SEVERITY = {
    "E1": 2,
    "E2": 4,
    "E3": 9,
    "E4": 5,
    "E5": 6,
    "E6": 10,
    "E7": 7,
    "E8": 8,
}

# Base prior table P(E_k | F_j)
# rows = E1..E8, columns = F1..F6
BASE_EFFECT_GIVEN_MODE = pd.DataFrame(
    [
        [0.50, 0.20, 0.00, 0.00, 0.00, 0.10],  # E1
        [0.00, 0.25, 0.00, 0.00, 0.00, 0.15],  # E2
        [0.00, 0.00, 0.00, 0.05, 0.00, 0.05],  # E3
        [0.30, 0.05, 0.00, 0.00, 0.00, 0.00],  # E4
        [0.00, 0.15, 0.00, 0.00, 0.05, 0.25],  # E5
        [0.00, 0.05, 0.70, 0.00, 0.20, 0.25],  # E6
        [0.00, 0.00, 0.00, 0.75, 0.70, 0.10],  # E7
        [0.20, 0.30, 0.30, 0.20, 0.05, 0.10],  # E8
    ],
    index=FAILURE_EFFECTS,
    columns=FAILURE_MODES,
)

# ============================================================
# DYNAMIC EFFECT UPDATE SETTINGS
# ============================================================
EFFECT_PRIOR_STRENGTH = 8.0
EFFECT_EVIDENCE_SCALE = 6.0
EFFECT_BETA = 1.2

_EFFECT_SEV = np.array([FAILURE_EFFECT_SEVERITY[e] for e in FAILURE_EFFECTS], dtype=float)
EFFECT_SEV_NORM = (_EFFECT_SEV - _EFFECT_SEV.min()) / (_EFFECT_SEV.max() - _EFFECT_SEV.min() + 1e-9)

MODE_RISK_WEIGHT = {
    "F1": 0.35,   # veers off
    "F2": 0.45,   # gets stuck
    "F3": 1.00,   # rollover
    "F4": 0.85,   # stops functioning
    "F5": 0.80,   # deadlock
    "F6": 0.40,   # path conflict
}

# ============================================================
# PREVIOUS FMEA TABLE
# ============================================================
PREV_PROBS = {
    "C1": [0.40, 0.45, 0.05, 0.00, 0.00, 0.10],
    "C2": [0.35, 0.50, 0.00, 0.00, 0.10, 0.05],
    "C3": [0.00, 0.00, 0.00, 0.70, 0.20, 0.10],
    "C4": [0.15, 0.50, 0.00, 0.00, 0.00, 0.35],
    "C5": [0.10, 0.60, 0.00, 0.00, 0.10, 0.20],
    "C6": [0.05, 0.45, 0.10, 0.00, 0.40, 0.00],
}

PREV_DETECTABILITY = {
    "C1": 7,
    "C2": 8,
    "C3": 2,
    "C4": 3,
    "C5": 2,
    "C6": 8,
}

# ============================================================
# MULTI-LEVEL HAZARD WEIGHTS
# 0 = absent
# 1 = mild
# 2 = moderate/severe
# 3 = critical
# ============================================================
LEVEL_WEIGHTS = {
    "C1": {0: 0.0, 1: 1.0, 2: 1.8, 3: 2.8},  # terrain
    "C2": {0: 0.0, 1: 1.0, 2: 1.7, 3: 2.5},  # slip
    "C3": {0: 0.0, 1: 1.0, 2: 1.6, 3: 2.2},  # comm shadow
    "C4": {0: 0.0, 1: 1.0, 2: 1.5, 3: 2.0},  # dynamic obstacle intensity
    "C5": {0: 0.0, 1: 1.0, 2: 1.4, 3: 1.8},  # static blockage severity
    "C6": {0: 0.0, 1: 1.2, 2: 2.0, 3: 3.0},  # amplified hazard
}

LEVEL_LABELS = {
    "C1": {1: "mild rough terrain", 2: "deep ruts / severe irregularity", 3: "pit-like terrain"},
    "C2": {1: "damp / mildly slippery", 2: "very slippery", 3: "oil/mud-like extreme slip"},
    "C3": {1: "weak communication shadow", 2: "strong communication shadow", 3: "near-blackout zone"},
    "C4": {1: "occasional moving obstacle", 2: "frequent moving obstacle", 3: "highly congested dynamic zone"},
    "C5": {1: "minor barrier", 2: "tight blockage", 3: "nearly impassable bottleneck"},
    "C6": {1: "elevated hazard", 2: "severe hazard", 3: "critical hazard zone"},
}

# ============================================================
# PRIOR STRENGTHS AND EVIDENCE WEIGHTS
# ============================================================
PRIOR_STRENGTH = {
    "C1": 5.0,
    "C2": 5.0,
    "C3": 4.5,
    "C4": 4.5,
    "C5": 5.0,
    "C6": 4.0,
}

EVIDENCE_SCALE = 8.0
LLM_SHIFT_WEIGHT = 0.35

EPS = 1e-6
DETECTABILITY_BETA = 0.65
OBSERVABILITY_KAPPA = 2.0


CAUSE_BASE_TARGET = {
    "C1": np.array([0.30, 0.22, 0.28, 0.00, 0.04, 0.16]),
    "C2": np.array([0.18, 0.34, 0.02, 0.00, 0.26, 0.20]),
    "C3": np.array([0.00, 0.02, 0.00, 0.52, 0.28, 0.18]),
    "C4": np.array([0.16, 0.22, 0.00, 0.00, 0.04, 0.58]),
    "C5": np.array([0.10, 0.40, 0.00, 0.00, 0.18, 0.32]),
    "C6": np.array([0.05, 0.24, 0.40, 0.00, 0.27, 0.04]),
}

CAUSE_L2_BOOST = {
    "C1": np.array([0.00, 0.00, 0.08, 0.00, 0.00, 0.02]),
    "C2": np.array([0.00, 0.06, 0.00, 0.00, 0.05, 0.04]),
    "C3": np.array([0.00, 0.00, 0.00, 0.05, 0.04, 0.03]),
    "C4": np.array([0.02, 0.03, 0.00, 0.00, 0.00, 0.10]),
    "C5": np.array([0.00, 0.05, 0.00, 0.00, 0.03, 0.05]),
    "C6": np.array([0.00, 0.04, 0.12, 0.00, 0.06, 0.00]),
}

CAUSE_L3_BOOST = {
    "C1": np.array([0.00, 0.02, 0.20, 0.00, 0.00, 0.05]),
    "C2": np.array([0.00, 0.08, 0.00, 0.00, 0.08, 0.06]),
    "C3": np.array([0.00, 0.00, 0.00, 0.12, 0.08, 0.06]),
    "C4": np.array([0.04, 0.06, 0.00, 0.00, 0.02, 0.20]),
    "C5": np.array([0.00, 0.08, 0.00, 0.00, 0.05, 0.08]),
    "C6": np.array([0.00, 0.06, 0.22, 0.00, 0.10, 0.00]),
}

CAUSE_ANOM_BOOST = {
    "C1": np.array([0.05, 0.04, 0.10, 0.00, 0.00, 0.03]),
    "C2": np.array([0.02, 0.08, 0.00, 0.00, 0.08, 0.05]),
    "C3": np.array([0.00, 0.00, 0.00, 0.09, 0.06, 0.04]),
    "C4": np.array([0.03, 0.04, 0.00, 0.00, 0.02, 0.12]),
    "C5": np.array([0.02, 0.08, 0.00, 0.00, 0.04, 0.05]),
    "C6": np.array([0.01, 0.05, 0.12, 0.00, 0.08, 0.00]),
}
# ============================================================
# MOVEMENT / SENSOR TO CAUSE ANOMALY WEIGHTS
# ============================================================
EVENT_WEIGHTS = {
    "C1": {
        "lateral_deviation": 1.1,
        "tilt_warning": 1.8,
        "rough_progress_loss": 1.0,
    },
    "C2": {
        "wheel_slip": 1.6,
        "stall_steps": 0.35,
        "traction_recovery": 0.7,
    },
    "C3": {
        "comm_drop": 1.8,
        "heartbeat_timeout": 1.6,
        "reconnect_attempt": 0.7,
    },
    "C4": {
        "dynamic_replan": 1.4,
        "near_collision": 1.8,
        "avoidance_brake": 1.2,
    },
    "C5": {
        "blocked_turnback": 1.3,
        "clearance_warning": 1.0,
        "wall_following_detour": 0.8,
    },
    "C6": {
        "risk_spike": 1.4,
        "tilt_warning": 1.0,
        "hard_escape_maneuver": 1.5,
    },
}

# ============================================================
# SAMPLE MULTI-LEVEL GRID LAYERS
# Each cause is its own severity map, allowing overlaps.
# ============================================================
def make_empty_layers(h=12, w=12):
    return {c: np.zeros((h, w), dtype=int) for c in ACTIVE_CAUSES}

def assign(layer, cells, level):
    for r, c in cells:
        layer[r, c] = level

def make_previous_layers():
    layers = make_empty_layers()

    # C1 terrain irregularities: level 1 and level 2
    assign(layers["C1"], [(2,8), (2,9), (3,8), (3,9)], 1)
    assign(layers["C1"], [(4,8), (4,9)], 2)

    # C2 slip
    assign(layers["C2"], [(7,5), (7,6), (8,5), (8,6)], 1)

    # C3 comm shadow
    assign(layers["C3"], [(0,9), (0,10), (1,9), (1,10)], 1)

    # C4 dynamic obstacles
    assign(layers["C4"], [(5,7), (6,7), (7,7)], 1)

    # C5 static obstacles / bottlenecks
    assign(layers["C5"], [(1,1), (1,2), (1,3), (1,4), (2,4), (3,4)], 1)
    assign(layers["C5"], [(8,7), (8,8), (8,9), (9,9)], 2)

    # C6 amplified hazard absent initially
    return layers

def make_current_layers():
    layers = make_previous_layers()

    # C1 terrain becomes more severe in some cells
    assign(layers["C1"], [(5,8), (5,9)], 2)
    assign(layers["C1"], [(6,8)], 3)  # new pit-like terrain

    # C2 slip expands and includes a more dangerous patch
    assign(layers["C2"], [(6,5), (6,6), (9,5), (9,6)], 1)
    assign(layers["C2"], [(8,4), (9,4)], 2)

    # C3 communication shadow expands and deepens
    assign(layers["C3"], [(2,10), (3,10), (4,10)], 1)
    assign(layers["C3"], [(5,10), (6,10)], 2)

    # C4 more dynamic obstacles / congestion
    assign(layers["C4"], [(4,6), (5,6)], 1)
    assign(layers["C4"], [(6,9), (7,9), (7,10)], 2)

    # C5 static bottleneck slightly worsens
    assign(layers["C5"], [(6,2), (7,2)], 1)
    assign(layers["C5"], [(10,9)], 2)

    # C6 new amplified hazard region
    assign(layers["C6"], [(9,3), (9,4)], 2)
    assign(layers["C6"], [(10,3), (10,4)], 3)

    return layers

def make_layers_at_step(step):
    layers = make_previous_layers()

    if step >= 5:
        assign(layers["C1"], [(5, 8)], 1)
        assign(layers["C2"], [(6, 5)], 1)
        assign(layers["C4"], [(4, 6)], 1)

    if step >= 10:
        assign(layers["C1"], [(5, 8), (5, 9)], 1)
        assign(layers["C2"], [(6, 5), (6, 6)], 1)
        assign(layers["C4"], [(4, 6), (5, 6)], 1)
        assign(layers["C5"], [(6, 2)], 1)

    if step >= 15:
        assign(layers["C1"], [(5, 8), (5, 9)], 2)
        assign(layers["C2"], [(8, 4)], 2)
        assign(layers["C3"], [(2, 10)], 1)
        assign(layers["C5"], [(7, 2)], 1)

    if step >= 20:
        assign(layers["C2"], [(8, 4), (9, 4)], 2)
        assign(layers["C3"], [(2, 10), (3, 10)], 1)
        assign(layers["C4"], [(5, 6), (6, 9)], 1)

    if step >= 25:
        assign(layers["C1"], [(6, 8)], 2)
        assign(layers["C2"], [(9, 5)], 1)
        assign(layers["C3"], [(4, 10)], 1)
        assign(layers["C6"], [(9, 3)], 2)

    if step >= 30:
        assign(layers["C1"], [(6, 8)], 3)
        assign(layers["C2"], [(9, 5), (9, 6)], 1)
        assign(layers["C3"], [(4, 10), (5, 10)], 1)
        assign(layers["C4"], [(7, 9)], 2)
        assign(layers["C5"], [(10, 9)], 2)
        assign(layers["C6"], [(9, 3), (9, 4)], 2)

    if step >= 35:
        assign(layers["C1"], [(6, 9)], 2)
        assign(layers["C2"], [(8, 5)], 2)
        assign(layers["C3"], [(5, 10)], 2)
        assign(layers["C4"], [(7, 10)], 2)
        assign(layers["C6"], [(10, 3)], 2)

    if step >= 40:
        assign(layers["C2"], [(6, 6), (8, 5)], 2)
        assign(layers["C3"], [(6, 10)], 2)
        assign(layers["C6"], [(10, 3)], 3)

    if step >= 45:
        assign(layers["C1"], [(7, 8)], 2)
        assign(layers["C2"], [(7, 5)], 2)
        assign(layers["C3"], [(7, 10)], 1)
        assign(layers["C4"], [(6, 7)], 2)
        assign(layers["C5"], [(8, 9)], 2)
        assign(layers["C6"], [(10, 4)], 2)

    if step >= 50:
        assign(layers["C1"], [(7, 8)], 3)
        assign(layers["C3"], [(7, 10)], 2)
        assign(layers["C4"], [(6, 7), (7, 7)], 2)
        assign(layers["C5"], [(8, 9), (9, 9)], 2)
        assign(layers["C6"], [(10, 4)], 3)

    return layers


def get_movement_summary_at_step(step):
    progress = max(0.0, min(step / max(UPDATE_STEPS), 1.0))
    factor = 0.35 + 1.45 * progress

    summary = {
        k: max(0, int(round(v * factor)))
        for k, v in BASE_MOVEMENT_SUMMARY.items()
    }

    spike_terms = {
        "comm_drop": int(round(5 * progress)),
        "risk_spike": int(round(7 * progress)),
        "tilt_warning": int(round(4 * progress)),
        "near_collision": int(round(4 * progress)),
        "hard_escape_maneuver": int(round(3 * progress)),
    }

    for k, inc in spike_terms.items():
        summary[k] = summary.get(k, 0) + inc

    return summary


def get_sensor_observability_at_step(step):
    progress = max(0.0, min(step / max(UPDATE_STEPS), 1.0))
    base = {
        "C1": {"direct_detections": 1, "exposures": 5},
        "C2": {"direct_detections": 1, "exposures": 5},
        "C3": {"direct_detections": 3, "exposures": 5},
        "C4": {"direct_detections": 4, "exposures": 6},
        "C5": {"direct_detections": 5, "exposures": 6},
        "C6": {"direct_detections": 0, "exposures": 2},
    }
    terminal = {
        "C1": {"direct_detections": 5, "exposures": 15},
        "C2": {"direct_detections": 4, "exposures": 14},
        "C3": {"direct_detections": 8, "exposures": 12},
        "C4": {"direct_detections": 9, "exposures": 13},
        "C5": {"direct_detections": 10, "exposures": 13},
        "C6": {"direct_detections": 2, "exposures": 9},
    }

    sensor_profiles = {}
    for cause in ACTIVE_CAUSES:
        sensor_profiles[cause] = {
            "direct_detections": int(round(base[cause]["direct_detections"] + progress * (terminal[cause]["direct_detections"] - base[cause]["direct_detections"]))),
            "exposures": int(round(base[cause]["exposures"] + progress * (terminal[cause]["exposures"] - base[cause]["exposures"]))),
        }

    return sensor_profiles


def get_state_at_step(step):
    return (
        make_layers_at_step(step),
        get_movement_summary_at_step(step),
        get_sensor_observability_at_step(step),
    )


def prob_df_to_dict(prob_df):
    return {
        c: prob_df.loc[c, FAILURE_MODES].to_numpy(dtype=float)
        for c in prob_df.index
    }


def detectability_df_to_dict(evidence_df):
    return {
        row["Cause"]: int(row["Updated D"])
        for _, row in evidence_df.iterrows()
    }


def mean_abs_df_delta(df_a, df_b):
    a = df_a.loc[df_b.index, df_b.columns].to_numpy(dtype=float)
    b = df_b.to_numpy(dtype=float)
    return float(np.mean(np.abs(a - b)))


# ============================================================
# MOVEMENT / SENSOR SUMMARY
# ============================================================
BASE_MOVEMENT_SUMMARY = {
    "lateral_deviation": 5,
    "tilt_warning": 2,
    "rough_progress_loss": 4,
    "wheel_slip": 6,
    "stall_steps": 8,
    "traction_recovery": 3,
    "comm_drop": 2,
    "heartbeat_timeout": 2,
    "reconnect_attempt": 3,
    "dynamic_replan": 4,
    "near_collision": 3,
    "avoidance_brake": 5,
    "blocked_turnback": 3,
    "clearance_warning": 4,
    "wall_following_detour": 2,
    "risk_spike": 2,
    "hard_escape_maneuver": 1,
}

BASE_SENSOR_OBSERVABILITY = {
    "C1": {"direct_detections": 3, "exposures": 11},
    "C2": {"direct_detections": 2, "exposures": 10},
    "C3": {"direct_detections": 6, "exposures": 8},
    "C4": {"direct_detections": 7, "exposures": 9},
    "C5": {"direct_detections": 8, "exposures": 9},
    "C6": {"direct_detections": 1, "exposures": 5},
}

UPDATE_STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

BASE_UPDATE_PROFILE = {
    "profile_name": "base",
    "prior_strength_scale": 1.00,
    "evidence_scale": EVIDENCE_SCALE,
    "llm_shift_weight": 0.00,
    "gamma_floor": 0.08,
    "gamma_cap": 0.85,
    "semantic_gamma_boost": 0.00,
    "semantic_evidence_boost": 0.00,
    "detectability_beta": DETECTABILITY_BETA,
    "effect_prior_strength": EFFECT_PRIOR_STRENGTH,
    "effect_evidence_scale": EFFECT_EVIDENCE_SCALE,
    "effect_beta": EFFECT_BETA,
    "effect_bias_weight": 0.35,
    "effect_semantic_mass_boost": 0.00,
}

LLM_UPDATE_PROFILE = {
    # Stronger but still evidence-gated profile.
    # The LLM can separate the dynamic+LLM method only when map growth,
    # anomaly, severity level, and observability support the update.
    "profile_name": "llm",
    "prior_strength_scale": 0.52,
    "evidence_scale": 11.5,
    "llm_shift_weight": 1.35,

    "gamma_floor": 0.14,
    "gamma_cap": 0.96,

    "semantic_gamma_boost": 0.38,
    "semantic_evidence_boost": 0.65,

    "detectability_beta": 0.82,

    "effect_prior_strength": 3.2,
    "effect_evidence_scale": 13.0,
    "effect_beta": 2.6,
    "effect_bias_weight": 0.90,
    "effect_semantic_mass_boost": 0.75,
}

# ============================================================
# GRID SUMMARY
# ============================================================
def cause_level_counts(layer):
    vals, cnts = np.unique(layer, return_counts=True)
    result = {int(v): int(c) for v, c in zip(vals, cnts)}
    return {k: result.get(k, 0) for k in [0, 1, 2, 3]}

def weighted_extent(layer, cause):
    counts = cause_level_counts(layer)
    return sum(LEVEL_WEIGHTS[cause][lvl] * counts[lvl] for lvl in [1, 2, 3])

def summarize_layers(layers):
    summary = {}
    for c in ACTIVE_CAUSES:
        counts = cause_level_counts(layers[c])
        summary[c] = {
            "counts": counts,
            "weighted_extent": weighted_extent(layers[c], c),
        }
    return summary

def anomaly_score_for_cause(cause, movement_summary):
    weights = EVENT_WEIGHTS[cause]
    return sum(weights.get(k, 0.0) * movement_summary.get(k, 0) for k in movement_summary)

# ============================================================
# LLM SCHEMA
# ============================================================
def build_schema():
    cause_obj = {
        "type": "object",
        "properties": {
            "severity_multiplier": {"type": "number"},
            "detectability_delta": {"type": "integer"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
            "mode_shift": {
                "type": "object",
                "properties": {fm: {"type": "number"} for fm in FAILURE_MODES},
                "required": FAILURE_MODES,
                "additionalProperties": False,
            },
        },
        "required": [
            "severity_multiplier",
            "detectability_delta",
            "confidence",
            "reason",
            "mode_shift",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "causes": {
                "type": "object",
                "properties": {c: cause_obj for c in ACTIVE_CAUSES},
                "required": ACTIVE_CAUSES,
                "additionalProperties": False,
            }
        },
        "required": ["causes"],
        "additionalProperties": False,
    }

# ============================================================
# LLM PROMPT
# ============================================================
def build_prompt(prev_layers, curr_layers, movement_summary, sensor_observability, prev_detectability, checkpoint_step=None):
    prev_summary = summarize_layers(prev_layers)
    curr_summary = summarize_layers(curr_layers)

    compact = {}
    for c in ACTIVE_CAUSES:
        counts = cause_level_counts(curr_layers[c])
        prev_w = float(prev_summary[c]["weighted_extent"])
        curr_w = float(curr_summary[c]["weighted_extent"])
        growth = max(0.0, curr_w - prev_w)

        obs = sensor_observability[c]
        obs_ratio = obs["direct_detections"] / max(obs["exposures"], 1)

        compact[c] = {
            "prev_w": round(prev_w, 2),
            "curr_w": round(curr_w, 2),
            "growth": round(growth, 2),
            "l2": int(counts[2]),
            "l3": int(counts[3]),
            "anom": round(float(anomaly_score_for_cause(c, movement_summary)), 2),
            "obs": round(obs_ratio, 2),
            "prev_D": int(prev_detectability[c]),
        }

    checkpoint_text = f"Checkpoint step: {checkpoint_step}.\n" if checkpoint_step is not None else ""

    prompt = f"""
Return ONLY valid JSON in one line.

{checkpoint_text}Task:
For each cause C1..C6, output:
- severity_multiplier in [0.75,1.85]
- detectability_delta in [-1,0,1]
- confidence in [0,1]
- reason with at most 4 words
- mode_shift for F1..F6, each in [-0.60,0.60]

Interpretation:
- positive mode_shift means push probability toward that failure mode
- negative mode_shift means suppress that failure mode
- use stronger shifts when l3, growth, and anomaly are all high

Rules:
- At least 3 causes should be non-neutral when evidence is strong.
- If l3 > 0 for C1 or C6, severity_multiplier should usually be >= 1.15.
- If growth > 5, severity_multiplier should usually be > 1.08.
- If obs >= 0.70, detectability_delta should usually be -1.
- If obs <= 0.30 and hazard is strong, detectability_delta can be 1.
- Do not return all-zero mode_shift vectors unless evidence is weak.

Return exactly this shape:
{{"causes":{{
"C1":{{"severity_multiplier":1.15,"detectability_delta":0,"confidence":0.80,"reason":"rough l3 growth","mode_shift":{{"F1":0.04,"F2":0.02,"F3":0.15,"F4":0.00,"F5":0.00,"F6":-0.01}}}},
"C2":{{"severity_multiplier":1.05,"detectability_delta":0,"confidence":0.60,"reason":"moderate slip","mode_shift":{{"F1":0.00,"F2":0.08,"F3":0.00,"F4":0.00,"F5":0.04,"F6":0.02}}}},
"C3":{{"severity_multiplier":1.12,"detectability_delta":1,"confidence":0.75,"reason":"shadow growth","mode_shift":{{"F1":0.00,"F2":0.00,"F3":0.00,"F4":0.10,"F5":0.06,"F6":0.03}}}},
"C4":{{"severity_multiplier":1.08,"detectability_delta":0,"confidence":0.70,"reason":"dynamic congestion","mode_shift":{{"F1":0.03,"F2":0.02,"F3":0.00,"F4":0.00,"F5":0.00,"F6":0.10}}}},
"C5":{{"severity_multiplier":1.03,"detectability_delta":-1,"confidence":0.65,"reason":"visible bottleneck","mode_shift":{{"F1":0.00,"F2":0.06,"F3":0.00,"F4":0.00,"F5":0.03,"F6":0.04}}}},
"C6":{{"severity_multiplier":1.22,"detectability_delta":1,"confidence":0.85,"reason":"critical l3 zone","mode_shift":{{"F1":0.00,"F2":0.04,"F3":0.18,"F4":0.00,"F5":0.08,"F6":0.00}}}}
}}}}

Evidence:
{json.dumps(compact, separators=(",", ":"))}
"""
    return prompt



def build_effect_prompt(prob_df, evidence_df):
    """
    Builds a compact prompt for updating P(E_k | F_j).
    """
    mode_mass = {fm: round(float(prob_df[fm].mean()), 4) for fm in FAILURE_MODES}

    curr_norm = float(
        evidence_df["Curr weighted extent"].mean()
        / max(evidence_df["Curr weighted extent"].max(), EPS)
    )
    growth_norm = float(
        evidence_df["Positive growth"].mean()
        / max(evidence_df["Positive growth"].max(), EPS)
    )
    anom_norm = float(
        evidence_df["Movement anomaly"].mean()
        / max(evidence_df["Movement anomaly"].max(), EPS)
    )
    l3_total = int(evidence_df["L3 cells"].sum())
    l23_total = int((evidence_df["L2 cells"] + evidence_df["L3 cells"]).sum())
    l3_ratio = float(l3_total / max(l23_total, 1))

    compact = {
        "mode_mass": mode_mass,
        "global_context": {
            "curr_norm": round(curr_norm, 3),
            "growth_norm": round(growth_norm, 3),
            "anom_norm": round(anom_norm, 3),
            "l3_ratio": round(l3_ratio, 3),
        },
        "base_effect_table_columns": {
            fm: {
                e: round(float(BASE_EFFECT_GIVEN_MODE.loc[e, fm]), 2)
                for e in FAILURE_EFFECTS
            }
            for fm in FAILURE_MODES
        },
    }

    prompt = f"""
Return ONLY valid compact JSON in one line.

Task:
For each failure mode F1..F6, output:
- severity_multiplier in [0.75,1.85]
- escalation_bias in [-1,0,1]
- confidence in [0,1]
- reason with at most 4 words

Interpretation:
- escalation_bias = 1 means shift toward more severe effects
- escalation_bias = 0 means keep near prior table
- escalation_bias = -1 means shift toward less severe/recoverable effects

Rules:
1. Do not rewrite the whole effect table directly.
2. Use the current mode_mass and global_context.
3. If F3 or F5 is strong and context is severe, escalation_bias should usually be 1.
4. If global l3_ratio is high, catastrophic effects should be more plausible.
5. Keep outputs short, but allow clearly stronger escalation under severe context.
6. Do not return all-neutral values unless the context is truly weak.

Return exactly this structure:
{{"modes":{{"F1":{{"severity_multiplier":1.02,"escalation_bias":0,"confidence":0.55,"reason":"minor drift"}},"F2":{{"severity_multiplier":1.08,"escalation_bias":0,"confidence":0.60,"reason":"stall risk"}},"F3":{{"severity_multiplier":1.20,"escalation_bias":1,"confidence":0.80,"reason":"severe rollover"}},"F4":{{"severity_multiplier":1.10,"escalation_bias":1,"confidence":0.70,"reason":"task abort"}},"F5":{{"severity_multiplier":1.14,"escalation_bias":1,"confidence":0.75,"reason":"abort danger"}},"F6":{{"severity_multiplier":1.04,"escalation_bias":0,"confidence":0.58,"reason":"conflict mild"}}}}}}

Evidence:
{json.dumps(compact, separators=(",", ":"))}
"""
    return prompt
# ============================================================
# OLLAMA CALL
# ============================================================

def is_neutral_llm_output(raw):
    """
    Robust neutral-output checker.
    Handles malformed LLM outputs safely, including:
    - list outputs
    - {"causes": list}
    - missing causes
    - missing mode_shift
    """

    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            raw = raw[0]
        else:
            return True

    if not isinstance(raw, dict):
        return True

    causes_obj = raw.get("causes", None)

    if not isinstance(causes_obj, dict):
        return True

    neutral_count = 0
    total = 0

    for c in ACTIVE_CAUSES:
        item = causes_obj.get(c, {})

        if not isinstance(item, dict):
            neutral_count += 1
            total += 1
            continue

        try:
            sev = float(item.get("severity_multiplier", 1.0))
            dd = int(item.get("detectability_delta", 0))
            conf = float(item.get("confidence", 0.5))
            reason = str(item.get("reason", "")).strip().lower()
        except Exception:
            neutral_count += 1
            total += 1
            continue

        cause_name = CAUSE_DESC[c].lower()

        sev_neutral = abs(sev - 1.0) < 1e-9
        dd_neutral = dd == 0
        conf_neutral = abs(conf - 0.5) < 1e-9
        reason_neutral = (
            reason == cause_name
            or reason == ""
            or "short reason" in reason
            or "fallback" in reason
        )

        mode_shift = item.get("mode_shift", {})
        if not isinstance(mode_shift, dict):
            mode_shift_neutral = True
        else:
            try:
                vals = [float(mode_shift.get(fm, 0.0)) for fm in FAILURE_MODES]
                mode_shift_neutral = all(abs(v) < 1e-12 for v in vals)
            except Exception:
                mode_shift_neutral = True

        if sev_neutral and dd_neutral and conf_neutral and reason_neutral and mode_shift_neutral:
            neutral_count += 1

        total += 1

    return neutral_count == total

def is_neutral_effect_llm_output(raw):
    """
    Robust neutral checker for effect-level LLM output.
    Handles malformed outputs safely.
    """

    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            raw = raw[0]
        else:
            return True

    if not isinstance(raw, dict):
        return True

    modes_obj = raw.get("modes", None)

    if not isinstance(modes_obj, dict):
        return True

    neutral_count = 0
    total = 0

    for fm in FAILURE_MODES:
        item = modes_obj.get(fm, {})

        if not isinstance(item, dict):
            neutral_count += 1
            total += 1
            continue

        try:
            sev = float(item.get("severity_multiplier", 1.0))
            bias = int(item.get("escalation_bias", 0))
            conf = float(item.get("confidence", 0.5))
            reason = str(item.get("reason", "")).strip().lower()
        except Exception:
            neutral_count += 1
            total += 1
            continue

        sev_neutral = abs(sev - 1.0) < 1e-9
        bias_neutral = bias == 0
        conf_neutral = abs(conf - 0.5) < 1e-9
        reason_neutral = (
            reason == ""
            or "short" in reason
            or "fallback" in reason
        )

        if sev_neutral and bias_neutral and conf_neutral and reason_neutral:
            neutral_count += 1

        total += 1

    return neutral_count == total

def call_ollama_qwen2(prompt):
    def _post(prompt_text):
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only compact valid JSON. "
                        "No markdown. No extra text. "
                        "Keep reasons very short."
                    )
                },
                {"role": "user", "content": prompt_text},
            ],
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_predict": 500,
            },
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        print("STATUS:", r.status_code)
        print("RAW RESPONSE:")
        print(r.text[:3000])

        r.raise_for_status()
        data = r.json()
        content = data["message"]["content"]
        print("MODEL CONTENT:")
        print(content[:3000])

        return json.loads(content)

    raw = _post(prompt)

    if is_neutral_llm_output(raw):
        stricter_prompt = prompt + """
Return compact JSON only.
Do not use neutral defaults for all causes.
"""
        print("=== Retrying because first LLM output was neutral ===")
        raw = _post(stricter_prompt)

    return raw

def call_ollama_effects(prompt):
    def _post(prompt_text):
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only compact valid JSON. "
                        "No markdown. No extra text."
                    )
                },
                {"role": "user", "content": prompt_text},
            ],
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_predict": 500,
            },
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        print("STATUS (effects):", r.status_code)
        print("RAW RESPONSE (effects):")
        print(r.text[:3000])

        r.raise_for_status()
        data = r.json()
        content = data["message"]["content"]
        print("MODEL CONTENT (effects):")
        print(content[:3000])

        return json.loads(content)

    raw = _post(prompt)

    if is_neutral_effect_llm_output(raw):
        stricter_prompt = prompt + """
Return compact JSON only.
Do not use all-neutral values when context is strong.
"""
        print("=== Retrying effect LLM because first output was neutral ===")
        raw = _post(stricter_prompt)

    return raw
# ============================================================
# VALIDATION + SAFE FALLBACK
# ============================================================
def safe_llm_output(raw):
    """
    Safely converts any malformed cause-level LLM output into the expected format.

    Expected format:
    {
        "causes": {
            "C1": {
                "severity_multiplier": float,
                "detectability_delta": int,
                "confidence": float,
                "reason": str,
                "mode_shift": {"F1": float, ..., "F6": float}
            },
            ...
        }
    }
    """

    fallback = {
        "causes": {
            c: {
                "severity_multiplier": 1.0,
                "detectability_delta": 0,
                "confidence": 0.5,
                "reason": "fallback",
                "mode_shift": {fm: 0.0 for fm in FAILURE_MODES},
            }
            for c in ACTIVE_CAUSES
        }
    }

    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            raw = raw[0]
        else:
            return fallback

    if not isinstance(raw, dict):
        return fallback

    causes_obj = raw.get("causes", None)

    if not isinstance(causes_obj, dict):
        return fallback

    out = deepcopy(fallback)

    for c in ACTIVE_CAUSES:
        item = causes_obj.get(c, {})

        if not isinstance(item, dict):
            continue

        try:
            out["causes"][c]["severity_multiplier"] = float(
                np.clip(float(item.get("severity_multiplier", 1.0)), 0.75, 1.85)
            )
        except Exception:
            out["causes"][c]["severity_multiplier"] = 1.0

        try:
            out["causes"][c]["detectability_delta"] = int(
                np.clip(int(item.get("detectability_delta", 0)), -1, 1)
            )
        except Exception:
            out["causes"][c]["detectability_delta"] = 0

        try:
            out["causes"][c]["confidence"] = float(
                np.clip(float(item.get("confidence", 0.5)), 0.0, 1.0)
            )
        except Exception:
            out["causes"][c]["confidence"] = 0.5

        out["causes"][c]["reason"] = str(item.get("reason", "fallback"))

        raw_mode_shift = item.get("mode_shift", {})

        if not isinstance(raw_mode_shift, dict):
            raw_mode_shift = {}

        for fm in FAILURE_MODES:
            try:
                out["causes"][c]["mode_shift"][fm] = float(
                    np.clip(float(raw_mode_shift.get(fm, 0.0)), -0.60, 0.60)
                )
            except Exception:
                out["causes"][c]["mode_shift"][fm] = 0.0

    return out

def safe_effect_llm_output(raw):
    """
    Safely converts malformed effect-level LLM output into the expected format.

    Expected format:
    {
        "modes": {
            "F1": {
                "severity_multiplier": float,
                "escalation_bias": int,
                "confidence": float,
                "reason": str
            },
            ...
        }
    }
    """

    fallback = {
        "modes": {
            fm: {
                "severity_multiplier": 1.0,
                "escalation_bias": 0,
                "confidence": 0.5,
                "reason": "fallback",
            }
            for fm in FAILURE_MODES
        }
    }

    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            raw = raw[0]
        else:
            return fallback

    if not isinstance(raw, dict):
        return fallback

    modes_obj = raw.get("modes", None)

    if not isinstance(modes_obj, dict):
        return fallback

    out = deepcopy(fallback)

    for fm in FAILURE_MODES:
        item = modes_obj.get(fm, {})

        if not isinstance(item, dict):
            continue

        try:
            out["modes"][fm]["severity_multiplier"] = float(
                np.clip(float(item.get("severity_multiplier", 1.0)), 0.75, 1.85)
            )
        except Exception:
            out["modes"][fm]["severity_multiplier"] = 1.0

        try:
            out["modes"][fm]["escalation_bias"] = int(
                np.clip(int(item.get("escalation_bias", 0)), -1, 1)
            )
        except Exception:
            out["modes"][fm]["escalation_bias"] = 0

        try:
            out["modes"][fm]["confidence"] = float(
                np.clip(float(item.get("confidence", 0.5)), 0.0, 1.0)
            )
        except Exception:
            out["modes"][fm]["confidence"] = 0.5

        out["modes"][fm]["reason"] = str(item.get("reason", "fallback"))

    return out

# ============================================================
# LLM CALIBRATION / EVIDENCE GATING
# ============================================================

def calibrate_cause_llm_output(
    cause,
    sev_mult,
    dD,
    conf,
    shift,
    growth,
    counts,
    anomaly_norm,
    obs_ratio,
    level_pressure,
):
    """
    Evidence-gated calibration for cause-level LLM output.

    Motivation:
    The raw LLM output may be semantically reasonable but too conservative.
    This function allows strong LLM influence only when the environment evidence
    also supports a strong hazard update.

    It prevents:
    - excessive severity inflation,
    - excessive mode-shift amplification,
    - detectability becoming worse when observability is actually high,
    - LLM-induced detours that increase cumulative risk.
    """

    l2 = counts.get(2, 0)
    l3 = counts.get(3, 0)

    growth_strength = float(np.clip(growth / 8.0, 0.0, 1.0))
    level_strength = float(np.clip((l2 + 2.0 * l3) / 6.0, 0.0, 1.0))
    anomaly_strength = float(np.clip(anomaly_norm, 0.0, 1.0))

    evidence_strength = float(np.clip(
        0.35 * growth_strength
        + 0.35 * level_strength
        + 0.20 * anomaly_strength
        + 0.10 * np.clip(level_pressure * 8.0, 0.0, 1.0),
        0.0,
        1.0,
    ))

    if evidence_strength < 0.35:
        conf = min(conf, 0.20)
        sev_mult = 1.0 + 0.20 * (sev_mult - 1.0)
        shift = np.zeros_like(shift)
        dD = 0

    elif evidence_strength < 0.50:
        shift = 0.35 * shift
        conf = min(conf, 0.45)

    # C6 and C1 with L3 evidence deserve stronger semantic influence.
    if cause in ["C1", "C6"] and l3 > 0:
        evidence_strength = float(np.clip(evidence_strength + 0.15, 0.0, 1.0))

    # If evidence is weak, shrink confidence strongly.
    # If evidence is strong, preserve more of the LLM confidence.
    conf_scale = 0.30 + 0.70 * evidence_strength
    conf = float(np.clip(conf * conf_scale, 0.0, 0.85))

    # Severity should not explode unless there is real level/growth evidence.
    sev_deviation = sev_mult - 1.0
    sev_scale = 0.35 + 0.65 * evidence_strength
    sev_mult = 1.0 + sev_deviation * sev_scale

    # Dynamic cap on severity multiplier.
    sev_cap = 1.08 + 0.42 * evidence_strength
    if l3 > 0:
        sev_cap += 0.10

    sev_mult = float(np.clip(sev_mult, 0.85, sev_cap))

    # Mode shift should be very small unless evidence supports it.
    shift_scale = 0.25 + 0.75 * evidence_strength
    shift = np.array(shift, dtype=float) * shift_scale

    shift_cap = 0.04 + 0.18 * evidence_strength
    if l3 > 0:
        shift_cap += 0.06

    shift = np.clip(shift, -shift_cap, shift_cap)

    # Detectability correction:
    # high observability should not make D worse.
    if obs_ratio >= 0.70:
        dD = min(dD, 0)

    # weak evidence should not alter detectability.
    if evidence_strength < 0.25:
        dD = 0

    # strong hidden hazard may worsen detectability.
    if obs_ratio <= 0.30 and evidence_strength > 0.55:
        dD = max(dD, 0)

    dD = int(np.clip(dD, -1, 1))

    return sev_mult, dD, conf, shift, evidence_strength


def calibrate_effect_llm_output(
    fm,
    sev_mult,
    bias,
    conf,
    mode_mass_norm,
    global_context,
):
    """
    Evidence-gated calibration for effect-level LLM output.

    This prevents the LLM from pushing too much probability mass toward
    catastrophic effects unless the current mode mass and global context justify it.
    """

    risk_support = float(np.clip(
        0.60 * mode_mass_norm + 0.40 * global_context,
        0.0,
        1.0,
    ))

    # F3 rollover and F5 deadlock can justify stronger escalation.
    if fm in ["F3", "F5"] and risk_support > 0.45:
        risk_support = float(np.clip(risk_support + 0.10, 0.0, 1.0))

    conf_scale = 0.35 + 0.65 * risk_support
    conf = float(np.clip(conf * conf_scale, 0.0, 0.85))

    sev_mult = 1.0 + (sev_mult - 1.0) * (0.35 + 0.65 * risk_support)

    sev_cap = 1.06 + 0.36 * risk_support
    sev_mult = float(np.clip(sev_mult, 0.85, sev_cap))

    # Do not escalate to catastrophic effects if support is weak.
    if risk_support < 0.45 and bias > 0:
        bias = 0

    # If support is very weak, keep neutral.
    if risk_support < 0.25:
        bias = 0
        conf = min(conf, 0.35)

    bias = int(np.clip(bias, -1, 1))

    return sev_mult, bias, conf, risk_support
# ============================================================
# BAYESIAN UPDATE
# ============================================================
def normalize_dict(d):
    vals = np.array(list(d.values()), dtype=float)
    mx = max(vals.max(), EPS)
    return {k: float(v / mx) for k, v in d.items()}

def normalize_vec(v):
    v = np.array(v, dtype=float)
    v = np.maximum(v, EPS)
    s = v.sum()
    if s <= 0:
        return np.ones_like(v) / len(v)
    return v / s

def evidence_target_for_cause(cause, curr_layers, anomaly_norm):
    counts = cause_level_counts(curr_layers[cause])
    l2 = counts[2]
    l3 = counts[3]

    raw = (
        CAUSE_BASE_TARGET[cause]
        + l2 * CAUSE_L2_BOOST[cause]
        + l3 * CAUSE_L3_BOOST[cause]
        + anomaly_norm * CAUSE_ANOM_BOOST[cause]
    )
    return normalize_vec(raw), counts

def update_fmea(prev_layers, curr_layers, llm_out, prev_probs, prev_detectability, movement_summary, sensor_observability, profile):
    prev_summary = summarize_layers(prev_layers)
    curr_summary = summarize_layers(curr_layers)

    W = {}
    DELTA = {}
    A = {}

    for c in ACTIVE_CAUSES:
        W[c] = curr_summary[c]["weighted_extent"]
        DELTA[c] = max(0.0, curr_summary[c]["weighted_extent"] - prev_summary[c]["weighted_extent"])
        A[c] = anomaly_score_for_cause(c, movement_summary)

    Wn = normalize_dict(W)
    Dn = normalize_dict(DELTA)
    An = normalize_dict(A)
    maxA = max(A.values()) if len(A) > 0 else 1.0

    grid_size = next(iter(curr_layers.values())).size

    post_probs = {}
    post_D = {}
    evidence_rows = []

    for c in ACTIVE_CAUSES:
        prev_row = np.array(prev_probs[c], dtype=float)
        prior = PRIOR_STRENGTH[c] * profile["prior_strength_scale"] * prev_row

        llm_c = llm_out["causes"][c]
        sev_mult = llm_c["severity_multiplier"]
        dD = llm_c["detectability_delta"]
        conf = llm_c["confidence"]
        shift = np.array([llm_c["mode_shift"][fm] for fm in FAILURE_MODES], dtype=float)

        anomaly_norm = A[c] / max(maxA, EPS)

        r_env, counts = evidence_target_for_cause(c, curr_layers, anomaly_norm)

        obs = sensor_observability[c]
        obs_ratio = obs["direct_detections"] / max(obs["exposures"], 1)
        growth = DELTA[c]

        if obs_ratio >= 0.70:
            dD = min(dD, 0)
        elif obs_ratio <= 0.30 and growth > 0:
            dD = max(dD, 0)

        if c in ["C6"] and counts[3] > 0 and obs_ratio <= 0.30:
            dD = max(dD, 1)

        level_pressure = (counts[2] + 2 * counts[3]) / grid_size

        # Calibrate LLM output using actual environmental evidence.
        sev_mult, dD, conf, shift, evidence_strength = calibrate_cause_llm_output(
            cause=c,
            sev_mult=sev_mult,
            dD=dD,
            conf=conf,
            shift=shift,
            growth=growth,
            counts=counts,
            anomaly_norm=anomaly_norm,
            obs_ratio=obs_ratio,
            level_pressure=level_pressure,
        )

        semantic_intensity = conf * (
            max(sev_mult - 1.0, 0.0)
            + np.mean(np.abs(shift))
        )

        gamma = float(np.clip(
            0.08
            + 0.30 * Dn[c]
            + 0.20 * anomaly_norm
            + 0.25 * level_pressure
            + profile["semantic_gamma_boost"] * semantic_intensity,
            profile["gamma_floor"],
            profile["gamma_cap"],
        ))

        q_raw = (1 - gamma) * prev_row + gamma * r_env + profile["llm_shift_weight"] * conf * shift
        q = normalize_vec(q_raw)

        z = profile["evidence_scale"] * sev_mult * (
            0.40 * Wn[c] + 0.25 * Dn[c] + 0.20 * An[c] + 0.15 * level_pressure
        ) * (1.0 + profile["semantic_evidence_boost"] * semantic_intensity)

        posterior_alpha = prior + z * q
        posterior_row = posterior_alpha / posterior_alpha.sum()
        post_probs[c] = posterior_row

        prev_D = prev_detectability[c]
        D_candidate = prev_D + dD + OBSERVABILITY_KAPPA * (0.5 - obs_ratio)
        D_smoothed = (1 - profile["detectability_beta"]) * prev_D + profile["detectability_beta"] * D_candidate
        D_final = int(np.clip(round(D_smoothed), 1, 10))
        post_D[c] = D_final

        evidence_rows.append({
            "Cause": c,
            "Description": CAUSE_DESC[c],
            "Prev weighted extent": round(prev_summary[c]["weighted_extent"], 3),
            "Curr weighted extent": round(curr_summary[c]["weighted_extent"], 3),
            "Positive growth": round(DELTA[c], 3),
            "Movement anomaly": round(A[c], 3),
            "L2 cells": counts[2],
            "L3 cells": counts[3],
            "gamma": round(gamma, 3),
            "z": round(z, 3),
            "LLM severity mult": round(sev_mult, 3),
            "LLM detectability delta": dD,
            "LLM evidence strength": round(evidence_strength, 3),
            "LLM calibrated confidence": round(conf, 3),
            "Prev D": prev_D,
            "Updated D": D_final,
            "LLM reason": llm_c["reason"],
        })

    prob_df = pd.DataFrame(
        {fm: [post_probs[c][j] for c in ACTIVE_CAUSES] for j, fm in enumerate(FAILURE_MODES)},
        index=ACTIVE_CAUSES,
    )
    evidence_df = pd.DataFrame(evidence_rows)
    return prob_df, evidence_df, post_D


def softmax_stable(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def update_effect_table(prev_effect_df, prob_df, evidence_df, effect_llm_out, profile):
    """
    Updates P(E_k | F_j) using:
    - previous/base effect table as prior
    - current dynamic cause->mode table prob_df
    - current evidence_df
    - LLM semantic escalation for each mode
    """
    mode_mass = {fm: float(prob_df[fm].mean()) for fm in FAILURE_MODES}
    max_mass = max(mode_mass.values()) if len(mode_mass) > 0 else 1.0
    mode_mass_norm = {fm: mode_mass[fm] / max(max_mass, EPS) for fm in FAILURE_MODES}

    curr_norm = float(evidence_df["Curr weighted extent"].mean() / max(evidence_df["Curr weighted extent"].max(), EPS))
    growth_norm = float(evidence_df["Positive growth"].mean() / max(evidence_df["Positive growth"].max(), EPS))
    anom_norm = float(evidence_df["Movement anomaly"].mean() / max(evidence_df["Movement anomaly"].max(), EPS))
    l3_total = float(evidence_df["L3 cells"].sum())
    l23_total = float((evidence_df["L2 cells"] + evidence_df["L3 cells"]).sum())
    l3_ratio = l3_total / max(l23_total, 1.0)

    global_context = 0.35 * curr_norm + 0.25 * growth_norm + 0.20 * anom_norm + 0.20 * l3_ratio
    global_context = float(np.clip(global_context, 0.0, 1.0))

    updated_cols = {}
    meta_rows = []

    for fm in FAILURE_MODES:
        base_col = prev_effect_df[fm].values.astype(float)
        base_col = base_col / max(base_col.sum(), EPS)

        llm_f = effect_llm_out["modes"][fm]
        sev_mult = llm_f["severity_multiplier"]
        bias = llm_f["escalation_bias"]
        conf = llm_f["confidence"]
        reason = llm_f["reason"]

        # Calibrate effect-level LLM escalation.
        sev_mult, bias, conf, effect_support = calibrate_effect_llm_output(
            fm=fm,
            sev_mult=sev_mult,
            bias=bias,
            conf=conf,
            mode_mass_norm=mode_mass_norm[fm],
            global_context=global_context,
        )

        u = (
            0.55 * mode_mass_norm[fm]
            + 0.25 * global_context
            + 0.20 * MODE_RISK_WEIGHT[fm]
        )

        semantic_bias = max(bias, 0) * conf
        xi = sev_mult * (u + profile["effect_bias_weight"] * conf * bias)

        logits = np.log(np.maximum(base_col, EPS)) + profile["effect_beta"] * xi * EFFECT_SEV_NORM
        target_col = softmax_stable(logits)

        prior_alpha = profile["effect_prior_strength"] * base_col
        evidence_mass = profile["effect_evidence_scale"] * sev_mult * (0.60 * mode_mass_norm[fm] + 0.40 * global_context) * (1.0 + profile["effect_semantic_mass_boost"] * semantic_bias)

        posterior_alpha = prior_alpha + evidence_mass * target_col
        posterior_col = posterior_alpha / posterior_alpha.sum()
        if profile.get("profile_name") == "llm":
            e6_idx = FAILURE_EFFECTS.index("E6")
            e6_cap = {
                "F1": 0.35,
                "F2": 0.65,
                "F3": 0.82,
                "F4": 0.45,
                "F5": 0.75,
                "F6": 0.65,
            }[fm]

            if posterior_col[e6_idx] > e6_cap:
                excess = posterior_col[e6_idx] - e6_cap
                posterior_col[e6_idx] = e6_cap

                other_idx = [i for i, e in enumerate(FAILURE_EFFECTS) if e != "E6"]
                other_sum = posterior_col[other_idx].sum()

                if other_sum > 1e-9:
                    posterior_col[other_idx] += excess * posterior_col[other_idx] / other_sum
                else:
                    posterior_col[other_idx] += excess / len(other_idx)

            posterior_col = posterior_col / posterior_col.sum()


        
        updated_cols[fm] = posterior_col

        meta_rows.append({
            "Failure Mode": fm,
            "Mode Description": MODE_DESC[fm],
            "Mode mass": round(mode_mass[fm], 4),
            "Mode mass norm": round(mode_mass_norm[fm], 4),
            "Global context": round(global_context, 4),
            "Effect support": round(effect_support, 4),
            "LLM severity mult": round(sev_mult, 4),
            "LLM escalation bias": bias,
            "LLM confidence": round(conf, 4),
            "Calibrated confidence": round(conf, 4),
            "evidence_mass": round(evidence_mass, 4),
            "LLM reason": reason,
        })

    effect_df_dyn = pd.DataFrame(updated_cols, index=FAILURE_EFFECTS)

    for fm in FAILURE_MODES:
        s = effect_df_dyn[fm].sum()
        if s > 0:
            effect_df_dyn[fm] = effect_df_dyn[fm] / s

    meta_df = pd.DataFrame(meta_rows)
    return effect_df_dyn, meta_df

# ============================================================
# DISPLAY
# ============================================================
def print_fmea(prob_df, evidence_df):
    print("\n=== Updated evidence table ===")
    print(evidence_df.to_string(index=False))

    print("\n=== Updated P(F_j | C_i) ===")
    print(prob_df.round(4).to_string())

def plot_probability_table(prob_df, title="(a) Failure cause to failure mode probabilities, P(F_j | C_i)"):
    df = prob_df.copy()
    df.insert(0, "Failure Causes", df.index)
    df = df.reset_index(drop=True)

    fig_h = 1.8 + 0.45 * len(df)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    cell_text = []
    for _, row in df.iterrows():
        cell_text.append(
            [row["Failure Causes"]] + [f"{row[fm]:.2f}" for fm in FAILURE_MODES]
        )

    table = ax.table(
        cellText=cell_text,
        colLabels=["Failure Causes"] + FAILURE_MODES,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 1.6)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.show()

def plot_evidence_table(evidence_df, title="Updated dynamic FMEA evidence summary"):
    fig_h = 1.8 + 0.32 * len(evidence_df)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=evidence_df.values,
        colLabels=evidence_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.show()

def build_effect_probability_df(source_df=None):
    df = BASE_EFFECT_GIVEN_MODE.copy() if source_df is None else source_df.copy()
    for fm in FAILURE_MODES:
        s = df[fm].sum()
        if s > 0:
            df[fm] = df[fm] / s
    return df


def build_effect_description_table(effect_df):
    rows = []
    for e in FAILURE_EFFECTS:
        row = {fm: effect_df.loc[e, fm] for fm in FAILURE_MODES}
        row["Effect"] = e
        row["Failure Effect Description"] = FAILURE_EFFECT_DESC[e]
        row["Severity (S)"] = FAILURE_EFFECT_SEVERITY[e]
        rows.append(row)

    cols = FAILURE_MODES + ["Effect", "Failure Effect Description", "Severity (S)"]
    return pd.DataFrame(rows)[cols]


def compute_effect_given_cause(prob_df, effect_df):
    """
    Computes P(E_k | C_i) = sum_j P(E_k | F_j) P(F_j | C_i)
    """
    mode_given_cause = prob_df[FAILURE_MODES].T.values
    effect_given_mode = effect_df[FAILURE_MODES].values

    effect_given_cause = effect_given_mode @ mode_given_cause

    out_df = pd.DataFrame(
        effect_given_cause,
        index=FAILURE_EFFECTS,
        columns=prob_df.index.tolist(),
    )

    for c in out_df.columns:
        s = out_df[c].sum()
        if s > 0:
            out_df[c] = out_df[c] / s

    return out_df


def plot_effect_probability_table(effect_df, title="(b) Failure mode to failure effect probabilities, P(E_k | F_j)"):
    df = effect_df.copy()
    df.insert(0, "Failure Effects", df.index)
    df = df.reset_index(drop=True)

    fig_h = 2.0 + 0.42 * len(df)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    ax.axis("off")

    cell_text = []
    for _, row in df.iterrows():
        vals = [row["Failure Effects"]] + [f"{row[fm]:.2f}" for fm in FAILURE_MODES]
        cell_text.append(vals)

    table = ax.table(
        cellText=cell_text,
        colLabels=["Failure Effects"] + FAILURE_MODES,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.55)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig, ax


def plot_effect_description_table(effect_desc_df, title="Table 3: Probability of each failure effect resulting from each failure mode."):
    display_df = effect_desc_df.copy()

    for fm in FAILURE_MODES:
        display_df[fm] = display_df[fm].map(lambda x: f"{x:.2f}")

    fig_h = 2.2 + 0.42 * len(display_df)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")

    col_labels = FAILURE_MODES + ["Effect", "Failure Effect Description", "Severity (S)"]
    col_widths = [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05, 0.45, 0.08]

    table = ax.table(
        cellText=display_df.values,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    desc_col_idx = col_labels.index("Failure Effect Description")
    for (r, c), cell in table.get_celld().items():
        if c == desc_col_idx:
            cell.get_text().set_ha("left")
            cell.PAD = 0.02

    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig, ax


def plot_effect_given_cause_table(effect_cause_df, title="Failure cause to failure effect probabilities, P(E_k | C_i)"):
    df = effect_cause_df.copy()
    df.insert(0, "Failure Effects", df.index)
    df = df.reset_index(drop=True)

    fig_h = 2.0 + 0.42 * len(df)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axis("off")

    cause_cols = effect_cause_df.columns.tolist()
    cell_text = []
    for _, row in df.iterrows():
        vals = [row["Failure Effects"]] + [f"{row[c]:.2f}" for c in cause_cols]
        cell_text.append(vals)

    table = ax.table(
        cellText=cell_text,
        colLabels=["Failure Effects"] + cause_cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.55)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig, ax
# ============================================================
# MAIN
# ============================================================
def demo_single_update_unused():
    """
    Optional one-checkpoint demonstration of the NORMAL update only.

    This function is not called by the evaluator. It is kept only for manual
    inspection of the FMEA tables. It updates only P(F|C). The effect table
    P(E|F) remains fixed and is used only to compute P(E|C).
    """
    prev_layers = make_previous_layers()
    curr_layers = make_current_layers()
    movement_summary = BASE_MOVEMENT_SUMMARY
    sensor_observability = BASE_SENSOR_OBSERVABILITY
    prev_detectability = deepcopy(PREV_DETECTABILITY)

    prompt = build_prompt(
        prev_layers,
        curr_layers,
        movement_summary,
        sensor_observability,
        prev_detectability,
        checkpoint_step="demo",
    )

    print("=== Sending cause-to-mode FMEA prompt to local Ollama/Qwen2 ===")
    try:
        raw_llm = call_ollama_qwen2(prompt)
        llm_out = safe_llm_output(raw_llm)
    except Exception as e:
        print(f"Ollama call failed, using neutral fallback. Error: {e}")
        llm_out = neutral_fmea_output()

    prob_df, evidence_df, _ = update_fmea(
        prev_layers,
        curr_layers,
        llm_out,
        deepcopy(PREV_PROBS),
        prev_detectability,
        movement_summary,
        sensor_observability,
        LLM_UPDATE_PROFILE,
    )

    fixed_effect_df = build_effect_probability_df()
    effect_cause_df = compute_effect_given_cause(prob_df, fixed_effect_df)

    print("\n=== Updated P(F_j | C_i); LLM used only here ===")
    print(prob_df.round(4).to_string())

    print("\n=== Fixed P(E_k | F_j); not updated ===")
    print(fixed_effect_df.round(4).to_string())

    print("\n=== Integrated P(E_k | C_i) from fixed P(E|F) and updated P(F|C) ===")
    print(effect_cause_df.round(4).to_string())

    print_fmea(prob_df, evidence_df)
    plot_evidence_table(evidence_df)
    plot_probability_table(prob_df)


# ============================================================
# COMPARISON HELPERS: 3 SITUATIONS
# ============================================================

def neutral_fmea_output():
    """
    Dynamic update WITHOUT LLM:
    keep all semantic knobs neutral so only the deterministic
    evidence-driven Bayesian update acts.
    """
    return {
        "causes": {
            c: {
                "severity_multiplier": 1.0,
                "detectability_delta": 0,
                "confidence": 0.0,
                "reason": "no llm",
                "mode_shift": {fm: 0.0 for fm in FAILURE_MODES},
            }
            for c in ACTIVE_CAUSES
        }
    }


def neutral_effect_output():
    """
    Effect update WITHOUT LLM:
    keep effect-level semantic escalation neutral.
    """
    return {
        "modes": {
            fm: {
                "severity_multiplier": 1.0,
                "escalation_bias": 0,
                "confidence": 0.0,
                "reason": "no llm",
            }
            for fm in FAILURE_MODES
        }
    }


def build_static_prob_df():
    """
    Situation 1:
    fixed cause -> mode table from PREV_PROBS
    """
    return pd.DataFrame.from_dict(
        PREV_PROBS, orient="index", columns=FAILURE_MODES
    ).loc[ACTIVE_CAUSES, FAILURE_MODES]


def build_static_detectability_df():
    """
    Static baseline detectability table.
    """
    return pd.DataFrame({
        "Cause": ACTIVE_CAUSES,
        "Description": [CAUSE_DESC[c] for c in ACTIVE_CAUSES],
        "Updated D": [PREV_DETECTABILITY[c] for c in ACTIVE_CAUSES],
        "Remark": ["Static baseline"] * len(ACTIVE_CAUSES),
    })


def show_tables_for_situation(name, prob_df, effect_df, effect_cause_df, evidence_df=None):
    """
    Print + plot all relevant tables for one situation.
    """
    print(f"\n{'=' * 30} {name} {'=' * 30}")

    if evidence_df is not None:
        print("\n--- Evidence / detectability table ---")
        print(evidence_df.to_string(index=False))
        plot_evidence_table(evidence_df, title=f"{name}: Evidence summary")

    print("\n--- P(F_j | C_i) ---")
    print(prob_df.round(4).to_string())
    plot_probability_table(prob_df, title=f"{name}: P(F_j | C_i)")

    print("\n--- P(E_k | F_j) ---")
    print(effect_df.round(4).to_string())
    plot_effect_probability_table(effect_df, title=f"{name}: P(E_k | F_j)")

    effect_desc_df = build_effect_description_table(effect_df)
    print("\n--- Failure effect description table ---")
    print(effect_desc_df.to_string(index=False))
    plot_effect_description_table(
        effect_desc_df,
        title=f"{name}: Failure effect description table"
    )

    print("\n--- P(E_k | C_i) ---")
    print(effect_cause_df.round(4).to_string())
    plot_effect_given_cause_table(effect_cause_df, title=f"{name}: P(E_k | C_i)")


def summarize_checkpoint_metrics(step, prob_static, effect_static, prob_df, effect_df, effect_cause_df, label):
    static_effect_cause = compute_effect_given_cause(prob_static, effect_static)
    return {
        "Step": step,
        f"|P(F|C)-static| ({label})": round(mean_abs_df_delta(prob_df, prob_static), 5),
        f"|P(E|F)-static| ({label})": round(mean_abs_df_delta(effect_df, effect_static), 5),
        f"|P(E|C)-static| ({label})": round(mean_abs_df_delta(effect_cause_df, static_effect_cause), 5),
    }


def run_dynamic_sequence(update_steps, use_llm):
    """
    Normal FMEA update sequence.

    Only P(F_j | C_i) is updated over time. The effect table P(E_k | F_j)
    is kept fixed, so the integrated risk table is computed as:

        P(E|C) = P(E|F)_fixed @ P(F|C)_updated

    This avoids a second effect-level update and keeps the experiment easy to
    explain: Static = no update, Dynamic = evidence update, LLM = evidence +
    semantic update.
    """
    prev_layers = make_previous_layers()
    prev_probs = deepcopy(PREV_PROBS)
    prev_detectability = deepcopy(PREV_DETECTABILITY)
    fixed_effect_df = build_effect_probability_df()

    history = []
    snapshots = {}
    last_payload = None
    profile = LLM_UPDATE_PROFILE if use_llm else BASE_UPDATE_PROFILE

    for step in update_steps:
        curr_layers, movement_summary, sensor_observability = get_state_at_step(step)

        if use_llm:
            prompt = build_prompt(
                prev_layers,
                curr_layers,
                movement_summary,
                sensor_observability,
                prev_detectability,
                checkpoint_step=step,
            )

            print(f"\n=== Sending normal FMEA update prompt for step {step} ===")
            try:
                raw_llm = call_ollama_qwen2(prompt)
                llm_out = safe_llm_output(raw_llm)
            except Exception as e:
                print(f"LLM failed at step {step}, using neutral fallback. Error: {e}")
                llm_out = neutral_fmea_output()
        else:
            llm_out = neutral_fmea_output()

        # Normal update: update only P(F|C).
        prob_df, evidence_df, post_detectability = update_fmea(
            prev_layers,
            curr_layers,
            llm_out,
            prev_probs,
            prev_detectability,
            movement_summary,
            sensor_observability,
            profile,
        )

        effect_df = fixed_effect_df.copy()
        effect_meta_df = pd.DataFrame([{
            "Step": step,
            "Remark": "P(E|F) fixed; only P(F|C) updated",
        }])
        effect_cause_df = compute_effect_given_cause(prob_df, effect_df)

        prev_prob_df = pd.DataFrame.from_dict(
            prev_probs, orient="index", columns=FAILURE_MODES
        ).loc[ACTIVE_CAUSES, FAILURE_MODES]

        history.append({
            "Step": step,
            "Mean |Δ P(F|C)| from previous": round(mean_abs_df_delta(prob_df, prev_prob_df), 5),
            "Mean |Δ P(E|F)| from previous": 0.0,
            "Mean detectability": round(float(evidence_df["Updated D"].mean()), 4),
            "Total L3 cells": int(evidence_df["L3 cells"].sum()),
            "Total anomaly": round(float(evidence_df["Movement anomaly"].sum()), 4),
        })

        last_payload = {
            "prob_df": prob_df,
            "effect_df": effect_df,
            "effect_cause_df": effect_cause_df,
            "evidence_df": evidence_df,
            "effect_meta_df": effect_meta_df,
        }
        snapshots[step] = {
            "prob_df": prob_df.copy(),
            "effect_df": effect_df.copy(),
            "effect_cause_df": effect_cause_df.copy(),
            "evidence_df": evidence_df.copy(),
            "effect_meta_df": effect_meta_df.copy(),
        }

        prev_layers = deepcopy(curr_layers)
        prev_probs = prob_df_to_dict(prob_df)
        prev_detectability = post_detectability

    return last_payload, pd.DataFrame(history), snapshots


def catastrophic_mass(effect_cause_df):
    return float((effect_cause_df.loc["E6"] + effect_cause_df.loc["E7"]).mean())

def plot_multi_update_history(history_compare_df):
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(history_compare_df["Step"], history_compare_df["|P(F|C)-static| (No LLM)"], marker="o", label="P(F|C) vs static - No LLM")
    ax.plot(history_compare_df["Step"], history_compare_df["|P(F|C)-static| (With LLM)"], marker="o", label="P(F|C) vs static - With LLM")
    ax.plot(history_compare_df["Step"], history_compare_df["|P(F|C) noLLM-vs-LLM|"], marker="D", label="P(F|C) No LLM vs LLM")
    ax.plot(history_compare_df["Step"], history_compare_df["|P(E|F) noLLM-vs-LLM|"], marker="s", label="P(E|F) No LLM vs LLM")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Mean absolute table difference")
    ax.set_title("Multi-update FMEA drift and LLM separation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def run_three_situations():
    update_steps = UPDATE_STEPS

    prob_static = build_static_prob_df()
    effect_static = build_effect_probability_df()
    effect_cause_static = compute_effect_given_cause(prob_static, effect_static)
    static_detect_df = build_static_detectability_df()

    no_llm_payload, no_llm_history, no_llm_snapshots = run_dynamic_sequence(update_steps, use_llm=False)
    llm_payload, llm_history, llm_snapshots = run_dynamic_sequence(update_steps, use_llm=True)

    compare_rows = []
    for step in update_steps:
        no_row = no_llm_history[no_llm_history["Step"] == step].iloc[0].to_dict()
        llm_row = llm_history[llm_history["Step"] == step].iloc[0].to_dict()

        no_metrics = summarize_checkpoint_metrics(
            step,
            prob_static,
            effect_static,
            no_llm_snapshots[step]["prob_df"],
            no_llm_snapshots[step]["effect_df"],
            no_llm_snapshots[step]["effect_cause_df"],
            "No LLM",
        )
        llm_metrics = summarize_checkpoint_metrics(
            step,
            prob_static,
            effect_static,
            llm_snapshots[step]["prob_df"],
            llm_snapshots[step]["effect_df"],
            llm_snapshots[step]["effect_cause_df"],
            "With LLM",
        )

        compare_rows.append({
            "Step": step,
            "Mean |Δ P(F|C)| from previous (No LLM)": no_row["Mean |Δ P(F|C)| from previous"],
            "Mean |Δ P(F|C)| from previous (With LLM)": llm_row["Mean |Δ P(F|C)| from previous"],
            "Mean |Δ P(E|F)| from previous (No LLM)": no_row["Mean |Δ P(E|F)| from previous"],
            "Mean |Δ P(E|F)| from previous (With LLM)": llm_row["Mean |Δ P(E|F)| from previous"],
            "Mean detectability (No LLM)": no_row["Mean detectability"],
            "Mean detectability (With LLM)": llm_row["Mean detectability"],
            "Total L3 cells": no_row["Total L3 cells"],
            "|P(F|C)-static| (No LLM)": no_metrics["|P(F|C)-static| (No LLM)"],
            "|P(F|C)-static| (With LLM)": llm_metrics["|P(F|C)-static| (With LLM)"],
            "|P(E|F)-static| (No LLM)": no_metrics["|P(E|F)-static| (No LLM)"],
            "|P(E|F)-static| (With LLM)": llm_metrics["|P(E|F)-static| (With LLM)"],
            "|P(E|C)-static| (No LLM)": no_metrics["|P(E|C)-static| (No LLM)"],
            "|P(E|C)-static| (With LLM)": llm_metrics["|P(E|C)-static| (With LLM)"],
            "|P(F|C) noLLM-vs-LLM|": round(mean_abs_df_delta(no_llm_snapshots[step]["prob_df"], llm_snapshots[step]["prob_df"]), 5),
            "|P(E|F) noLLM-vs-LLM|": round(mean_abs_df_delta(no_llm_snapshots[step]["effect_df"], llm_snapshots[step]["effect_df"]), 5),
            "Catastrophic mass (No LLM)": round(catastrophic_mass(no_llm_snapshots[step]["effect_cause_df"]), 5),
            "Catastrophic mass (With LLM)": round(catastrophic_mass(llm_snapshots[step]["effect_cause_df"]), 5),
        })

    history_compare_df = pd.DataFrame(compare_rows)

    print("\n=== Multi-update comparison history ===")
    print(history_compare_df.to_string(index=False))
    plot_multi_update_history(history_compare_df)

    show_tables_for_situation(
        "Situation 1 - Static FMEA",
        prob_static,
        effect_static,
        effect_cause_static,
        static_detect_df,
    )

    show_tables_for_situation(
        f"Situation 2 - Dynamic FMEA without LLM (final step {update_steps[-1]})",
        no_llm_payload["prob_df"],
        no_llm_payload["effect_df"],
        no_llm_payload["effect_cause_df"],
        no_llm_payload["evidence_df"],
    )

    print("\n=== Fixed effect-table note (without LLM, final checkpoint) ===")
    print(no_llm_payload["effect_meta_df"].to_string(index=False))

    show_tables_for_situation(
        f"Situation 3 - Dynamic FMEA with LLM (final step {update_steps[-1]})",
        llm_payload["prob_df"],
        llm_payload["effect_df"],
        llm_payload["effect_cause_df"],
        llm_payload["evidence_df"],
    )

    print("\n=== Fixed effect-table note (with LLM, final checkpoint) ===")
    print(llm_payload["effect_meta_df"].to_string(index=False))


def main():
    run_three_situations()


if __name__ == "__main__":
    main()
