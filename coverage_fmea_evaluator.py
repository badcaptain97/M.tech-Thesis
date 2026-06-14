
import json
import heapq
import hashlib
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import multi_update_fmea as fmea


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path("coverage_randomized_astar_results")
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = OUTPUT_DIR / "llm_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

# Final setting: 10 randomized maps × 5 seeds.
# For quick debugging, temporarily set this to 2.
N_RANDOM_MAPS = 10
RUN_SEEDS = [11, 23, 37, 101, 202]
SCENARIO_SEEDS = [1001 + i for i in range(N_RANDOM_MAPS)]


ROBOT_COUNTS = [3, 5, 7]

# Higher coverage target.
TARGET_COVERAGE = 1.00
MAX_COVERAGE_STEPS = 220

# Stronger dynamic hazard suite.
HIGH_RISK_SUITE = True

USE_LLM_CACHE = True
# ============================================================
# PLANNER WEIGHTS
# ============================================================
# More distance pressure and less over-conservative risk penalty.
# This reduces long detours and cumulative exposure.

BASE_ASTAR_DAMAGE_RISK_WEIGHT = 3.25
BASE_ASTAR_MISSION_RISK_WEIGHT = 1.10

TARGET_DISTANCE_WEIGHT = 0.80
TARGET_DAMAGE_WEIGHT = 2.10
TARGET_MISSION_WEIGHT = 0.75
TARGET_CLUSTER_BONUS_WEIGHT = 1.35
TARGET_UNVISITED_BONUS_WEIGHT = 3.25

TARGET_ALREADY_ASSIGNED_PENALTY = 1000.0

# Planner ignores tiny/moderate risk and reacts mainly to truly risky cells.
PLANNER_RISK_FREE_MARGIN = 0.18

RISKY_CELL_THRESHOLD = 0.25
HIGH_RISK_CELL_THRESHOLD = 0.50

# ============================================================
# EFFECT CATEGORIES
# ============================================================

DAMAGE_EFFECTS = ["E2", "E3", "E5", "E6"]
CATASTROPHIC_DAMAGE_EFFECTS = ["E3", "E6"]
MISSION_FAILURE_EFFECTS = ["E6", "E7", "E8"]


# ============================================================
# COMMON ORACLE RISK MODEL
# Used only for evaluation, not planning.
# ============================================================

ORACLE_DAMAGE_BY_CAUSE = {
    "C1": 0.42,  # terrain irregularity
    "C2": 0.36,  # slip
    "C3": 0.28,  # communication failure
    "C4": 0.34,  # dynamic obstacle
    "C5": 0.31,  # static obstacle
    "C6": 0.55,  # amplified hazard
}

ORACLE_CATASTROPHIC_BY_CAUSE = {
    "C1": 0.24,
    "C2": 0.18,
    "C3": 0.12,
    "C4": 0.20,
    "C5": 0.16,
    "C6": 0.34,
}

ORACLE_MISSION_FAILURE_BY_CAUSE = {
    "C1": 0.30,
    "C2": 0.28,
    "C3": 0.52,
    "C4": 0.36,
    "C5": 0.40,
    "C6": 0.60,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def get_grid_shape():
    layers = fmea.make_empty_layers()
    first = next(iter(layers.values()))
    return first.shape


def clone_layers(layers):
    return {c: layers[c].copy() for c in fmea.ACTIVE_CAUSES}


def in_bounds(pos, h, w):
    r, c = pos
    return 0 <= r < h and 0 <= c < w


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors4(pos, h, w):
    r, c = pos
    cand = [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    ]
    return [x for x in cand if in_bounds(x, h, w)]


def combine_probabilities(prob_list):
    survival = 1.0
    for p in prob_list:
        p = float(np.clip(p, 0.0, 0.999))
        survival *= (1.0 - p)
    return float(1.0 - survival)


def normalized_level_weight(cause, level):
    if level <= 0:
        return 0.0

    max_w = max(fmea.LEVEL_WEIGHTS[cause].values())
    raw_w = fmea.LEVEL_WEIGHTS[cause].get(int(level), 0.0)

    return float(raw_w / max(max_w, 1e-9))


def effect_mass(effect_cause_df, cause, effects):
    value = 0.0
    for e in effects:
        if e in effect_cause_df.index and cause in effect_cause_df.columns:
            value += float(effect_cause_df.loc[e, cause])

    return float(np.clip(value, 0.0, 1.0))


# ============================================================
# RANDOMIZED HAZARD MAP GENERATION
# ============================================================

def add_patch(layer, center, radius, level, h, w):
    cr, cc = center

    for r in range(cr - radius, cr + radius + 1):
        for c in range(cc - radius, cc + radius + 1):
            if 0 <= r < h and 0 <= c < w:
                if abs(r - cr) + abs(c - cc) <= radius:
                    layer[r, c] = max(layer[r, c], level)


def add_random_patch(layers, cause, rng, h, w, level, radius=None):
    if radius is None:
        radius = int(rng.choice([0, 1, 1, 2], p=[0.20, 0.50, 0.20, 0.10]))

    center = (
        int(rng.integers(0, h)),
        int(rng.integers(0, w)),
    )

    add_patch(layers[cause], center, radius, int(level), h, w)


def escalate_existing_cells(layer, rng, probability):
    h, w = layer.shape

    for r in range(h):
        for c in range(w):
            if layer[r, c] > 0 and layer[r, c] < 3:
                if rng.random() < probability:
                    layer[r, c] += 1


def generate_random_hazard_sequence(scenario_seed):
    """
    Generates one randomized dynamic hazard sequence.

    HIGH_RISK_SUITE=True creates stronger dynamic hazard growth so that
    static FMEA becomes outdated and dynamic/LLM FMEA has a meaningful role.
    """
    rng = np.random.default_rng(scenario_seed)
    h, w = get_grid_shape()

    layers = fmea.make_empty_layers(h, w)

    # --------------------------------------------------------
    # Initial hazard density
    # --------------------------------------------------------
    for cause in fmea.ACTIVE_CAUSES:
        if HIGH_RISK_SUITE:
            if cause == "C6":
                n_base = int(rng.integers(1, 3))
            elif cause == "C3":
                n_base = int(rng.integers(2, 4))
            elif cause in ["C1", "C2", "C4"]:
                n_base = int(rng.integers(3, 5))
            else:
                n_base = int(rng.integers(2, 4))

            level_probs = [0.45, 0.40, 0.15]  # more level-2 and level-3
        else:
            if cause == "C6":
                n_base = int(rng.integers(0, 2))
            elif cause == "C3":
                n_base = int(rng.integers(1, 3))
            else:
                n_base = int(rng.integers(1, 4))

            level_probs = [0.70, 0.25, 0.05]

        for _ in range(n_base):
            level = int(rng.choice([1, 2, 3], p=level_probs))
            add_random_patch(layers, cause, rng, h, w, level=level)

    sequence = {0: clone_layers(layers)}

    # --------------------------------------------------------
    # Dynamic growth over update checkpoints
    # --------------------------------------------------------
    for step in fmea.UPDATE_STEPS:
        progress = step / max(fmea.UPDATE_STEPS)
        layers = clone_layers(layers)

        for cause in fmea.ACTIVE_CAUSES:
            # Stronger escalation in high-risk suite.
            if HIGH_RISK_SUITE:
                if cause in ["C1", "C2", "C6"]:
                    esc_prob = 0.08 + 0.18 * progress
                elif cause == "C3":
                    esc_prob = 0.06 + 0.14 * progress
                else:
                    esc_prob = 0.05 + 0.12 * progress
            else:
                if cause in ["C1", "C2", "C6"]:
                    esc_prob = 0.04 + 0.10 * progress
                elif cause == "C3":
                    esc_prob = 0.03 + 0.07 * progress
                else:
                    esc_prob = 0.03 + 0.06 * progress

            escalate_existing_cells(layers[cause], rng, esc_prob)

            # Growth probability.
            if HIGH_RISK_SUITE:
                if cause == "C6":
                    growth_prob = 0.45 + 0.45 * progress
                elif cause in ["C1", "C2", "C4"]:
                    growth_prob = 0.55 + 0.35 * progress
                elif cause == "C3":
                    growth_prob = 0.45 + 0.35 * progress
                else:
                    growth_prob = 0.40 + 0.30 * progress
            else:
                if cause == "C6":
                    growth_prob = 0.20 + 0.55 * progress
                elif cause in ["C1", "C2", "C4"]:
                    growth_prob = 0.35 + 0.45 * progress
                else:
                    growth_prob = 0.25 + 0.35 * progress

            if rng.random() < growth_prob:
                if HIGH_RISK_SUITE:
                    n_new = int(rng.integers(1, 4))
                else:
                    n_new = int(rng.integers(1, 3))

                for _ in range(n_new):
                    if HIGH_RISK_SUITE:
                        if progress > 0.70:
                            level = int(rng.choice([1, 2, 3], p=[0.20, 0.45, 0.35]))
                        elif progress > 0.40:
                            level = int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
                        else:
                            level = int(rng.choice([1, 2], p=[0.60, 0.40]))
                    else:
                        if progress > 0.70:
                            level = int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
                        elif progress > 0.40:
                            level = int(rng.choice([1, 2], p=[0.55, 0.45]))
                        else:
                            level = 1

                    add_random_patch(layers, cause, rng, h, w, level=level)

            # Extra late amplified hazards.
            if HIGH_RISK_SUITE and cause == "C6" and progress >= 0.50:
                if rng.random() < 0.45:
                    level = int(rng.choice([2, 3], p=[0.55, 0.45]))
                    add_random_patch(layers, cause, rng, h, w, level=level)

        sequence[step] = clone_layers(layers)

    return sequence


def nearest_update_step(t):
    valid = [s for s in fmea.UPDATE_STEPS if s <= t]
    if not valid:
        return 0
    return max(valid)


def layers_at_time(sequence, t):
    return sequence[nearest_update_step(t)]


# ============================================================
# RISK MAPS
# ============================================================

def build_estimated_risk_maps(layers, effect_cause_df):
    h, w = next(iter(layers.values())).shape

    damage_map = np.zeros((h, w), dtype=float)
    catastrophic_map = np.zeros((h, w), dtype=float)
    mission_map = np.zeros((h, w), dtype=float)

    for r in range(h):
        for c in range(w):
            damage_terms = []
            catastrophic_terms = []
            mission_terms = []

            for cause in fmea.ACTIVE_CAUSES:
                level = int(layers[cause][r, c])
                if level <= 0:
                    continue

                lw = normalized_level_weight(cause, level)

                p_damage = effect_mass(effect_cause_df, cause, DAMAGE_EFFECTS)
                p_cat = effect_mass(effect_cause_df, cause, CATASTROPHIC_DAMAGE_EFFECTS)
                p_mission = effect_mass(effect_cause_df, cause, MISSION_FAILURE_EFFECTS)

                damage_terms.append(lw * p_damage)
                catastrophic_terms.append(lw * p_cat)
                mission_terms.append(lw * p_mission)

            damage_map[r, c] = combine_probabilities(damage_terms)
            catastrophic_map[r, c] = combine_probabilities(catastrophic_terms)
            mission_map[r, c] = combine_probabilities(mission_terms)

    return damage_map, catastrophic_map, mission_map


def build_oracle_risk_maps(layers):
    h, w = next(iter(layers.values())).shape

    damage_map = np.zeros((h, w), dtype=float)
    catastrophic_map = np.zeros((h, w), dtype=float)
    mission_map = np.zeros((h, w), dtype=float)

    for r in range(h):
        for c in range(w):
            damage_terms = []
            catastrophic_terms = []
            mission_terms = []

            for cause in fmea.ACTIVE_CAUSES:
                level = int(layers[cause][r, c])
                if level <= 0:
                    continue

                lw = normalized_level_weight(cause, level)

                damage_terms.append(lw * ORACLE_DAMAGE_BY_CAUSE[cause])
                catastrophic_terms.append(lw * ORACLE_CATASTROPHIC_BY_CAUSE[cause])
                mission_terms.append(lw * ORACLE_MISSION_FAILURE_BY_CAUSE[cause])

            damage_map[r, c] = combine_probabilities(damage_terms)
            catastrophic_map[r, c] = combine_probabilities(catastrophic_terms)
            mission_map[r, c] = combine_probabilities(mission_terms)

    return damage_map, catastrophic_map, mission_map


# ============================================================
# LLM CACHE
# ============================================================

def prompt_hash(prompt):
    """Hash the full prompt so old cache files cannot contaminate new maps/profiles."""
    if prompt is None:
        return "nohash"
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def cache_path(kind, scenario_id, step, prompt=None):
    h = prompt_hash(prompt)
    return CACHE_DIR / f"{kind}_scenario_{scenario_id}_step_{step}_{h}.json"


def cached_cause_llm(prompt, scenario_id, step):
    path = cache_path("cause", scenario_id, step, prompt)

    if USE_LLM_CACHE and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            # If old cache is already in correct safe format
            if isinstance(cached, dict) and "causes" in cached:
                return fmea.safe_llm_output(cached)

            # If old cache accidentally stored a list or malformed object
            print(f"Bad cause cache format at {path}. Recomputing...")
            path.unlink(missing_ok=True)

        except Exception as e:
            print(f"Could not read cause cache {path}. Recomputing. Error: {e}")
            path.unlink(missing_ok=True)

    raw = fmea.call_ollama_qwen2(prompt)

    # Sometimes local LLM may return a JSON list. Convert safely.
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            raw = raw[0]
        else:
            raw = {}

    safe = fmea.safe_llm_output(raw)

    if USE_LLM_CACHE:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2)

    return safe


def cached_effect_llm(prompt, scenario_id, step):
    path = cache_path("effect", scenario_id, step, prompt)

    if USE_LLM_CACHE and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            if isinstance(cached, dict) and "modes" in cached:
                return fmea.safe_effect_llm_output(cached)

            print(f"Bad effect cache format at {path}. Recomputing...")
            path.unlink(missing_ok=True)

        except Exception as e:
            print(f"Could not read effect cache {path}. Recomputing. Error: {e}")
            path.unlink(missing_ok=True)

    raw = fmea.call_ollama_effects(prompt)

    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            raw = raw[0]
        else:
            raw = {}

    safe = fmea.safe_effect_llm_output(raw)

    if USE_LLM_CACHE:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2)

    return safe


# ============================================================
# FMEA SNAPSHOT PACKS FOR RANDOMIZED MAPS
# ============================================================

def build_static_pack_for_scenario(sequence):
    prob_static = fmea.build_static_prob_df()
    effect_static = fmea.build_effect_probability_df()
    effect_cause_static = fmea.compute_effect_given_cause(prob_static, effect_static)

    snapshots = {
        0: {
            "prob_df": prob_static.copy(),
            "effect_df": effect_static.copy(),
            "effect_cause_df": effect_cause_static.copy(),
        }
    }
    for step in fmea.UPDATE_STEPS:
        snapshots[step] = {
            "prob_df": prob_static.copy(),
            "effect_df": effect_static.copy(),
            "effect_cause_df": effect_cause_static.copy(),
        }

    return {
        "method": "Static FMEA",
        "snapshots": snapshots,
    }


def validate_static_pack(static_pack):
    """Guarantee that Static FMEA tables are fixed at every checkpoint."""
    base = static_pack["snapshots"][0]
    for step, snap in static_pack["snapshots"].items():
        for key in ["prob_df", "effect_df", "effect_cause_df"]:
            if not np.allclose(base[key].values, snap[key].values):
                raise AssertionError(f"Static FMEA changed at step {step} for {key}")


def build_dynamic_pack_for_scenario(sequence, use_llm, scenario_id):
    """
    Build dynamic FMEA snapshots for one randomized map.

    Important experimental rule:
    - Dynamic methods update only the cause-to-mode table P(F|C).
    - The failure-mode-to-effect table P(E|F) remains fixed for all methods.
    - The LLM, when enabled, is used only inside the P(F|C) update.
    """
    if not use_llm:
        label = "Dynamic FMEA without LLM"
    else:
        label = "Dynamic FMEA + LLM"

    cause_profile = fmea.LLM_UPDATE_PROFILE if use_llm else fmea.BASE_UPDATE_PROFILE

    prev_layers = clone_layers(sequence[0])
    prev_probs = deepcopy(fmea.PREV_PROBS)
    prev_detectability = deepcopy(fmea.PREV_DETECTABILITY)
    prev_effect_df = fmea.build_effect_probability_df()

    initial_prob_df = pd.DataFrame.from_dict(
        prev_probs, orient="index", columns=fmea.FAILURE_MODES
    ).loc[fmea.ACTIVE_CAUSES, fmea.FAILURE_MODES]
    initial_effect_df = fmea.build_effect_probability_df(prev_effect_df)
    initial_effect_cause_df = fmea.compute_effect_given_cause(
        initial_prob_df, initial_effect_df
    )

    snapshots = {
        0: {
            "prob_df": initial_prob_df.copy(),
            "effect_df": initial_effect_df.copy(),
            "effect_cause_df": initial_effect_cause_df.copy(),
        }
    }
    history_rows = []

    for step in fmea.UPDATE_STEPS:
        curr_layers = clone_layers(sequence[step])

        def scenario_movement_summary(prev_layers, curr_layers, step, scenario_id):
            """
            Builds movement/anomaly summary from actual hazard growth.
            This makes FMEA evidence scenario-specific instead of only step-specific.
            """
            summary = {k: 0 for k in fmea.BASE_MOVEMENT_SUMMARY.keys()}

            growth_by_cause = {}

            for cause in fmea.ACTIVE_CAUSES:
                prev = prev_layers[cause]
                curr = curr_layers[cause]

                growth = np.maximum(curr - prev, 0)
                l2_cells = int(np.sum(curr == 2))
                l3_cells = int(np.sum(curr == 3))
                new_cells = int(np.sum((prev == 0) & (curr > 0)))

                growth_by_cause[cause] = {
                    "growth": int(np.sum(growth)),
                    "l2": l2_cells,
                    "l3": l3_cells,
                    "new": new_cells,
                }

            # C1 terrain irregularity evidence
            summary["lateral_deviation"] = 2 + growth_by_cause["C1"]["new"]
            summary["tilt_warning"] = growth_by_cause["C1"]["l3"] + growth_by_cause["C6"]["l3"]
            summary["rough_progress_loss"] = growth_by_cause["C1"]["growth"]

            # C2 slip evidence
            summary["wheel_slip"] = 2 + growth_by_cause["C2"]["new"] + growth_by_cause["C2"]["l2"]
            summary["stall_steps"] = 2 + growth_by_cause["C2"]["growth"]
            summary["traction_recovery"] = max(1, growth_by_cause["C2"]["new"] // 2)

            # C3 communication evidence
            summary["comm_drop"] = 1 + growth_by_cause["C3"]["new"] + growth_by_cause["C3"]["l2"]
            summary["heartbeat_timeout"] = growth_by_cause["C3"]["l2"] + growth_by_cause["C3"]["l3"]
            summary["reconnect_attempt"] = max(1, growth_by_cause["C3"]["new"] // 2)

            # C4 dynamic obstacle evidence
            summary["dynamic_replan"] = 1 + growth_by_cause["C4"]["new"]
            summary["near_collision"] = growth_by_cause["C4"]["l2"] + growth_by_cause["C4"]["l3"]
            summary["avoidance_brake"] = 1 + growth_by_cause["C4"]["growth"]

            # C5 static obstacle evidence
            summary["blocked_turnback"] = 1 + growth_by_cause["C5"]["new"]
            summary["clearance_warning"] = growth_by_cause["C5"]["l2"] + growth_by_cause["C5"]["l3"]
            summary["wall_following_detour"] = max(1, growth_by_cause["C5"]["growth"] // 2)

            # C6 amplified hazard evidence
            summary["risk_spike"] = 1 + 2 * growth_by_cause["C6"]["l3"] + growth_by_cause["C6"]["l2"]
            summary["hard_escape_maneuver"] = growth_by_cause["C6"]["new"] + growth_by_cause["C6"]["l3"]

            return {k: int(max(0, v)) for k, v in summary.items()}
            
        def scenario_sensor_observability(curr_layers, scenario_id, step):
            """
            Builds sensor observability from the actual randomized hazard map.
            Higher visible obstacle hazards get better observability.
            Hidden hazards such as C6 and C3 remain harder to detect.
            """
            sensor_profiles = {}

            for cause in fmea.ACTIVE_CAUSES:
                layer = curr_layers[cause]

                active = int(np.sum(layer > 0))
                l2 = int(np.sum(layer == 2))
                l3 = int(np.sum(layer == 3))

                exposures = max(3, active + l2 + 2 * l3)

                if cause == "C5":
                    # static obstacles are highly visible
                    detect_rate = 0.75
                elif cause == "C4":
                    # dynamic obstacles often sensed directly
                    detect_rate = 0.70
                elif cause == "C3":
                    # communication shadow is partially observable
                    detect_rate = 0.45
                elif cause == "C6":
                    # amplified hazard is difficult to detect
                    detect_rate = 0.30
                elif cause == "C1":
                    detect_rate = 0.50
                elif cause == "C2":
                    detect_rate = 0.45
                else:
                    detect_rate = 0.50

                # Severe cells slightly increase detections, but not perfectly.
                severity_bonus = min(0.15, 0.03 * l2 + 0.05 * l3)
                detect_rate = min(0.90, detect_rate + severity_bonus)

                direct_detections = int(round(exposures * detect_rate))

                sensor_profiles[cause] = {
                    "direct_detections": max(0, direct_detections),
                    "exposures": max(1, exposures),
                }

            return sensor_profiles
        
        movement_summary = scenario_movement_summary(
            prev_layers=prev_layers,
            curr_layers=curr_layers,
            step=step,
            scenario_id=scenario_id,
        )

        sensor_observability = scenario_sensor_observability(
            curr_layers=curr_layers,
            scenario_id=scenario_id,
            step=step,
        )

        assert isinstance(movement_summary, dict), type(movement_summary)
        assert isinstance(sensor_observability, dict), type(sensor_observability)

        if use_llm:
            prompt = fmea.build_prompt(
                prev_layers,
                curr_layers,
                movement_summary,
                sensor_observability,
                prev_detectability,
                checkpoint_step=step,
            )

            print(f"Scenario {scenario_id}, step {step}: cause-level LLM")
            try:
                llm_out = cached_cause_llm(prompt, scenario_id, step)
            except Exception as e:
                print(f"Cause LLM failed. Using neutral fallback. Error: {e}")
                llm_out = fmea.neutral_fmea_output()
        else:
            llm_out = fmea.neutral_fmea_output()

        prob_df, evidence_df, post_detectability = fmea.update_fmea(
            prev_layers,
            curr_layers,
            llm_out,
            prev_probs,
            prev_detectability,
            movement_summary,
            sensor_observability,
            cause_profile,
        )

        # Keep P(E|F) fixed for every method.
        # Only P(F|C) changes over time; risk is recomputed as P(E|C)=P(E|F)_fixed x P(F|C).
        effect_df = fmea.build_effect_probability_df()
        effect_cause_df = fmea.compute_effect_given_cause(prob_df, effect_df)
        effect_meta_df = pd.DataFrame([{
            "Scenario": scenario_id,
            "Step": step,
            "Method": label,
            "Note": "P(E|F) fixed; no effect-level update",
        }])

        snapshots[step] = {
            "prob_df": prob_df.copy(),
            "effect_df": effect_df.copy(),
            "effect_cause_df": effect_cause_df.copy(),
            "evidence_df": evidence_df.copy(),
            "effect_meta_df": effect_meta_df.copy(),
        }

        history_rows.append({
            "Scenario": scenario_id,
            "Step": step,
            "Method": label,
            "Mean detectability": float(evidence_df["Updated D"].mean()),
            "Total L3 cells": int(evidence_df["L3 cells"].sum()),
            "Total anomaly": float(evidence_df["Movement anomaly"].sum()),
        })

        prev_layers = clone_layers(curr_layers)
        prev_probs = fmea.prob_df_to_dict(prob_df)
        prev_detectability = deepcopy(post_detectability)
        # P(E|F) intentionally remains fixed; do not update prev_effect_df.

    return {
        "method": label,
        "snapshots": snapshots,
        "history": pd.DataFrame(history_rows),
    }


def get_effect_cause_for_time(pack, t):
    """
    Use the latest FMEA table whose checkpoint has actually occurred.
    Before step 5, this correctly uses snapshot 0 instead of leaking snapshot 5.
    """
    step = nearest_update_step(t)
    available = sorted(pack["snapshots"].keys())
    if step not in pack["snapshots"]:
        step = max(s for s in available if s <= step)
    return pack["snapshots"][step]["effect_cause_df"]


# ============================================================
# START POSITION SAMPLING
# ============================================================

def sample_start_positions(sequence, run_seed, n_robots):
    rng = np.random.default_rng(run_seed)
    h, w = get_grid_shape()

    initial_layers = sequence[0]
    damage_map, _, _ = build_oracle_risk_maps(initial_layers)

    all_cells = []
    for r in range(h):
        for c in range(w):
            all_cells.append(((r, c), float(damage_map[r, c])))

    all_cells.sort(key=lambda x: x[1])

    # Larger safe pool for 4/5 robots.
    pool_size = min(70, len(all_cells))
    pool = [cell for cell, _ in all_cells[:pool_size]]

    selected = []
    tries = 0

    # Slightly relaxed separation for more robots.
    min_sep = 3 if n_robots <= 4 else 2

    while len(selected) < n_robots and tries < 2000:
        tries += 1
        cand = pool[int(rng.integers(0, len(pool)))]

        if cand in selected:
            continue

        if all(manhattan(cand, x) >= min_sep for x in selected):
            selected.append(cand)

    while len(selected) < n_robots:
        cand = pool[int(rng.integers(0, len(pool)))]
        if cand not in selected:
            selected.append(cand)

    return selected


# ============================================================
# FRONTIER A* PLANNER
# ============================================================

def get_frontier_cells(visited, h, w):
    frontiers = []

    for r in range(h):
        for c in range(w):
            cell = (r, c)

            if cell in visited:
                continue

            if any(nb in visited for nb in neighbors4(cell, h, w)):
                frontiers.append(cell)

    if not frontiers:
        for r in range(h):
            for c in range(w):
                if (r, c) not in visited:
                    frontiers.append((r, c))

    return frontiers


def local_unvisited_cluster(cell, visited, h, w, radius=2):
    r0, c0 = cell
    count = 0

    for r in range(max(0, r0 - radius), min(h, r0 + radius + 1)):
        for c in range(max(0, c0 - radius), min(w, c0 + radius + 1)):
            if (r, c) not in visited:
                count += 1

    return count


def risk_weight_scale(current_coverage):
    """
    Coverage-pressure mode.

    Early in the run, risk avoidance is strong. Near the end, the risk
    penalty is intentionally relaxed so the planner does not keep avoiding
    the final high-risk frontier cells forever. This schedule is shared by
    all methods, so it improves completion fairness without favoring one FMEA.
    """
    if current_coverage < 0.75:
        return 1.00
    if current_coverage < 0.90:
        return 0.60
    if current_coverage < 0.97:
        return 0.25
    return 0.05


def planner_effective_risk(raw_risk):
    """
    Avoid over-penalizing low/moderate risk cells.
    Only the excess above PLANNER_RISK_FREE_MARGIN affects planning.
    This reduces unnecessary long detours.
    """
    return max(0.0, float(raw_risk) - PLANNER_RISK_FREE_MARGIN)

def choose_frontier_target(
    start,
    frontiers,
    visited,
    assigned_targets,
    damage_map,
    mission_map,
    h,
    w,
    current_coverage,
    rng,
):
    risk_scale = risk_weight_scale(current_coverage)

    shuffled = list(frontiers)
    rng.shuffle(shuffled)

    best_target = None
    best_score = -1e18

    for target in shuffled:
        r, c = target

        dist = manhattan(start, target)
        raw_d_risk = float(damage_map[r, c])
        raw_m_risk = float(mission_map[r, c])

        d_risk = planner_effective_risk(raw_d_risk)
        m_risk = planner_effective_risk(raw_m_risk)
        cluster = local_unvisited_cluster(target, visited, h, w)

        assigned_penalty = TARGET_ALREADY_ASSIGNED_PENALTY if target in assigned_targets else 0.0

        score = (
            TARGET_UNVISITED_BONUS_WEIGHT
            + TARGET_CLUSTER_BONUS_WEIGHT * cluster
            - TARGET_DISTANCE_WEIGHT * dist
            - risk_scale * TARGET_DAMAGE_WEIGHT * d_risk
            - risk_scale * TARGET_MISSION_WEIGHT * m_risk
            - assigned_penalty
        )

        if score > best_score:
            best_score = score
            best_target = target

    if best_target is None and frontiers:
        best_target = min(frontiers, key=lambda x: manhattan(start, x))

    return best_target


def astar_risk_path(start, goal, damage_map, mission_map, h, w, current_coverage):
    if start == goal:
        return [start]

    risk_scale = risk_weight_scale(current_coverage)

    damage_weight = BASE_ASTAR_DAMAGE_RISK_WEIGHT * risk_scale
    mission_weight = BASE_ASTAR_MISSION_RISK_WEIGHT * risk_scale

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))

    came_from = {}
    g_score = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()
            return path

        for nb in neighbors4(current, h, w):
            r, c = nb

            d_risk = planner_effective_risk(float(damage_map[r, c]))
            m_risk = planner_effective_risk(float(mission_map[r, c]))

            step_cost = (
                1.0
                + damage_weight * d_risk
                + mission_weight * m_risk
            )

            tentative_g = g_score[current] + step_cost

            if nb not in g_score or tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative_g
                f_score = tentative_g + manhattan(nb, goal)
                heapq.heappush(open_heap, (f_score, nb))

    return [start]


def resolve_move_conflicts(current_positions, proposed_positions, visited, damage_map, mission_map, h, w, current_coverage):
    """
    Prevent two robots from occupying the same next cell when a simple local
    alternative exists. This reduces artificial overlap and improves coverage
    fairly for every FMEA method.
    """
    risk_scale = risk_weight_scale(current_coverage)
    reserved = set()
    final_positions = []

    for old_pos, proposed in zip(current_positions, proposed_positions):
        # Do not second-guess a valid planned move. Only intervene when two
        # robots would occupy the same cell.
        if in_bounds(proposed, h, w) and proposed not in reserved:
            reserved.add(proposed)
            final_positions.append(proposed)
            continue

        candidates = neighbors4(old_pos, h, w) + [old_pos]
        unique_candidates = []
        for cell in candidates:
            if cell not in unique_candidates and in_bounds(cell, h, w):
                unique_candidates.append(cell)

        best_cell = old_pos
        best_score = -1e18

        for cell in unique_candidates:
            if cell in reserved:
                continue
            r, c = cell
            new_cell_bonus = 1.0 if cell not in visited else 0.0
            stay_penalty = 0.25 if cell == old_pos else 0.0
            d_risk = planner_effective_risk(float(damage_map[r, c]))
            m_risk = planner_effective_risk(float(mission_map[r, c]))
            score = (
                2.0 * new_cell_bonus
                - 0.10 * manhattan(old_pos, cell)
                - risk_scale * TARGET_DAMAGE_WEIGHT * d_risk
                - risk_scale * TARGET_MISSION_WEIGHT * m_risk
                - stay_penalty
            )
            if score > best_score:
                best_score = score
                best_cell = cell

        reserved.add(best_cell)
        final_positions.append(best_cell)

    return final_positions


# ============================================================
# SIMULATION
# ============================================================

def simulate_one_run(sequence, pack, scenario_id, run_seed, n_robots, render_path=False):
    rng = random.Random(10_000 * scenario_id + run_seed)

    h, w = get_grid_shape()
    total_cells = h * w

    positions = sample_start_positions(sequence, run_seed + 1000 * n_robots, n_robots)
    visited = set(positions)

    paths = {i: [positions[i]] for i in range(n_robots)}
    # Keep each robot committed to a frontier until it reaches it.
    # Without this, robots may oscillate between frontiers and coverage stalls.
    robot_targets = [None for _ in range(n_robots)]

    cumulative_damage = 0.0
    cumulative_catastrophic = 0.0
    cumulative_mission = 0.0

    risky_visits = 0
    high_risk_visits = 0
    overlap_count = 0

    coverage_history = []
    damage_history = []
    mission_history = []

    steps_to_target = None

    for t in range(1, MAX_COVERAGE_STEPS + 1):
        current_coverage = len(visited) / total_cells

        layers = layers_at_time(sequence, t)
        effect_cause_df = get_effect_cause_for_time(pack, t)

        # Planner uses method-specific estimated risk.
        est_damage_map, _, est_mission_map = build_estimated_risk_maps(
            layers,
            effect_cause_df,
        )

        # Evaluation uses common oracle risk.
        true_damage_map, true_cat_map, true_mission_map = build_oracle_risk_maps(
            layers
        )

        frontiers = get_frontier_cells(visited, h, w)
        assigned_targets = {
            tgt for tgt in robot_targets
            if tgt is not None and tgt not in visited
        }

        new_positions = []

        for rid, pos in enumerate(positions):
            # Reassign only when the previous target is already covered.
            target = robot_targets[rid]
            if target is None or target in visited:
                target = choose_frontier_target(
                    start=pos,
                    frontiers=frontiers,
                    visited=visited,
                    assigned_targets=assigned_targets,
                    damage_map=est_damage_map,
                    mission_map=est_mission_map,
                    h=h,
                    w=w,
                    current_coverage=current_coverage,
                    rng=rng,
                )
                robot_targets[rid] = target

            if target is None:
                next_pos = pos
            else:
                assigned_targets.add(target)

                path = astar_risk_path(
                    start=pos,
                    goal=target,
                    damage_map=est_damage_map,
                    mission_map=est_mission_map,
                    h=h,
                    w=w,
                    current_coverage=current_coverage,
                )

                if len(path) >= 2:
                    next_pos = path[1]
                else:
                    next_pos = pos

            new_positions.append(next_pos)

        new_positions = resolve_move_conflicts(
            current_positions=positions,
            proposed_positions=new_positions,
            visited=visited,
            damage_map=est_damage_map,
            mission_map=est_mission_map,
            h=h,
            w=w,
            current_coverage=current_coverage,
        )

        if len(set(new_positions)) < len(new_positions):
            overlap_count += len(new_positions) - len(set(new_positions))

        positions = new_positions

        for rid, pos in enumerate(positions):
            r, c = pos
            visited.add(pos)
            paths[rid].append(pos)

            d = float(true_damage_map[r, c])
            cat = float(true_cat_map[r, c])
            m = float(true_mission_map[r, c])

            cumulative_damage += d
            cumulative_catastrophic += cat
            cumulative_mission += m

            if d >= RISKY_CELL_THRESHOLD:
                risky_visits += 1

            if d >= HIGH_RISK_CELL_THRESHOLD:
                high_risk_visits += 1

        coverage = len(visited) / total_cells
        coverage_history.append(coverage)
        damage_history.append(cumulative_damage)
        mission_history.append(cumulative_mission)

        if coverage >= TARGET_COVERAGE:
            steps_to_target = t
            break

    steps_used = len(coverage_history)
    robot_steps = max(steps_used * n_robots, 1)

    row = {
    "Scenario": scenario_id,
    "Run Seed": run_seed,
    "Robots": n_robots,
    "Method": pack["method"],
    "Final Coverage (%)": round(100.0 * coverage_history[-1], 4),
    "Steps Used": steps_used,
    f"Steps to {int(TARGET_COVERAGE * 100)}% Coverage": (
        steps_to_target if steps_to_target is not None else "Not reached"
    ),
    "Cumulative Damage Risk": round(cumulative_damage, 6),
    "Mean Damage Risk / Robot-Step": round(cumulative_damage / robot_steps, 6),
    "Damage Risk / Visited Cell": round(cumulative_damage / max(len(visited), 1), 6),
    "Cumulative Catastrophic Damage Risk": round(cumulative_catastrophic, 6),
    "Cumulative Mission Failure Risk": round(cumulative_mission, 6),
    "Mean Mission Failure Risk / Robot-Step": round(cumulative_mission / robot_steps, 6),
    "Mission Risk / Visited Cell": round(cumulative_mission / max(len(visited), 1), 6),
    "Risky Cell Visits": risky_visits,
    "High-Risk Cell Visits": high_risk_visits,
    "Overlap Count": overlap_count,
    "Visited Cells": len(visited),
    "Total Cells": total_cells,
}

    history = pd.DataFrame({
        "Scenario": scenario_id,
        "Run Seed": run_seed,
        "Robots": n_robots,
        "Method": pack["method"],
        "Step": np.arange(1, steps_used + 1),
        "Coverage (%)": 100.0 * np.array(coverage_history),
        "Cumulative Damage Risk": damage_history,
        "Cumulative Mission Failure Risk": mission_history,
    })

    if render_path:
        plot_paths(paths, h, w, pack["method"], scenario_id, run_seed)

    return row, history


# ============================================================
# EVALUATION LOOP
# ============================================================

def evaluate_all():
    all_rows = []
    all_histories = []
    all_fmea_histories = []

    for scenario_idx, scenario_seed in enumerate(SCENARIO_SEEDS):
        scenario_id = scenario_idx + 1

        print("\n" + "=" * 100)
        print(f"RANDOMIZED MAP SCENARIO {scenario_id}/{N_RANDOM_MAPS}, seed={scenario_seed}")
        print("=" * 100)

        sequence = generate_random_hazard_sequence(scenario_seed)

        static_pack = build_static_pack_for_scenario(sequence)
        validate_static_pack(static_pack)

        no_llm_pack = build_dynamic_pack_for_scenario(
            sequence=sequence,
            use_llm=False,
            scenario_id=scenario_id,
        )

        llm_pack = build_dynamic_pack_for_scenario(
            sequence=sequence,
            use_llm=True,
            scenario_id=scenario_id,
        )

        packs = [
            static_pack,
            no_llm_pack,
            llm_pack,
        ]

        for n_robots in ROBOT_COUNTS:
            for pack in packs:
                print("\n" + "-" * 90)
                print(f"Planner: {pack['method']} | Scenario: {scenario_id} | Robots: {n_robots}")
                print("-" * 90)

                if "history" in pack:
                    hist_copy = pack["history"].copy()
                    hist_copy["Robots"] = n_robots
                    all_fmea_histories.append(hist_copy)

                for run_seed in RUN_SEEDS:
                    row, hist = simulate_one_run(
                        sequence=sequence,
                        pack=pack,
                        scenario_id=scenario_id,
                        run_seed=run_seed,
                        n_robots=n_robots,
                        render_path=(scenario_id == 1 and run_seed == RUN_SEEDS[0] and n_robots == ROBOT_COUNTS[0]),
                    )

                    all_rows.append(row)
                    all_histories.append(hist)

    seed_results_df = pd.DataFrame(all_rows)
    history_df = pd.concat(all_histories, ignore_index=True)

    if all_fmea_histories:
        fmea_history_df = pd.concat(all_fmea_histories, ignore_index=True)
    else:
        fmea_history_df = pd.DataFrame()

    return seed_results_df, history_df, fmea_history_df


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(seed_results_df):
    numeric_cols = [
        "Final Coverage (%)",
        "Steps Used",
        "Cumulative Damage Risk",
        "Mean Damage Risk / Robot-Step",
        "Damage Risk / Visited Cell",
        "Cumulative Catastrophic Damage Risk",
        "Cumulative Mission Failure Risk",
        "Mean Mission Failure Risk / Robot-Step",
        "Mission Risk / Visited Cell",
        "Risky Cell Visits",
        "High-Risk Cell Visits",
        "Overlap Count",
        "Visited Cells",
    ]

    summary = seed_results_df.groupby(["Robots", "Method"])[numeric_cols].agg(["mean", "std"])
    summary.columns = [f"{a} {b}" for a, b in summary.columns]
    summary = summary.reset_index()

    extra_rows = []

    for n_robots in sorted(seed_results_df["Robots"].unique()):
        block = summary[summary["Robots"] == n_robots]

        static = block[block["Method"] == "Static FMEA"].iloc[0]
        no_llm = block[block["Method"] == "Dynamic FMEA without LLM"].iloc[0]

        static_damage = float(static["Cumulative Damage Risk mean"])
        static_mission = float(static["Cumulative Mission Failure Risk mean"])
        static_risky = float(static["Risky Cell Visits mean"])
        static_cov = float(static["Final Coverage (%) mean"])

        no_llm_damage = float(no_llm["Cumulative Damage Risk mean"])
        no_llm_mission = float(no_llm["Cumulative Mission Failure Risk mean"])
        no_llm_risky = float(no_llm["Risky Cell Visits mean"])

        for _, row in block.iterrows():
            method = row["Method"]

            damage = float(row["Cumulative Damage Risk mean"])
            mission = float(row["Cumulative Mission Failure Risk mean"])
            risky = float(row["Risky Cell Visits mean"])
            cov = float(row["Final Coverage (%) mean"])

            extra_rows.append({
                "Robots": n_robots,
                "Method": method,

                "Damage Risk Reduction vs Static (%)": round(
                    100.0 * (static_damage - damage) / max(static_damage, 1e-9), 2
                ),
                "Mission Failure Risk Reduction vs Static (%)": round(
                    100.0 * (static_mission - mission) / max(static_mission, 1e-9), 2
                ),
                "Risky Cell Visit Reduction vs Static (%)": round(
                    100.0 * (static_risky - risky) / max(static_risky, 1e-9), 2
                ),
                "Coverage Change vs Static (%)": round(cov - static_cov, 2),

                "Damage Risk Reduction vs No-LLM (%)": round(
                    100.0 * (no_llm_damage - damage) / max(no_llm_damage, 1e-9), 2
                ),
                "Mission Failure Risk Reduction vs No-LLM (%)": round(
                    100.0 * (no_llm_mission - mission) / max(no_llm_mission, 1e-9), 2
                ),
                "Risky Cell Visit Reduction vs No-LLM (%)": round(
                    100.0 * (no_llm_risky - risky) / max(no_llm_risky, 1e-9), 2
                ),
            })

    extra_df = pd.DataFrame(extra_rows)
    summary = summary.merge(extra_df, on=["Robots", "Method"], how="left")

    cols = [
        "Robots",
        "Method",
        "Final Coverage (%) mean",
        "Final Coverage (%) std",
        "Steps Used mean",
        "Steps Used std",
        "Cumulative Damage Risk mean",
        "Cumulative Damage Risk std",
        "Mean Damage Risk / Robot-Step mean",
        "Damage Risk / Visited Cell mean",
        "Cumulative Catastrophic Damage Risk mean",
        "Cumulative Mission Failure Risk mean",
        "Cumulative Mission Failure Risk std",
        "Mean Mission Failure Risk / Robot-Step mean",
        "Mission Risk / Visited Cell mean",
        "Risky Cell Visits mean",
        "Risky Cell Visits std",
        "High-Risk Cell Visits mean",
        "Overlap Count mean",
        "Damage Risk Reduction vs Static (%)",
        "Mission Failure Risk Reduction vs Static (%)",
        "Risky Cell Visit Reduction vs Static (%)",
        "Coverage Change vs Static (%)",
        "Damage Risk Reduction vs No-LLM (%)",
        "Mission Failure Risk Reduction vs No-LLM (%)",
        "Risky Cell Visit Reduction vs No-LLM (%)",
    ]

    return summary[cols]


def summarize_per_scenario(seed_results_df):
    scenario_summary = (
        seed_results_df
        .groupby(["Robots", "Scenario", "Method"])
        .agg({
            "Final Coverage (%)": "mean",
            "Cumulative Damage Risk": "mean",
            "Cumulative Mission Failure Risk": "mean",
            "Risky Cell Visits": "mean",
            "Steps Used": "mean",
        })
        .reset_index()
    )

    return scenario_summary
# ============================================================
# PLOTS
# ============================================================

def _safe_method_name(name):
    return name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")


def plot_summary(summary_df):
    """
    Save one set of summary bar plots per robot count.
    This avoids mixing 3-, 5-, and 7-robot results on the same x-axis.
    """
    for n_robots in sorted(summary_df["Robots"].unique()):
        block = summary_df[summary_df["Robots"] == n_robots].copy()
        methods = block["Method"].tolist()

        plt.figure(figsize=(10, 5))
        plt.bar(methods, block["Final Coverage (%) mean"], yerr=block["Final Coverage (%) std"])
        plt.ylabel("Coverage (%)")
        plt.title(f"Coverage Comparison ({n_robots} robots)")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"coverage_comparison_{n_robots}_robots.pdf")
        plt.savefig(OUTPUT_DIR / f"coverage_comparison_{n_robots}_robots.png", dpi=300)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(methods, block["Cumulative Damage Risk mean"], yerr=block["Cumulative Damage Risk std"])
        plt.ylabel("Cumulative True Damage Risk")
        plt.title(f"True Damage-Risk Comparison ({n_robots} robots)")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"damage_risk_comparison_{n_robots}_robots.pdf")
        plt.savefig(OUTPUT_DIR / f"damage_risk_comparison_{n_robots}_robots.png", dpi=300)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(methods, block["Cumulative Mission Failure Risk mean"], yerr=block["Cumulative Mission Failure Risk std"])
        plt.ylabel("Cumulative True Mission-Failure Risk")
        plt.title(f"True Mission-Failure Risk Comparison ({n_robots} robots)")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"mission_failure_comparison_{n_robots}_robots.pdf")
        plt.savefig(OUTPUT_DIR / f"mission_failure_comparison_{n_robots}_robots.png", dpi=300)
        plt.show()

def plot_histories(history_df):
    """Save average progress plots separately for each robot count."""
    grouped = (
        history_df
        .groupby(["Robots", "Method", "Step"])
        .agg({
            "Coverage (%)": "mean",
            "Cumulative Damage Risk": "mean",
            "Cumulative Mission Failure Risk": "mean",
        })
        .reset_index()
    )

    for n_robots in sorted(grouped["Robots"].unique()):
        block = grouped[grouped["Robots"] == n_robots]

        plt.figure(figsize=(10, 5))
        for method in block["Method"].unique():
            df = block[block["Method"] == method]
            plt.plot(df["Step"], df["Coverage (%)"], label=method)
        plt.xlabel("Coverage Step")
        plt.ylabel("Coverage (%)")
        plt.title(f"Average Coverage Progress ({n_robots} robots)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"coverage_progress_{n_robots}_robots.pdf")
        plt.savefig(OUTPUT_DIR / f"coverage_progress_{n_robots}_robots.png", dpi=300)
        plt.show()

        plt.figure(figsize=(10, 5))
        for method in block["Method"].unique():
            df = block[block["Method"] == method]
            plt.plot(df["Step"], df["Cumulative Damage Risk"], label=method)
        plt.xlabel("Coverage Step")
        plt.ylabel("Cumulative True Damage Risk")
        plt.title(f"Average Damage-Risk Accumulation ({n_robots} robots)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"damage_risk_progress_{n_robots}_robots.pdf")
        plt.savefig(OUTPUT_DIR / f"damage_risk_progress_{n_robots}_robots.png", dpi=300)
        plt.show()

def plot_paths(paths, h, w, method, scenario_id, run_seed):
    plt.figure(figsize=(6.5, 6.5))

    for rid, path in paths.items():
        rr = [p[0] for p in path]
        cc = [p[1] for p in path]
        plt.plot(cc, rr, marker="o", markersize=2.0, linewidth=1.3, label=f"Robot {rid + 1}")

    plt.xlim(-0.5, w - 0.5)
    plt.ylim(h - 0.5, -0.5)
    plt.xticks(range(w))
    plt.yticks(range(h))
    plt.grid(True, alpha=0.3)
    plt.title(f"{method}, Scenario {scenario_id}, Seed {run_seed}")
    plt.legend()
    plt.tight_layout()

    safe = method.lower().replace(" ", "_").replace("+", "plus")
    plt.savefig(OUTPUT_DIR / f"path_{safe}_scenario_{scenario_id}_seed_{run_seed}.pdf")
    plt.savefig(OUTPUT_DIR / f"path_{safe}_scenario_{scenario_id}_seed_{run_seed}.png", dpi=300)
    plt.show()


# ============================================================
# LATEX EXPORT
# ============================================================

def export_latex(summary_df):
    latex_df = summary_df[
        [
            "Robots",
            "Method",
            "Final Coverage (%) mean",
            "Final Coverage (%) std",
            "Cumulative Damage Risk mean",
            "Cumulative Damage Risk std",
            "Cumulative Mission Failure Risk mean",
            "Cumulative Mission Failure Risk std",
            "Risky Cell Visits mean",
            "Damage Risk Reduction vs Static (%)",
            "Mission Failure Risk Reduction vs Static (%)",
            "Damage Risk Reduction vs No-LLM (%)",
            "Mission Failure Risk Reduction vs No-LLM (%)",
        ]
    ].copy()

    latex_df = latex_df.rename(columns={
        "Final Coverage (%) mean": "Coverage Mean (\\%)",
        "Final Coverage (%) std": "Coverage Std",
        "Cumulative Damage Risk mean": "Damage Risk Mean",
        "Cumulative Damage Risk std": "Damage Risk Std",
        "Cumulative Mission Failure Risk mean": "Mission Risk Mean",
        "Cumulative Mission Failure Risk std": "Mission Risk Std",
        "Risky Cell Visits mean": "Risky Visits",
        "Damage Risk Reduction vs Static (%)": "Damage Red. vs Static (\\%)",
        "Mission Failure Risk Reduction vs Static (%)": "Mission Red. vs Static (\\%)",
        "Damage Risk Reduction vs No-LLM (%)": "Damage Red. vs No-LLM (\\%)",
        "Mission Failure Risk Reduction vs No-LLM (%)": "Mission Red. vs No-LLM (\\%)",
    })

    tex = latex_df.to_latex(
        index=False,
        float_format="%.3f",
        escape=False,
        caption=(
            "Risk-aware Frontier A* coverage comparison over 10 randomized "
            "hazard-map sequences, 5 random seeds per sequence, and robot counts "
            "$N \\in \\{3,5,7\\}$. Static FMEA keeps all tables fixed. Dynamic "
            "methods update only the cause-to-mode table $P(F|C)$. The LLM is used "
            "only in the Dynamic FMEA + LLM method for the $P(F|C)$ update, while "
            "$P(E|F)$ remains fixed for all methods. All paths are evaluated using "
            "the same oracle risk model."
        ),
        label="tab:randomized_astar_fmea_comparison",
    )

    with open(OUTPUT_DIR / "randomized_astar_comparison_table.tex", "w", encoding="utf-8") as f:
        f.write(tex)

    return tex


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nStarting randomized hazard-map Frontier A* evaluation...")
    print(f"Randomized maps: {N_RANDOM_MAPS}")
    print(f"Run seeds per map: {len(RUN_SEEDS)}")
    print(f"Runs per method: {N_RANDOM_MAPS * len(RUN_SEEDS)}")
    print(f"Robot counts: {ROBOT_COUNTS}")
    print(f"Total planner runs: {3 * len(ROBOT_COUNTS) * N_RANDOM_MAPS * len(RUN_SEEDS)}")
    print(f"Target coverage: {TARGET_COVERAGE * 100:.1f}%")
    print(f"Max coverage steps: {MAX_COVERAGE_STEPS}")

    seed_results_df, history_df, fmea_history_df = evaluate_all()

    summary_df = summarize_results(seed_results_df)
    scenario_summary_df = summarize_per_scenario(seed_results_df)

    seed_results_df.to_csv(OUTPUT_DIR / "all_seed_results_randomized.csv", index=False)
    history_df.to_csv(OUTPUT_DIR / "coverage_history_randomized.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary_randomized_astar.csv", index=False)
    scenario_summary_df.to_csv(OUTPUT_DIR / "per_scenario_summary_randomized.csv", index=False)

    if not fmea_history_df.empty:
        fmea_history_df.to_csv(OUTPUT_DIR / "fmea_update_history_randomized.csv", index=False)

    print("\n" + "=" * 120)
    print("FINAL RANDOMIZED FRONTIER A* COMPARISON TABLE")
    print("=" * 120)
    print(summary_df.round(4).to_string(index=False))

    print("\n" + "=" * 120)
    print("PER-SCENARIO SUMMARY")
    print("=" * 120)
    print(scenario_summary_df.round(4).to_string(index=False))

    tex = export_latex(summary_df)

    print("\n" + "=" * 120)
    print("LATEX TABLE")
    print("=" * 120)
    print(tex)

    plot_summary(summary_df)
    plot_histories(history_df)

    print("\nSaved outputs in:")
    print(OUTPUT_DIR.resolve())

    print("\nImportant files:")
    print("1. summary_randomized_astar.csv")
    print("2. all_seed_results_randomized.csv")
    print("3. per_scenario_summary_randomized.csv")
    print("4. coverage_history_randomized.csv")
    print("5. randomized_astar_comparison_table.tex")
    print("6. damage_risk_comparison_<robots>_robots.pdf")
    print("7. mission_failure_comparison_<robots>_robots.pdf")
    print("8. coverage_comparison_<robots>_robots.pdf")
    print("9. coverage_progress_<robots>_robots.pdf")
    print("10. damage_risk_progress_<robots>_robots.pdf")


if __name__ == "__main__":
    main()