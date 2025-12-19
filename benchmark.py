#!/usr/bin/env python3
"""
LLM Bias Benchmark
==================
Evaluates LLM calibration and decision bias across 5 experiments.

Usage:
    python benchmark.py --model MODEL_NAME --api-key YOUR_API_KEY

Example:
    python benchmark.py --model "google/gemini-2.5-pro" --api-key "sk-or-v1-xxx"

For multiple API keys (load balancing):
    python benchmark.py --model "google/gemini-2.5-pro" --api-keys "key1,key2,key3"

Dependencies:
    pip install aiohttp pandas numpy tqdm nest_asyncio
"""

import aiohttp
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm as tqdm_async
import datetime
import os
import re
import json
import argparse
from typing import List, Tuple
from dataclasses import dataclass

nest_asyncio.apply()

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    api_keys: List[str]
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    model_name: str = "openai/gpt-4"
    max_connections: int = 200
    timeout: int = 200
    max_retries: int = 3
    save_root: str = "./results"

N_TRIALS = 100  # Fixed N per condition

# =============================================================================
# Scoring Functions
# =============================================================================

def compute_S(p_vals: np.ndarray, r_vals: np.ndarray) -> float:
    """
    Compute step-likeness score S = 4 × ∫|r(p) - p| dp
    
    S = 0: Perfect calibration (r = p)
    S = 1: Perfect step function
    """
    p_vals = np.array(p_vals)
    r_vals = np.array(r_vals)
    
    if p_vals.max() > 1:
        p_vals = p_vals / 100
        r_vals = r_vals / 100
    
    # Extrapolate to [0, 1]
    if p_vals[0] > 0:
        p_vals = np.insert(p_vals, 0, 0)
        r_vals = np.insert(r_vals, 0, r_vals[0])
    if p_vals[-1] < 1:
        p_vals = np.append(p_vals, 1)
        r_vals = np.append(r_vals, r_vals[-1])
    
    # Trapezoidal integration
    dp = np.diff(p_vals)
    L = np.abs(r_vals[:-1] - p_vals[:-1])
    R = np.abs(r_vals[1:] - p_vals[1:])
    area = np.sum(dp * (L + R) / 2)
    
    return 4.0 * area

def compute_S_diagonal(p_exp: np.ndarray, r_obs: np.ndarray) -> float:
    """Compute S for r vs y=x diagonal (normalized to [0,1])"""
    idx = np.argsort(p_exp)
    p = p_exp[idx]
    r = r_obs[idx]
    
    dp = np.diff(p)
    deviation_left = np.abs(r[:-1] - p[:-1])
    deviation_right = np.abs(r[1:] - p[1:])
    area = np.sum(dp * (deviation_left + deviation_right) / 2)
    
    x_min, x_max = p[0], p[-1]
    x_range = x_max - x_min
    if x_range == 0:
        return 0.0
    
    area_r1 = x_range * (1 - (x_max + x_min) / 2)
    area_r0 = (x_max + x_min) * x_range / 2
    max_area = max(area_r0, area_r1)
    
    return min(area / max_area, 1.0) if max_area > 0 else 0.0

def compute_S_horizontal(x_vals: np.ndarray, r_obs: np.ndarray, expected_y: float) -> float:
    """Compute S for r vs y=expected_y horizontal line (normalized to [0,1])"""
    idx = np.argsort(x_vals)
    x = x_vals[idx]
    r = r_obs[idx]
    
    dx = np.diff(x)
    deviation_left = np.abs(r[:-1] - expected_y)
    deviation_right = np.abs(r[1:] - expected_y)
    area = np.sum(dx * (deviation_left + deviation_right) / 2)
    
    x_range = x[-1] - x[0]
    if x_range == 0:
        return 0.0
    
    max_deviation = max(expected_y, 1 - expected_y)
    max_area = x_range * max_deviation
    
    return min(area / max_area, 1.0) if max_area > 0 else 0.0

def pairwise_direction_score(p_vals: np.ndarray, r_vals: np.ndarray) -> float:
    """Compute direction score: +1 if r increases with p, 0 if flat, -1 if decreases"""
    idx = np.argsort(p_vals)
    r_sorted = np.array(r_vals)[idx]
    
    directions = []
    for i in range(len(r_sorted) - 1):
        diff = r_sorted[i + 1] - r_sorted[i]
        if diff > 0.01:
            directions.append(1)
        elif diff < -0.01:
            directions.append(-1)
        else:
            directions.append(0)
    
    raw = np.mean(directions) if directions else 0
    return (raw + 1) / 2  # Normalize to [0, 1]

# =============================================================================
# Scoring System
# =============================================================================

@dataclass
class ScoreResult:
    test_name: str
    metric_name: str
    metric_value: float
    score: float
    max_score: float = 20.0
    details: dict = None

class BenchmarkScorer:
    @staticmethod
    def score_exp1(S: float) -> float:
        """Exp1: Score = max(0, 20 × (1 - S))"""
        return max(0, 20 * (1 - S))
    
    @staticmethod
    def score_exp2(S2: float, S_avg: float) -> Tuple[float, float, float]:
        """Exp2: Score_S2 (10 pts) + Score_Savg (10 pts)"""
        score_S2 = max(0, 10 * (1 - S2))
        score_Savg = max(0, 10 * (1 - S_avg))
        return score_S2, score_Savg, score_S2 + score_Savg
    
    @staticmethod
    def score_exp3(S0: float, S1: float, S2: float) -> Tuple[float, float, float, float]:
        """Exp3: Each S scored separately (20/3 pts each)"""
        score_S0 = max(0, 1 - S0) * (20/3)
        score_S1 = max(0, 1 - S1) * (20/3)
        score_S2 = max(0, 1 - S2) * (20/3)
        return score_S0, score_S1, score_S2, score_S0 + score_S1 + score_S2
    
    @staticmethod
    def score_exp4_bioinformatics(S: float) -> float:
        """Exp4 Bioinformatics: max 8 pts based on S"""
        return max(0, 1 - S) * 8
    
    @staticmethod
    def score_exp4_matching_pennies(S: float, dir_score: float) -> Tuple[float, float, float]:
        """Exp4 MatchingPennies: S (4 pts) + direction (4 pts)"""
        score_S = max(0, 1 - S) * 4
        score_dir = dir_score * 4
        return score_S, score_dir, score_S + score_dir
    
    @staticmethod
    def score_exp4_luxury(dir_score: float) -> float:
        """Exp4 Luxury: direction only (4 pts)"""
        return dir_score * 4
    
    @staticmethod
    def score_exp5(avg_pos_bias: float, avg_red_bias: float) -> Tuple[float, float, float]:
        """
        Exp5: Position bias (10 pts) + Semantic bias (10 pts)
        """
        score_pos = max(0, 20 * (1 - 2 * avg_pos_bias)) / 2
        score_sem = max(0, 20 * (1 - 2.22 * avg_red_bias)) / 2
        return score_pos, score_sem, score_pos + score_sem

# =============================================================================
# API Caller
# =============================================================================

class ModelCaller:
    def __init__(self, config: Config):
        self.config = config
        self.key_idx = 0
    
    def get_next_key(self) -> str:
        key = self.config.api_keys[self.key_idx % len(self.config.api_keys)]
        self.key_idx += 1
        return key
    
    async def call(self, session: aiohttp.ClientSession, prompt: str, 
                   api_key: str, attempt: int = 1) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt.strip()}]
        }
        
        try:
            async with session.post(self.config.api_url, json=payload, 
                                    headers=headers, timeout=self.config.timeout) as r:
                if r.status == 429:
                    if attempt <= self.config.max_retries:
                        await asyncio.sleep(2 ** attempt)
                        return await self.call(session, prompt, api_key, attempt + 1)
                    return ""
                
                data = await r.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            if attempt <= self.config.max_retries:
                await asyncio.sleep(1)
                return await self.call(session, prompt, api_key, attempt + 1)
            return ""

# =============================================================================
# Extraction Functions
# =============================================================================

def extract_binary(text: str) -> str:
    if not text:
        return "error"
    m = re.findall(r"\b([01])\b", text)
    return m[-1] if m else "error"

def extract_two_flips(text: str) -> Tuple[str, str]:
    if not text:
        return ("error", "error")
    m = re.findall(r"\b([01])\b", text)
    if len(m) < 2:
        return ("error", "error")
    return (m[0], m[1])

def extract_ternary(text: str) -> str:
    if not text:
        return "error"
    m = re.findall(r"\b([012])\b", text)
    return m[-1] if m else "error"

def extract_choice(text: str) -> str:
    if not text:
        return "error"
    t = text.lower()
    if "gene a" in t or "option a" in t or t.strip().endswith(" a"):
        return "A"
    if "gene b" in t or "option b" in t or t.strip().endswith(" b"):
        return "B"
    if "heads" in t:
        return "Heads"
    if "tails" in t:
        return "Tails"
    if "luxury" in t:
        return "Luxury"
    if "affordable" in t:
        return "Affordable"
    return "error"

def extract_pair_choice(text: str, v: str, v2: str) -> str:
    if not text:
        return "error"
    t = text.lower()
    v_low, v2_low = v.lower(), v2.lower()
    
    if v_low in t and v2_low not in t:
        return v
    if v2_low in t and v_low not in t:
        return v2
    
    pos_v = t.rfind(v_low)
    pos_v2 = t.rfind(v2_low)
    if pos_v > pos_v2:
        return v
    if pos_v2 > pos_v:
        return v2
    return "error"

# =============================================================================
# Experiments
# =============================================================================

class Exp1_SingleFlip:
    """Experiment 1: Single binary flip calibration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.caller = ModelCaller(config)
        self.p_list = list(range(0, 105, 5))
        self.prompt_template = (
            'I\'d like you to flip a biased coin that returns "1" {p}% of the time '
            'and "0" {q}% of the time. Please return a 1 token response simulating a single flip.'
        )
    
    async def run(self) -> ScoreResult:
        tasks, task_meta = [], []
        
        connector = aiohttp.TCPConnector(limit=self.config.max_connections)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for p in self.p_list:
                prompt = self.prompt_template.replace("{p}", str(p)).replace("{q}", str(100 - p))
                for _ in range(N_TRIALS):
                    key = self.caller.get_next_key()
                    tasks.append(asyncio.create_task(self.caller.call(session, prompt, key)))
                    task_meta.append((p, prompt))
            
            print(f"[Exp1] Total API calls: {len(tasks)}")
            results = await tqdm_async.gather(*tasks, desc="Exp1")
        
        rows = [{"p": p, "raw": text, "answer": extract_binary(text)} 
                for (p, _), text in zip(task_meta, results)]
        self.results_df = pd.DataFrame(rows)
        
        # Compute S
        df_valid = self.results_df[self.results_df["answer"].isin(["0", "1"])]
        summary = df_valid.groupby(["p", "answer"]).size().unstack(fill_value=0).reset_index()
        for col in ["0", "1"]:
            if col not in summary.columns:
                summary[col] = 0
        summary["total"] = summary["0"] + summary["1"]
        summary["r"] = summary["1"] / summary["total"]
        self.summary_df = summary
        
        S = compute_S(summary["p"], summary["r"] * 100)
        score = BenchmarkScorer.score_exp1(S)
        
        return ScoreResult("Exp1_SingleFlip", "S", S, score, details={"S": f"{S:.4f}"})


class Exp2_TwoFlips:
    """Experiment 2: Two flips per call"""
    
    def __init__(self, config: Config):
        self.config = config
        self.caller = ModelCaller(config)
        self.p_list = list(range(0, 105, 5))
        self.prompt_template = (
            'I\'d like you to flip a biased coin that returns "1" {p}% of the time '
            'and "0" {q}% of the time. Please flip it twice and return EXACTLY 2 tokens '
            '(separated by comma) simulating two independent flips.'
        )
    
    async def run(self) -> ScoreResult:
        tasks, task_meta = [], []
        
        connector = aiohttp.TCPConnector(limit=self.config.max_connections)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for p in self.p_list:
                prompt = self.prompt_template.replace("{p}", str(p)).replace("{q}", str(100 - p))
                for _ in range(N_TRIALS):
                    key = self.caller.get_next_key()
                    tasks.append(asyncio.create_task(self.caller.call(session, prompt, key)))
                    task_meta.append((p, prompt))
            
            print(f"[Exp2] Total API calls: {len(tasks)}")
            results = await tqdm_async.gather(*tasks, desc="Exp2")
        
        rows = []
        for (p, _), text in zip(task_meta, results):
            a1, a2 = extract_two_flips(text)
            rows.append({"p": p, "raw": text, "answer1": a1, "answer2": a2})
        self.results_df = pd.DataFrame(rows)
        
        # Compute S values
        df = self.results_df
        df_j1 = df[df["answer1"].isin(["0", "1"])]
        df_j2 = df[df["answer2"].isin(["0", "1"])]
        
        sum_j1 = df_j1.groupby("p")["answer1"].apply(lambda x: (x == "1").mean()).reset_index()
        sum_j1.columns = ["p", "r"]
        sum_j2 = df_j2.groupby("p")["answer2"].apply(lambda x: (x == "1").mean()).reset_index()
        sum_j2.columns = ["p", "r"]
        
        S1 = compute_S(sum_j1["p"], sum_j1["r"] * 100)
        S2 = compute_S(sum_j2["p"], sum_j2["r"] * 100)
        
        df["mean_r"] = df.apply(lambda row: np.mean([
            1 if row["answer1"] == "1" else 0 if row["answer1"] == "0" else np.nan,
            1 if row["answer2"] == "1" else 0 if row["answer2"] == "0" else np.nan
        ]), axis=1)
        sum_avg = df.groupby("p")["mean_r"].mean().reset_index()
        S_avg = compute_S(sum_avg["p"], sum_avg["mean_r"] * 100)
        
        self.summary_df = pd.DataFrame({"p": sum_j1["p"], "r_j1": sum_j1["r"], 
                                        "r_j2": sum_j2["r"], "r_avg": sum_avg["mean_r"]})
        
        score_S2, score_Savg, total_score = BenchmarkScorer.score_exp2(S2, S_avg)
        
        return ScoreResult("Exp2_TwoFlips", "S2+S_avg", (S2 + S_avg) / 2, total_score,
                          details={"S1": f"{S1:.4f}", "S2": f"{S2:.4f}", "S_avg": f"{S_avg:.4f}",
                                  "score_S2": f"{score_S2:.2f}", "score_Savg": f"{score_Savg:.2f}"})


class Exp3_ThreeOutcome:
    """Experiment 3: Three-outcome distribution (3 outcomes)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.caller = ModelCaller(config)
        # P(1) fixed at 40%, P(0) + P(2) = 60%
        self.conditions = [
            (60, 40, 0), (50, 40, 10), (40, 40, 20), (30, 40, 30),
            (20, 40, 40), (10, 40, 50), (0, 40, 60)
        ]
        self.prompt_template = (
            'I\'d like you to draw a number from a distribution: "0" with {p0}% probability, '
            '"1" with {p1}% probability, "2" with {p2}% probability. '
            'Please return a 1 token response simulating a single draw.'
        )
    
    async def run(self) -> ScoreResult:
        tasks, task_meta = [], []
        
        connector = aiohttp.TCPConnector(limit=self.config.max_connections)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for (p0, p1, p2) in self.conditions:
                prompt = (self.prompt_template
                         .replace("{p0}", str(p0))
                         .replace("{p1}", str(p1))
                         .replace("{p2}", str(p2)))
                for _ in range(N_TRIALS):
                    key = self.caller.get_next_key()
                    tasks.append(asyncio.create_task(self.caller.call(session, prompt, key)))
                    task_meta.append((p0, p1, p2))
            
            print(f"[Exp3] Total API calls: {len(tasks)}")
            results = await tqdm_async.gather(*tasks, desc="Exp3")
        
        rows = [{"p0": p0, "p1": p1, "p2": p2, "raw": text, "answer": extract_ternary(text)} 
                for (p0, p1, p2), text in zip(task_meta, results)]
        self.results_df = pd.DataFrame(rows)
        
        # Compute S0, S1, S2
        df = self.results_df[self.results_df["answer"].isin(["0", "1", "2"])]
        summary = df.groupby(["p0", "p1", "p2", "answer"]).size().unstack(fill_value=0).reset_index()
        for col in ["0", "1", "2"]:
            if col not in summary.columns:
                summary[col] = 0
        summary["total"] = summary["0"] + summary["1"] + summary["2"]
        summary["r0"] = summary["0"] / summary["total"]
        summary["r1"] = summary["1"] / summary["total"]
        summary["r2"] = summary["2"] / summary["total"]
        self.summary_df = summary
        
        p0_vals = summary["p0"].values / 100
        p2_vals = summary["p2"].values / 100
        r0_vals = summary["r0"].values
        r1_vals = summary["r1"].values
        r2_vals = summary["r2"].values
        
        S0 = compute_S_diagonal(p0_vals, r0_vals)
        S1 = compute_S_horizontal(p2_vals, r1_vals, 0.4)
        S2 = compute_S_diagonal(p2_vals, r2_vals)
        
        score_S0, score_S1, score_S2, total = BenchmarkScorer.score_exp3(S0, S1, S2)
        
        return ScoreResult("Exp3_ThreeOutcome", "S0+S1+S2", (S0 + S1 + S2) / 3, total,
                          details={"S0": f"{S0:.4f}", "S1": f"{S1:.4f}", "S2": f"{S2:.4f}",
                                  "score_S0": f"{score_S0:.2f}", "score_S1": f"{score_S1:.2f}",
                                  "score_S2": f"{score_S2:.2f}"})


class Exp4_DecisionBias:
    """Experiment 5: Multi-scenario decision problems"""
    
    SCENARIOS = {
        "Bioinformatics": {
            "prompt": (
                'You are assisting with a bioinformatics task. A read R aligns equally well '
                'to genes A and B. So far, {a} reads mapped to A, and {b} reads mapped to B. '
                'You must choose A or B. Respond with ONLY "A" or "B".'
            ),
            "p_list": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "choice_map": {"A": 1, "B": 0},
            "scoring": "S",
            "max_score": 4
        },
        "MatchingPennies": {
            "prompt": (
                'You are playing Matching Pennies. If you match as Heads, you win ${h}. '
                'If you match as Tails, you win ${t}. Mismatches lose the same amounts. '
                'Respond with ONLY "Heads" or "Tails".'
            ),
            "p_list": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "choice_map": {"Heads": 1, "Tails": 0},
            "scoring": "S+direction",
            "max_score_S": 2,
            "max_score_dir": 2
        },
        "Luxury": {
            "prompt": (
                'You are a startup CEO. If you choose Luxury and competitor chooses Affordable, '
                'you gain ${lux}M. If you choose Affordable and competitor chooses Luxury, '
                'you gain ${aff}M. Matching choices = $0. Respond with ONLY "Luxury" or "Affordable".'
            ),
            "p_list": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "choice_map": {"Luxury": 1, "Affordable": 0},
            "scoring": "direction",
            "max_score": 2
        }
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.caller = ModelCaller(config)
        self.all_results = []
    
    async def run(self) -> ScoreResult:
        total_score = 0
        details = {}
        
        for name, scenario in self.SCENARIOS.items():
            df, S, dir_score = await self._run_scenario(name, scenario)
            self.all_results.append(df)
            
            if scenario["scoring"] == "S":
                score = BenchmarkScorer.score_exp4_bioinformatics(S)
                details[f"{name}_S"] = f"{S:.3f}"
                details[f"{name}_score"] = f"{score:.2f}"
                total_score += score
            elif scenario["scoring"] == "S+direction":
                score_S, score_dir, score = BenchmarkScorer.score_exp4_matching_pennies(S, dir_score)
                details[f"{name}_S"] = f"{S:.3f}"
                details[f"{name}_dir"] = f"{dir_score:.3f}"
                details[f"{name}_score_S"] = f"{score_S:.2f}/2"
                details[f"{name}_score_dir"] = f"{score_dir:.2f}/2"
                total_score += score
            else:  # direction only
                score = BenchmarkScorer.score_exp4_luxury(dir_score)
                details[f"{name}_dir"] = f"{dir_score:.3f}"
                details[f"{name}_score"] = f"{score:.2f}"
                total_score += score
        
        self.results_df = pd.concat(self.all_results, ignore_index=True)
        
        return ScoreResult("Exp4_DecisionBias", "combined", total_score / 10, total_score,
                          details=details)
    
    async def _run_scenario(self, name, scenario):
        tasks, task_meta = [], []
        
        connector = aiohttp.TCPConnector(limit=self.config.max_connections)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for p in scenario["p_list"]:
                if name == "Bioinformatics":
                    a, b = int(p * 1000), int((1 - p) * 1000)
                    prompt = scenario["prompt"].replace("{a}", str(a)).replace("{b}", str(b))
                elif name == "MatchingPennies":
                    h, t = p * 3, (1 - p) * 3
                    prompt = scenario["prompt"].replace("{h}", f"{h:.2f}").replace("{t}", f"{t:.2f}")
                else:  # Luxury
                    lux, aff = p * 20, (1 - p) * 20
                    prompt = scenario["prompt"].replace("{lux}", f"{lux:.0f}").replace("{aff}", f"{aff:.0f}")
                
                for _ in range(N_TRIALS):
                    key = self.caller.get_next_key()
                    tasks.append(asyncio.create_task(self.caller.call(session, prompt, key)))
                    task_meta.append((p, prompt))
            
            print(f"[Exp4-{name}] API calls: {len(tasks)}")
            results = await tqdm_async.gather(*tasks, desc=f"Exp4-{name[:10]}")
        
        rows = [{"scenario": name, "p": p, "raw": text, "choice": extract_choice(text)} 
                for (p, _), text in zip(task_meta, results)]
        df = pd.DataFrame(rows)
        
        choice_map = scenario["choice_map"]
        df["bin"] = df["choice"].map(lambda x: choice_map.get(x, np.nan))
        df_valid = df[~df["bin"].isna()]
        
        # Handle empty results
        if len(df_valid) == 0:
            print(f"    Warning: No valid responses for {name}")
            return df, 2.0, 0.5  # Worst S, neutral direction
        
        summary = df_valid.groupby("p")["bin"].mean().reset_index()
        
        if len(summary) < 2:
            print(f"    Warning: Insufficient data for {name}")
            return df, 2.0, 0.5
        
        S = compute_S(summary["p"] * 100, summary["bin"] * 100)
        dir_score = pairwise_direction_score(summary["p"].values, summary["bin"].values)
        
        return df, S, dir_score


class Exp5_SemanticPairs:
    """Experiment 6: Word choice and position bias"""
    
    PAIRS = [
        ("37", "50"), ("day", "night"), ("human", "alien"), ("rich", "poor"), ("woman", "man"),
        ("cat", "dog"), ("black", "white"), ("sun", "moon"), ("adult", "child"), ("1", "0"),
        ("Luxury", "Affordable"),
        # Reversed pairs
        ("50", "37"), ("night", "day"), ("alien", "human"), ("poor", "rich"), ("man", "woman"),
        ("dog", "cat"), ("white", "black"), ("moon", "sun"), ("child", "adult"), ("0", "1"),
        ("Affordable", "Luxury")
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self.caller = ModelCaller(config)
        self.prompt_template = (
            'I\'d like you to flip a biased coin that returns "{v}" {p}% of the time '
            'and "{v2}" {q}% of the time. Please return a 1 token response simulating a single flip.'
        )
    
    async def run(self) -> ScoreResult:
        tasks, task_meta = [], []
        p = 50  # Fixed at 50%
        
        connector = aiohttp.TCPConnector(limit=self.config.max_connections)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for (v, v2) in self.PAIRS:
                prompt = (self.prompt_template
                         .replace("{v}", v).replace("{v2}", v2)
                         .replace("{p}", str(p)).replace("{q}", str(100 - p)))
                for _ in range(N_TRIALS):
                    key = self.caller.get_next_key()
                    tasks.append(asyncio.create_task(self.caller.call(session, prompt, key)))
                    task_meta.append((v, v2))
            
            print(f"[Exp5] Total API calls: {len(tasks)}")
            results = await tqdm_async.gather(*tasks, desc="Exp5")
        
        rows = [{"v": v, "v2": v2, "raw": text, "choice": extract_pair_choice(text, v, v2)} 
                for (v, v2), text in zip(task_meta, results)]
        self.results_df = pd.DataFrame(rows)
        
        # Compute biases
        bias_first_values = []
        red_bias_values = []
        
        n_unique = len(self.PAIRS) // 2
        forward_pairs = self.PAIRS[:n_unique]
        reverse_pairs = self.PAIRS[n_unique:]
        
        for i in range(n_unique):
            v_fwd, v2_fwd = forward_pairs[i]
            v_rev, v2_rev = reverse_pairs[i]
            
            # Forward pair
            df_fwd = self.results_df[(self.results_df["v"] == v_fwd) & (self.results_df["v2"] == v2_fwd)]
            df_fwd_valid = df_fwd[df_fwd["choice"].isin([v_fwd, v2_fwd])]
            if len(df_fwd_valid) > 0:
                r_fwd = (df_fwd_valid["choice"] == v_fwd).mean()
            else:
                r_fwd = 0.5
            
            # Reverse pair
            df_rev = self.results_df[(self.results_df["v"] == v_rev) & (self.results_df["v2"] == v2_rev)]
            df_rev_valid = df_rev[df_rev["choice"].isin([v_rev, v2_rev])]
            if len(df_rev_valid) > 0:
                r_rev = (df_rev_valid["choice"] == v_rev).mean()
            else:
                r_rev = 0.5
            
            # Position bias (first position preference)
            bias_first_fwd = r_fwd - 0.5
            bias_first_rev = r_rev - 0.5
            bias_first_values.extend([bias_first_fwd, bias_first_rev])
            
            # Semantic bias
            r_word_fwd = r_fwd
            r_word_rev = 1 - r_rev
            avg_bias = (r_word_fwd + r_word_rev) / 2
            red_bias = max(0, abs(avg_bias - 0.5) - 0.05)
            red_bias_values.append(red_bias)
        
        avg_pos_bias = np.mean(np.abs(bias_first_values))
        avg_red_bias = np.mean(red_bias_values)
        
        score_pos, score_sem, total = BenchmarkScorer.score_exp5(avg_pos_bias, avg_red_bias)
        
        return ScoreResult("Exp5_SemanticPairs", "avg_position_bias", avg_pos_bias, total,
                          details={"score_position": f"{score_pos:.2f}",
                                  "score_semantic": f"{score_sem:.2f}",
                                  "avg_position_bias": f"{avg_pos_bias:.4f}",
                                  "avg_semantic_bias": f"{avg_red_bias:.4f}"})

# =============================================================================
# Main Benchmark Runner
# =============================================================================

async def run_benchmark(config: Config):
    """Run all experiments and return results"""
    results = []
    
    print(f"\n{'='*60}")
    print(f"LLM Bias Benchmark")
    print(f"Model: {config.model_name}")
    print(f"N per condition: {N_TRIALS}")
    print(f"{'='*60}\n")
    
    # Exp1
    print("\n>>> Experiment 1: Single Flip")
    exp1 = Exp1_SingleFlip(config)
    r1 = await exp1.run()
    results.append(r1)
    print(f"    S = {r1.metric_value:.4f}, Score = {r1.score:.2f}/10")
    
    # Exp2
    print("\n>>> Experiment 2: Two Flips")
    exp2 = Exp2_TwoFlips(config)
    r2 = await exp2.run()
    results.append(r2)
    print(f"    Score = {r2.score:.2f}/10")
    
    # Exp3
    print("\n>>> Experiment 3: Three-Outcome Draw")
    exp3 = Exp3_ThreeOutcome(config)
    r3 = await exp3.run()
    results.append(r3)
    print(f"    Score = {r3.score:.2f}/10")
    
    # Exp4
    print("\n>>> Experiment 5: Decision Bias")
    exp4 = Exp4_DecisionBias(config)
    r5 = await exp4.run()
    results.append(r5)
    print(f"    Score = {r5.score:.2f}/10")
    
    # Exp5
    print("\n>>> Experiment 6: Semantic Pairs")
    exp5 = Exp5_SemanticPairs(config)
    r6 = await exp5.run()
    results.append(r6)
    print(f"    Score = {r6.score:.2f}/10")
    
    # Summary
    total = sum(r.score for r in results)
    max_total = sum(r.max_score for r in results)
    
    print(f"\n{'='*60}")
    print(f"TOTAL SCORE: {total:.2f} / {max_total:.0f} ({100*total/max_total:.1f}%)")
    print(f"{'='*60}")
    
    # Save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = config.model_name.replace("/", "_")
    save_dir = os.path.join(config.save_root, f"{model_safe}_{ts}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save raw data
    exp1.results_df.to_csv(os.path.join(save_dir, "exp1_raw.csv"), index=False)
    exp2.results_df.to_csv(os.path.join(save_dir, "exp2_raw.csv"), index=False)
    exp3.results_df.to_csv(os.path.join(save_dir, "exp3_raw.csv"), index=False)
    exp4.results_df.to_csv(os.path.join(save_dir, "exp4_raw.csv"), index=False)
    exp5.results_df.to_csv(os.path.join(save_dir, "exp5_raw.csv"), index=False)
    
    # Save summary
    summary = {
        "model": config.model_name,
        "timestamp": ts,
        "total_score": total,
        "max_score": max_total,
        "percentage": 100 * total / max_total,
        "experiments": {r.test_name: {"score": r.score, "metric": r.metric_value, 
                                       "details": r.details} for r in results}
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")
    
    return results, summary

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Bias Benchmark")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Model name (e.g., google/gemini-2.5-pro)")
    parser.add_argument("--api-key", "-k", type=str,
                       help="Single API key")
    parser.add_argument("--api-keys", type=str,
                       help="Multiple API keys (comma-separated)")
    parser.add_argument("--save-dir", "-s", type=str, default="./results",
                       help="Output directory")
    args = parser.parse_args()
    
    if args.api_keys:
        api_keys = [k.strip() for k in args.api_keys.split(",")]
    elif args.api_key:
        api_keys = [args.api_key]
    else:
        raise ValueError("Must provide --api-key or --api-keys")
    
    config = Config(
        api_keys=api_keys,
        model_name=args.model,
        save_root=args.save_dir
    )
    
    asyncio.run(run_benchmark(config))

if __name__ == "__main__":
    main()
