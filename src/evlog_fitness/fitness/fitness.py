# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:47:09 2026


Alignment-based conformance fitness computation.

This module provides functions to compute alignment fitness
between an event log and a Petri net model using PM4Py.

Two computation modes are supported:
    1. Interval-based fitness over fixed time windows
    2. Fitness at the end of follow-up (full trace)

The public API is the function `calculate_fitness`


@author: I. Oscoz Villanuevas
"""


from datetime import timedelta
from functools import lru_cache
from typing import Dict, Optional 

import pandas as pd
import pm4py
from tqdm import tqdm

from .preparation import evlog_preparation
from .petri_net import load_PetriNet


# ---------------------------------------------------------------------
# Interval-based fitness computation
# ---------------------------------------------------------------------
def _calculate_fitness_by_intervals(
    event_log: pd.DataFrame,
    net,
    initial_marking,
    final_marking,
    baseline: pd.Timestamp,
    endline: pd.Timestamp,
    followup_ends: Dict,
    outcomes: Dict,
    interval_days: int,
) -> pd.DataFrame:
    """
    Computes conformance fitness over successive time intervals for each case.

    Returns
    -------
    pd.DataFrame
        One row per case per interval with fitness and outcome status.
    """
    
    @lru_cache(maxsize=2048)
    def cached_alignment_fitness(trace_signature):
        """
        Computes alignment fitness for a trace signature and caches the result.
    
        Parameters
        ----------
        trace_signature : tuple
            Tuple of (activity, timestamp_int) pairs.
    
        Returns
        -------
        float
            Alignment fitness.
        """
        trace_df = pd.DataFrame(
            trace_signature,
            columns=["concept:name", "order"]
        )
        # Dummy case name
        trace_df["case:concept:name"] = "CACHED_CASE"
        
        # Dummy timestamps (order-preserving, deterministic)
        trace_df["time:timestamp"] = pd.to_datetime(
            trace_df["order"], unit="D", origin="unix"
        )

        alignment = pm4py.conformance_diagnostics_alignments(
            trace_df,
            net,
            initial_marking,
            final_marking
        )
    
        return alignment[0]["fitness"]
    
    pre_INI_date = baseline-timedelta(days=1)
    post_FIN_date = endline+timedelta(days=1)

    period2fitness = []
    
    grouped = event_log[[
        'case:concept:name',
        'concept:name',
        'time:timestamp',
        'days_since_baseline',
        'days_since_min_date'
        ]].groupby('case:concept:name', sort=False)

    
    for case_id, event_log_id in tqdm(grouped):
        case_start_days_since_baseline = event_log_id['days_since_baseline'].min()
        case_start_date = event_log_id['time:timestamp'].min() 
        outcome_day = outcomes.get(
            case_id,
            1000000
            ) - case_start_days_since_baseline
        followup_days = followup_ends.get(
            case_id,
            (endline-baseline).days
            ) - case_start_days_since_baseline
        case_followup_days = min(followup_days, outcome_day) 
        n_periods = case_followup_days//interval_days
        n_periods_r = case_followup_days/interval_days
        if case_followup_days<=interval_days:
            continue
               
        # Artificial INI / FIN events to ensure sound model replay
        artificial_inifin = event_log_id.iloc[[0, -1]].copy()
        artificial_inifin["concept:name"] = ["INI", "FIN"]
        artificial_inifin["time:timestamp"] = [pre_INI_date, post_FIN_date]

        for n in range(1, n_periods + 1):
    
            cutoff = n * interval_days
            trace_until_cutoff = event_log_id[
                event_log_id['days_since_min_date'] < cutoff
            ]
            trace_until_cutoff = (
                pd.concat([trace_until_cutoff, artificial_inifin],
                          ignore_index=True
                ).sort_values('time:timestamp',
                          ignore_index=True)
            )
            
            aligned_fitness = cached_alignment_fitness(
                tuple(
                    zip(
                        trace_until_cutoff["concept:name"],
                        range(len(trace_until_cutoff["concept:name"]))
                    )
                )
            )
            
            interval_end = (
                cutoff if n < n_periods
                else case_followup_days - interval_days
            )
            
            period2fitness.append({
                "ID": case_id,
                "t_0": (n - 1) * interval_days,
                "t_1": interval_end,
                "fitness": aligned_fitness,
                "ini_date": case_start_date,
                "trace_length": len(trace_until_cutoff),
                "status": int(
                    outcome_day == case_followup_days and
                    (n == n_periods or n == n_periods_r - 1)
                    )
                })
    
            if period2fitness[-1]["status"] == 1:
                break
    cached_alignment_fitness.cache_clear()
    return pd.DataFrame(period2fitness)


# ---------------------------------------------------------------------
# End-of-follow-up fitness computation
# ---------------------------------------------------------------------
def _calculate_fitness_at_end(
    event_log: pd.DataFrame,
    net,
    initial_marking,
    final_marking,
    baseline: pd.Timestamp,
    endline: pd.Timestamp,
    followup_ends: Dict,
    outcomes: Dict,
) -> pd.DataFrame:
    """
    Computes conformance fitness at the end of follow up for each case.

    Returns
    -------
    pd.DataFrame
        One row per case with fitness and followup days.
    """
    
    @lru_cache(maxsize=2048)
    def cached_alignment_fitness(trace_signature):
        """
        Computes alignment fitness for a trace signature and caches the result.
    
        Parameters
        ----------
        trace_signature : tuple
            Tuple of (activity, timestamp_int) pairs.
    
        Returns
        -------
        float
            Alignment fitness.
        """
        trace_df = pd.DataFrame(
            trace_signature,
            columns=["concept:name", "order"]
        )
        # Dummy case name
        trace_df["case:concept:name"] = "CACHED_CASE"
        
        # Dummy timestamps (order-preserving, deterministic)
        trace_df["time:timestamp"] = pd.to_datetime(
            trace_df["order"], unit="D", origin="unix"
        )

        alignment = pm4py.conformance_diagnostics_alignments(
            trace_df,
            net,
            initial_marking,
            final_marking
        )
    
        return alignment[0]["fitness"]
    
    pre_INI_date = baseline-timedelta(days=1)
    post_FIN_date = endline+timedelta(days=1)

    id2fitness = []
    
    grouped = event_log[[
        'case:concept:name',
        'concept:name',
        'time:timestamp',
        'days_since_baseline',
        'days_since_min_date'
        ]].groupby('case:concept:name', sort=False)

    
    for case_id, event_log_id in tqdm(grouped):
        case_start_days_since_baseline = event_log_id['days_since_baseline'].min()
        case_start_date = event_log_id['time:timestamp'].min() 
        outcome_day = outcomes.get(
            case_id,
            1000000
            ) - case_start_days_since_baseline
        followup_days = followup_ends.get(
            case_id,
            (endline-baseline).days
            ) - case_start_days_since_baseline
        case_followup_days = min(followup_days, outcome_day) 

        # Artificial INI / FIN events to ensure sound model replay
        artificial_inifin = event_log_id.iloc[[0, -1]].copy()
        artificial_inifin["concept:name"] = ["INI", "FIN"]
        artificial_inifin["time:timestamp"] = [pre_INI_date, post_FIN_date]

        event_log_id = (
            pd.concat([event_log_id, artificial_inifin],
                      ignore_index=True
            ).sort_values('time:timestamp',
                      ignore_index=True)
        )
            
        aligned_fitness = cached_alignment_fitness(
            tuple(
                zip(
                    event_log_id["concept:name"],
                    range(len(event_log_id["concept:name"]))
                )
            )
        )
            
        id2fitness.append({
            "ID": case_id,
            "initial_date": case_start_date,
            "followup_days": case_followup_days,
            "fitness": aligned_fitness,
            "ini_date": case_start_date,
            "trace_length": len(event_log_id)
            })
    
    cached_alignment_fitness.cache_clear()
    return pd.DataFrame(id2fitness)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def calculate_fitness(
    df: pd.DataFrame,
    baseline: pd.Timestamp,
    endline: pd.Timestamp,
    pnml_file: str,
    initial_place: str,
    final_place: str,
    followup_ends: Optional[Dict] = None,
    outcomes: Optional[Dict] = None,
    case_id_key: str = "ID",
    activity_key: str = "Event",
    timestamp_key: str = "date",
    fixed_period_days: Optional[int] = 90,
) -> pd.DataFrame:
    """
    Compute alignment-based conformance fitness for an event log.

    Parameters
    ----------
    df : pd.DataFrame
        Event log in tabular form.

    baseline : pd.Timestamp
        Global study baseline.

    endline : pd.Timestamp
        Global study end date.

    pnml_file : str
        Path to Petri net model in PNML format.

    initial_place : str
        Name of the initial place in the Petri net.

    final_place : str
        Name of the final place in the Petri net.

    followup_ends : dict, optional
        Per-case follow-up end overrides.

    outcomes : dict, optional
        Per-case outcome days (relative to baseline).

    fixed_period_days : int or None
        Interval size in days.
        If None, fitness is computed at end of follow-up.

    Returns
    -------
    pd.DataFrame
        DataFrame with fitness results.
    """
    if followup_ends==None:
        followup_ends = dict()
    if outcomes==None:
        outcomes = dict()
    # Prepare PM4Py-compatible event log
    event_log = evlog_preparation(
            df,
            baseline,
            case_id_key,
            activity_key,
            timestamp_key
            )    
    # Load Petri net and markings
    net, ini_marking, fin_marking = load_PetriNet(
        pnml_file,
        initial_place,
        final_place
        )
    if fixed_period_days==None:
        return _calculate_fitness_at_end(
                event_log,
                net, 
                ini_marking,
                fin_marking,
                baseline,
                endline,
                followup_ends,
                outcomes
                )
    else:
        return _calculate_fitness_by_intervals(
                event_log,
                net, 
                ini_marking,
                fin_marking,
                baseline,
                endline,
                followup_ends,
                outcomes,
                fixed_period_days
                )
