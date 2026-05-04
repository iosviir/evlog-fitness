# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:48:50 2026


Event-log preparation utilities for conformance checking.

This module provides functions to transform a tabular event log
into a PM4Py-compatible format and to compute temporal features
used by downstream fitness calculations.

Responsibilities
----------------
- Format an event log using PM4Py conventions
- Normalize timestamps (timezone handling)
- Compute baseline-relative and case-relative time features


@author: I. Oscoz Villanuevas
"""

import pm4py
import pandas as pd

def evlog_preparation(
        df: pd.DataFrame,
        baseline: pd.Timestamp,
        case_id_key: str = "ID",
        activity_key: str = "Event",
        timestamp_key: str = "date"
        )-> pd.DataFrame:    
    """
    Prepare a PM4Py-compatible event log and compute
    case-relative temporal features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tabular event log with at least:
            - case identifier column
            - activity column
            - timestamp column

    baseline : pd.Timestamp
        Global study baseline date.

    case_id_key : str, default "ID"
        Column name identifying cases (patients).

    activity_key : str, default "Event"
        Column name identifying activities (treatments).

    timestamp_key : str, default "date"
        Column name containing timestamps.

    Returns
    -------
    pd.DataFrame
        PM4Py-formatted event log with additional columns:
            - 'days_since_baseline'
            - 'days_since_min_date'

    """
    
    # --------------------------------------------------
    # Format dataframe according to PM4Py conventions
    # --------------------------------------------------
    event_log = pm4py.format_dataframe(df,
                                case_id=case_id_key,
                                activity_key=activity_key,
                                timestamp_key=timestamp_key) 
    
    
    # --------------------------------------------------
    # Ensure timezone-naive timestamps
    # --------------------------------------------------
    ts = event_log["time:timestamp"]
    if getattr(ts.dt, "tz", None) is not None:
        event_log["time:timestamp"] = ts.dt.tz_convert(None)

    # --------------------------------------------------
    # Compute absolute days since study baseline
    # --------------------------------------------------
    event_log['days_since_baseline'] = (
        event_log['time:timestamp'] - baseline
        ).dt.days
    

    # --------------------------------------------------
    # Compute case-relative days since first event
    # --------------------------------------------------    
    event_log['days_since_min_date'] = event_log['days_since_baseline']    
    event_log['days_since_min_date'] = (
        event_log.groupby('case:concept:name')['days_since_min_date']
                 .transform(lambda x: x - x.min()))
    return event_log
