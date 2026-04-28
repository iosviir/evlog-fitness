# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:00:33 2026


Daily treatment status-matrix construction and correction rules.

This module contains functions that operate on a daily
(events × days) status matrix representing treatment activity.

The status matrix is an intermediate representation used to:
    1. Model uncertainty in treatment continuation
    2. Correct artifacts around measurement events
    3. Prepare clean input for event-log extraction

Status codes
------------
The status matrix uses the following integer codes:

    0 : INACTIVE   → no treatment
    1 : ACTIVE     → confirmed treatment
    2 : UNCERTAIN  → possible continuation (doubt window)
    3 : BLOCKED    → blocked due to measurement influence

All public functions in this module preserve these semantics.


@author: I. Oscoz Villanueva
"""


import itertools
import re
import numpy as np
import pandas as pd
from typing import List


# ---------------------------------------------------------------------
# Status-matrix construction
# ---------------------------------------------------------------------
def build_event_status_matrix(
    df: pd.DataFrame,
    date_list: List[pd.Timestamp],
    events: List[str],
    between_dispensation_uncertain_days_percent: float = 0.0,
) -> np.ndarray:
    """
    Construct a daily treatment status matrix from raw event intervals.

    Each row represents a treatment (event type), and each column
    represents a day in the observation window.

    States used
    -----------
    0 : inactive treatment
    1 : active treatment
    2 : uncertain continuation after interval end
    3: blocked by measurement

    Uncertainty model
    -----------------
    After the nominal end of a treatment interval, an uncertainty window
    may be added. Its length is proportional to the treatment duration
    (`between_dispensation_uncertain_days_percent`).

    This is mainly used for dispensation-based data, where stock-out
    dates may not perfectly match actual treatment usage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event data for a single patient, with columns:
            - Event
            - date0 (start date)
            - date1 (end date)

    date_list : list[pd.Timestamp]
        Ordered list of days defining the observation window.

    events : list[str]
        List of all possible treatment names (rows of the matrix).

    between_dispensation_uncertain_days_percent : float, optional
        Fraction of the active duration that is treated as an
        uncertainty window after the nominal end of treatment.
        Typically > 0 only for dispensation data.

    Returns
    -------
    np.ndarray
        Status matrix of shape (n_events, n_days) with values in
        {0, 1, 2}.
    """

    n_events = len(events)
    n_days = len(date_list)
    status_matrix = np.zeros((n_events, n_days), dtype=int)

    event_to_row = {e: i for i, e in enumerate(events)}
    date_to_col = {d: i for i, d in enumerate(date_list)}

    for _, df_row in df.iterrows():
        row_idx = event_to_row[df_row["Event"]]
        start_day = date_to_col.get(df_row["date0"])
        end_day = date_to_col.get(min(df_row["date1"], date_list[-1]))

        if start_day is None or start_day > end_day:
            continue
        
        status_matrix[row_idx, start_day : end_day + 1] = 1
        
        stop = min(end_day + int((end_day-start_day) * between_dispensation_uncertain_days_percent),
                   n_days)
        status_matrix[row_idx, end_day:stop] = 2

    return status_matrix


# ---------------------------------------------------------------------
# Measurement-only corrections (used for prescriptions)
# ---------------------------------------------------------------------

def apply_uncertainity_and_measurement_corrections(
    status_matrix: np.ndarray,
    measure_rows: List[int],
    non_measure_rows: List[int],
    uncertain_days_after_measure: int,
    uncertain_max_days: int,
) -> np.ndarray:   
    """
    Remove spurious treatment continuations immediately
    following measurement events.

    This correction is used for prescription data.


    Rationale
    ---------
    Measurements (e.g., lab tests) may cause artificial treatment
    interruptions or changes in the recorded data. This function removes
    spurious short-lived treatment continuations following measurements.

    Strategy
    --------
    - Treatments are suppressed on measurement days.
    - Short continuations immediately after measurements are removed
      if they do not represent a genuine treatment change.
    """

    if uncertain_days_after_measure < 3:
        return status_matrix
    
    measure_days = np.where(status_matrix[measure_rows].any(axis=0))[0]

    # Remove non-measurement treatments on measurement days
    status_matrix[np.ix_(non_measure_rows, measure_days)] = 0

    for day in measure_days:
        if day + 2 >= status_matrix.shape[1]:
            continue

        day_after = status_matrix[:, day + 1]
        future_window = status_matrix[:, day + 2 : day + uncertain_days_after_measure]

        # Find first day where treatment differs
        change_cols = np.where(
            (future_window != day_after[:, None]).any(axis=0)
        )[0]

        if change_cols.size == 0:
            continue

        first_change_day = day + change_cols[0] + 2

        if (
            np.array_equal(status_matrix[:, day - 1], day_after)
            and not np.array_equal(
                status_matrix[:, day - 1], status_matrix[:, first_change_day]
            )
        ):
            # Remove misleading continuation
            status_matrix[
                np.ix_(
                    non_measure_rows,
                    range(day + 1, first_change_day)
                    )
                ] = 0

    return status_matrix



# ---------------------------------------------------------------------
# Uncertainty + measurement corrections (used for dispensations)
# ---------------------------------------------------------------------
def apply_uncertainity_and_measurement_corrections(
    status_matrix: np.ndarray,
    measure_rows: List[int],
    non_measure_rows: List[int],
    uncertain_days_after_measure: int,
    uncertain_max_days: int,
) -> np.ndarray:
    """
    Apply uncertainty compaction and measurement-related corrections.

    
    RULE A — Uncertainty compaction
        Short UNCERTAIN gaps between ACTIVE segments are
        promoted to continuous ACTIVE treatment.

    RULE B — Measurement-day blocking
        Treatments on measurement days are marked as BLOCKED.

    RULE C — Cross-treatment reinforcement
        When treatments overlap until a measurement boundary,
        continuity of one may reinforce uncertainty of another.

    RULE D — Post-measurement blocking window
        Treatments are blocked for a fixed window after measurement.

    Final step
    ----------
    The matrix is binarized (ACTIVE vs INACTIVE).
    """

    measure_days = np.where(np.any(status_matrix[measure_rows,:] != 0,axis=0))[0]
    
    # ---------------------------------------------------------------------
    # RULE A — UNCERTAINTY COMPACTION (WITHIN SAME TREATMENT)
    #
    # If a treatment appears as:
    #   1 → 2 → 2 → ... → 2 → 1
    # within a time window shorter than uncertain_max_days,
    # then the uncertain period (state 2) is promoted to active (state 1).
    #
    # Interpretation:
    #   Short gaps in dispensing are assumed to be continuous treatment.
    #
    # Implementation note:
    #   This uses string-based replacement intentionally for speed on long
    #   1D time series (days axis).
    # ---------------------------------------------------------------------
    for i in non_measure_rows:
        seq = str(list(status_matrix[i,:]))
        for delta_days in range(1,uncertain_max_days):
            # Pattern: active → uncertain × delta_days → active
            pattern = ', '+str([1]+[2]*delta_days+[1])[1:-1]
            # Replacement: fully active across the gap
            new_pattern = ', '+str([1]+[1]*delta_days+[1])[1:-1]
            seq = seq.replace(pattern,new_pattern)
        # Convert the updated sequence back into numeric form
        status_matrix[i,:] = np.fromstring(seq[1:-1], sep=',', dtype=int) 
    
    # ---------------------------------------------------------------------
    # RULE B — MEASUREMENT DAY BLOCKING
    #
    # On days where a measurement occurs, treatments are temporarily blocked
    # (state 3).
    #
    # Rationale:
    #   Treatment signals on the same day as a measurement are ambiguous
    #   and should not be interpreted as active changes.
    # ---------------------------------------------------------------------
    status_matrix[np.ix_(non_measure_rows, measure_days)] = 3
    
    # ---------------------------------------------------------------------
    # RULE C — CROSS-TREATMENT PROPAGATION BEFORE MEASUREMENT
    #
    # If two treatments overlap up to a measurement block:
    #
    #   - One treatment is clearly continuous (state 1)
    #   - The other shows uncertainty (state 2)
    #
    # and both end at the same measurement block (state 3),
    # then the uncertain treatment is promoted to continuous.
    #
    # Interpretation:
    #   Concurrent treatments reinforce each other's continuity when
    #   interrupted by measurement day.
    #
    # Note:
    #   This is applied symmetrically across all pairs of treatments.
    # ---------------------------------------------------------------------
    for i,j in list(itertools.combinations(non_measure_rows,2)):
        seq_i = str(list(status_matrix[i,:]))
        seq_j = str(list(status_matrix[j,:]))
        # Case 1: treatment i is solid, treatment j is uncertain
        for delta_days in range(1,uncertain_max_days):
            pattern_i = ', '+str([1]+[1]*delta_days+[3])[1:-1]
            if re.search(pattern_i,seq_i)==None:
                continue
            pattern_j = ', '+str([1]+[2]*delta_days+[3])[1:-1]
            new_pattern_j = ', '+str([1]+[1]*delta_days+[3])[1:-1]
            seq_j = seq_j.replace(pattern_j,new_pattern_j)
        # Case 2: symmetric — treatment j solid, treatment i uncertai
        for delta_days in range(1,uncertain_max_days):
            pattern_j = ', '+str([1]+[1]*delta_days+[3])[1:-1]
            if re.search(pattern_j,seq_j)==None:
                continue
            pattern_i = ', '+str([1]+[2]*delta_days+[3])[1:-1]
            new_pattern_i = ', '+str([1]+[1]*delta_days+[3])[1:-1]
            seq_i = seq_i.replace(pattern_i,new_pattern_i)
        status_matrix[i,:] = np.fromstring(seq_i[1:-1], sep=',', dtype=int)
        status_matrix[j,:] = np.fromstring(seq_j[1:-1], sep=',', dtype=int)
    
    # ---------------------------------------------------------------------
    # RULE D — POST-MEASUREMENT BLOCKING WINDOW
    #
    # After each measurement day, non-measurement treatments are blocked
    # for a fixed look-ahead window (uncertain_days_after_measure).
    #
    # Rationale:
    #   Immediate post-measurement treatment signals may reflect delayed
    # ---------------------------------------------------------------------
    blocked_days = [[
        day + offset
        for offset in range(uncertain_days_after_measure + 1)
        if day + offset < status_matrix.shape[1]
    ] for day in measure_days]

    status_matrix[np.ix_(
        non_measure_rows,
        list(itertools.chain(*blocked_days))
        )] = 3
    # ---------------------------------------------------------------------
    # FINAL CLEANUP — BINARIZATION
    #
    # For non-measurement treatments:
    #   - Keep only confirmed activity (state 1)
    #   - Remove uncertainty (state 2) and block markers (state 3)
    #
    # Outcome:
    #   status_matrix is now a pure binary matrix (0/1) suitable
    #   for event-log extraction.
    # ---------------------------------------------------------------------
    status_matrix[non_measure_rows] = (
        status_matrix[non_measure_rows] == 1
        ).astype(int)               
    return status_matrix
