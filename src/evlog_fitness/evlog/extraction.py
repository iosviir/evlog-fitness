# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:26:04 2026


Event-log extraction from daily treatment status matrices.

This module converts a daily (events × days) status matrix into
a process-mining–ready event log by collapsing consecutive days
with identical treatment configurations into start/end intervals.

The extraction algorithm behaves like a finite-state scan over time:
    - unchanged treatment configuration → extend current segment
    - change detected → close previous segment and start a new one


@author: I. Oscoz Villanueva
"""


import numpy as np
from typing import List, Dict


def column_to_treatment(
    status_matrix: np.ndarray,
    event_names: List[str],
    day_col: int,
) -> str:
    """
    Encode the active treatments on a given day as a string.

    Parameters
    ----------
    status_matrix : np.ndarray
        Binary or categorical status matrix (events × days).

    event_names : list[str]
        List of treatment names corresponding to matrix rows.

    day_col : int
        Column index representing a specific day.

    Returns
    -------
    str
        '+'-separated treatment name(s) active on that day,
        or '_' if no treatment is active.
    """

    active_rows = np.flatnonzero(status_matrix[:, day_col])
    if active_rows.size == 0:
        return "_"
    return "+".join(sorted(event_names[i] for i in active_rows))



def extract_event_log_from_status_matrix(
    patient_id,
    status_matrix: np.ndarray,
    date_list: List,
    start_date: Dict,
    fin_date: Dict,
    events: List[str],
    measure_types: List[str],
    min_days_to_treatment_change: int,
    nid: int,
    event_log: Dict[str, list],
) -> Dict[str, list]:  
    """
    Convert a daily treatment status matrix into start/end events.

    The algorithm scans the matrix column by column and detects changes
    in the treatment configuration. Each maximal period with constant
    treatment state is emitted as a start/end pair.

    Short segments shorter than `min_days_to_treatment_change` are
    ignored unless they correspond to measurement events.

    Parameters
    ----------
    patient_id : any
        Patient identifier.

    status_matrix : np.ndarray
        Binary status matrix (events × days).

    date_list : list
        Ordered list of dates corresponding to matrix columns.

    start_date : dict
        Optional per-patient override for start of follow-up.

    fin_date : dict
        Optional per-patient override for end of follow-up.

    events : list[str]
        List of treatment names corresponding to matrix rows.

    measure_types : list[str]
        List of event types considered measurements.

    min_days_to_treatment_change : int
        Minimum duration (in days) required to emit a treatment episode,
        unless the episode corresponds to a measurement.

    nid : int
        Numeric patient index (used internally).

    event_log : dict
        Dictionary accumulating event-log columns.

    Returns
    -------
    dict
        Updated event_log dictionary.
    """

    # Determine patient-specific observation wind
    start_col = date_list.index(start_date[patient_id])
    end_col = (
        date_list.index(fin_date[patient_id])
        if str(fin_date[patient_id]) != "nan"
        else len(date_list)
    )

    segment_start = start_col
    day = start_col    
    # --------------------------------------------------
    # Finite-state scan over time
    # --------------------------------------------------
    while day < end_col - 1:
        if np.array_equal(status_matrix[:, day],
                          status_matrix[:, day + 1]):
            day += 1 
            continue
        # A change in treatment configuration is detected
        treatment = column_to_treatment(
            status_matrix, events, segment_start
        )
        # Ignore short non-measurement episode
        if (
            treatment not in measure_types
            and day - segment_start < min_days_to_treatment_change
        ):
            day += 1
            segment_start = day
            continue

        # Emit start/end events
        for date, cycle in (
            (date_list[segment_start], "start"),
            (date_list[day], "end")
        ):
            event_log["patient_id"].append(patient_id)
            event_log["date"].append(date)
            event_log["Event"].append(treatment)
            event_log["nid"].append(nid)
            event_log["cycle"].append(cycle)

        segment_start = day + 1
        day += 1

   # --------------------------------------------------
    # Final segment
    # --------------------------------------------------
    treatment = column_to_treatment(
        status_matrix, events, segment_start
    )
    if (
        treatment in measure_types
        or day - segment_start >= min_days_to_treatment_change
    ):
        for date, cycle in (
            (date_list[segment_start], "start"),
            (date_list[day], "end")
        ):
            event_log["patient_id"].append(patient_id)
            event_log["date"].append(date)
            event_log["Event"].append(treatment)
            event_log["nid"].append(nid)
            event_log["cycle"].append(cycle)

    return event_log
