# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:50:40 2026


Event-log construction from dispensation data.

This module provides the high-level function
`evlog_from_dispensations`, which converts longitudinal
drug dispensation records into a process-mining–ready
event log.

Compared to prescriptions, dispensation data may include
uncertainty due to delayed intake or stock-outs. Therefore,
this pipeline explicitly models uncertainty windows and
measurement-related blocking before extracting treatment
intervals.

Pipeline overview
-----------------
1. Build a daily treatment status matrix with uncertainty
2. Apply uncertainty compaction and measurement corrections
3. Extract start/end treatment intervals


@author: I. Oscoz Villanueva
"""
import pandas as pd
from datetime import  datetime, timedelta
from tqdm import  tqdm

from .status_matrix import (
    build_event_status_matrix,
    apply_uncertainity_and_measurement_corrections,
)
from .extraction import extract_event_log_from_status_matrix


def evlog_from_dispensations(
    
treat: pd.DataFrame,
    param: pd.DataFrame = None,
    study_start_date: str = "2017-01-01",
    study_end_date: str = "2023-01-01",
    start_date: dict = None,
    fin_date: dict = None,
    uncertain_days_after_measure: int = 5,
    min_days_to_treatment_change: int = 7,
    between_dispensation_uncertain_days_percent: float = 0.3,
    uncertain_max_days: int = 30,
) -> pd.DataFrame:
    
    """
    Build an event log from dispensation data.

    Parameters
    ----------
    treat : pd.DataFrame
        Dispensation data with at least the following columns:
            - patient_id
            - Event
            - dispensing_date
            - stock_out_date

    param : pd.DataFrame, optional
        Measurement events (e.g. lab tests). If provided, must contain:
            - patient_id
            - Event
            - date

    study_start_date : str
        Default study start date (ISO format).

    study_end_date : str
        Default study end date (ISO format).

    start_date : dict, optional
        Per-patient override for follow-up start dates.

    fin_date : dict, optional
        Per-patient override for follow-up end dates.

    uncertain_days_after_measure : int
        Number of days after a measurement during which treatments
        are considered unreliable.

    min_days_to_treatment_change : int
        Minimum duration (in days) required to emit a treatment episode.

    between_dispensation_uncertain_days_percent : float
        Fraction of the active dispensation duration that is treated
        as an uncertainty window after stock-out.

    uncertain_max_days : int
        Maximum allowed length (in days) for uncertainty compaction.

    Returns
    -------
    pd.DataFrame
        Event log with columns:
            - ID
            - date
            - Event
            - nid
            - cycle ('start' / 'end')
    """

    # -------------------------------
    # Input normalization
    # -------------------------------
    param = param if param is not None else pd.DataFrame()
    start_date = start_date if start_date is not None else {}
    fin_date = fin_date if fin_date is not None else {}

    measure_types: list[str] = []


    if not param.empty:
        param = param.rename(columns={"date": "date0"})
        param["date1"] = param["date0"]
        measure_types = param["Event"].unique().tolist()

    df = pd.concat([
        treat.rename(columns={
            "dispensing_date": "date0",
            "stock_out_date": "date1",
        }),
        param,
    ])


    events = sorted(df["Event"].unique())
    # Initialize output event log structure
    event_log = {k: [] for k in
                 ["patient_id", "date", "Event", "nid", "cycle"]}
    
    # -------------------------------
    # Per-patient processing
    # ------------------------------
    for nid, pid in enumerate(tqdm(df["patient_id"].unique())):
        df_id = df[df["patient_id"] == pid]
        # Determine patient-specific observation window
        date_min = min(
            df_id["date0"].min(),
            start_date.get(pid,
                datetime.strptime(study_start_date, "%Y-%m-%d"))
        )
        date_max = fin_date.get(
            pid,
            datetime.strptime(study_end_date, "%Y-%m-%d")
        )

        date_list = [
            date_min + timedelta(days=i)
            for i in range((date_max - date_min).days + 1)
        ]


        # -------------------------------
        # Build status matrix with uncertainty
        # ---------------------------
        status_matrix = build_event_status_matrix(
            df_id, date_list, events, between_dispensation_uncertain_days_percent
        )

        measure_rows = [events.index(e) for e in measure_types]
        non_measure_rows = list(set(range(len(events))) - set(measure_rows))


        # -------------------------------
        # Apply uncertainty & measurement rules
        # -------------------------------                
        status_matrix = apply_uncertainity_and_measurement_corrections(
            status_matrix,
            measure_rows,
            non_measure_rows,
            uncertain_days_after_measure,
            uncertain_max_days
        )


        # -------------------------------
        # Extract event log intervals
        # -------------------------------
        event_log = extract_event_log_from_status_matrix(
            pid,
            status_matrix,
            date_list,
            start_date.get(pid,
                datetime.strptime(study_start_date, "%Y-%m-%d")),
            fin_date.get(pid,
                datetime.strptime(study_end_date, "%Y-%m-%d")),
            events,
            measure_types,
            min_days_to_treatment_change,
            nid,
            event_log,
        )

    # -------------------------------
    # Final formatting
    # -------------------------------
    return (
        pd.DataFrame(event_log)
        .rename(columns={"patient_id": "ID"})
        .sort_values(["ID", "date"])
        .reset_index(drop=True)
    )
