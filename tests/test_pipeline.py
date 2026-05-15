# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:40:38 2026

@author: I. Oscoz Villanuevas
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from datetime import timedelta

from evlog_fitness import (
    evlog_from_prescriptions,
    calculate_fitness,
)


# ---------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------
def synthetic_prescriptions(seed=0, n_patients=5):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_patients):
        pid = f"P{i}"
        n_events = rng.integers(1, 5)

        for _ in range(n_events):
            start = rng.integers(0, 180)
            duration = rng.integers(15, 90)

            date_ini = pd.Timestamp("2020-01-01") + timedelta(days=int(start))
            date_end = date_ini + timedelta(days=int(duration))

            rows.append({
                "patient_id": pid,
                "Event": rng.choice(["MET", "DPP4", "SGLT2"]),
                "prescription_date_ini": date_ini,
                "prescription_date_end": date_end,
            })

    return pd.DataFrame(rows)



def synthetic_measurements(seed=1, n_patients=5):
    rng = np.random.default_rng(seed)
    rows = []
    
    for i in range(n_patients):
        pid = f"P{i}"
        n_events = rng.integers(0, 3)

        for _ in range(n_events):
            day = rng.integers(1, 180)

            date = pd.Timestamp("2020-01-01") + timedelta(days=int(day))

            rows.append({
                "patient_id": pid,
                "Event": "HbA1c>8",
                "date": date,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def pnml_file(tmp_path: Path):
    """
    Minimal Petri net (simple linear model).
    """
    pnml = """<?xml version="1.0" encoding="UTF-8"?>
<pnml>
  <net id="net1" type="http://www.pnml.org/version-2009/grammar/pnmlcoremodel">
    <name>
      <text>PN</text>
    </name>
    <page id="n0">
      <place id="place1">
        <name>
          <text>ini</text>
        </name>
        <graphics>
          <position x="28.65224" y="506.836165" />
          <dimension x="30" y="30" />
        </graphics>
      </place>
      <transition id="transition1">
        <name>
          <text>MET</text>
        </name>
        <graphics>
          <position x="77" y="431" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <transition id="transition3">
        <name>
          <text>HbA1c&gt;8</text>
        </name>
        <graphics>
          <position x="316.05230754717" y="435.457684339623" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <transition id="transition6">
        <name>
          <text>DPP4+MET+SU</text>
        </name>
        <graphics>
          <position x="970" y="220" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <transition id="transition7">
        <name>
          <text>DPP4+MET+SGLT2</text>
        </name>
        <graphics>
          <position x="970" y="290" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <transition id="transition9">
        <name>
          <text>DPP4+MET+REPA</text>
        </name>
        <graphics>
          <position x="970" y="360" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <transition id="transition10">
        <name>
          <text>DPP4+MET+PIO</text>
        </name>
        <graphics>
          <position x="970" y="430" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <place id="place7">
        <graphics>
          <position x="370" y="440" />
          <dimension x="30" y="30" />
        </graphics>
      </place>
      <transition id="transition24">
        <name>
          <text>DPP4+MET</text>
        </name>
        <graphics>
          <position x="470" y="431" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <transition id="transition25">
        <name>
          <text>HbA1c&gt;8</text>
        </name>
        <graphics>
          <position x="746.331960377359" y="433.999109245283" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <place id="place9">
        <graphics>
          <position x="788" y="440" />
          <dimension x="30" y="30" />
        </graphics>
      </place>
      <transition id="transition22">
        <name>
          <text>FIN</text>
        </name>
        <graphics>
          <position x="130" y="80" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <transition id="transition26">
        <name>
          <text>FIN</text>
        </name>
        <graphics>
          <position x="550" y="150" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <transition id="transition28">
        <name>
          <text>FIN</text>
        </name>
        <graphics>
          <position x="645.845300577014" y="6.38009306196102" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <place id="place111">
        <name>
          <text>fin</text>
        </name>
        <graphics>
          <position x="670" y="90" />
          <dimension x="30" y="30" />
        </graphics>
      </place>
      <transition id="transition325">
        <name>
          <text>FIN</text>
        </name>
        <graphics>
          <position x="1150" y="150" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <transition id="transition39">
        <name>
          <text>INI</text>
        </name>
        <graphics>
          <position x="394.100051701089" y="497.331496075607" />
          <dimension x="35" y="35" />
        </graphics>
      </transition>
      <place id="place100">
        <name>
          <text>00</text>
        </name>
        <graphics>
          <position x="725.316070002161" y="506.727695176346" />
          <dimension x="30" y="30" />
        </graphics>
        <initialMarking>
          <text>1</text>
        </initialMarking>
      </place>
      <place id="place93">
        <graphics>
          <position x="1210" y="440" />
          <dimension x="30" y="30" />
        </graphics>
      </place>
      <place id="place94">
        <graphics>
          <position x="670" y="440" />
          <dimension x="30" y="30" />
        </graphics>
      </place>
      <place id="place95">
        <graphics>
          <position x="250" y="440" />
          <dimension x="30" y="30" />
        </graphics>
      </place>
      <arc id="arc1" source="place1" target="transition1">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc47" source="place7" target="transition24">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc56" source="transition3" target="place7">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc58" source="transition25" target="place9">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc60" source="place9" target="transition7">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc61" source="place9" target="transition9">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc67" source="place9" target="transition6">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc71" source="place9" target="transition10">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc54" source="transition28" target="place111">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc59" source="transition26" target="place111">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc63" source="transition22" target="place111">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc670" source="transition325" target="place111">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc68" source="place100" target="transition39">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc69" source="transition39" target="place1">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc578" source="place95" target="transition3">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc581" source="place94" target="transition25">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc109" source="transition1" target="place95">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc110" source="place95" target="transition22">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc111" source="transition24" target="place94">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc112" source="place94" target="transition26">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc113" source="transition10" target="place93">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc114" source="transition9" target="place93">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc115" source="transition7" target="place93">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc116" source="transition6" target="place93">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
      <arc id="arc117" source="place93" target="transition325">
        <name>
          <text>1</text>
        </name>
        <arctype>
          <text>normal</text>
        </arctype>
      </arc>
    </page>
  </net>
</pnml>
"""
    path = tmp_path / "model.pnml"
    path.write_text(pnml)
    return path


# ---------------------------------------------------------------------
# Core pipeline test
# ---------------------------------------------------------------------
def test_full_pipeline_prescriptions(
    pnml_file,
):
    """
    End-to-end test:
        prescriptions → event log → fitness
    """

    treat = synthetic_prescriptions()
    param = synthetic_measurements()
    
    # Build event log
    evlog = evlog_from_prescriptions(
        treat=treat,
        param=param,
        study_start_date="2020-01-01",
        study_end_date="2020-05-31",
        min_days_to_treatment_change=7,
    )

    # Basic structure checks
    assert isinstance(evlog, pd.DataFrame)
    assert set(evlog.columns) == {"ID", "date", "Event", "nid", "cycle"}
    
    if evlog.empty:
            pytest.fail("Event log unexpectedly empty")

    # Cycle validit                    
    assert set(evlog["cycle"]) <= {"start", "end"}

    # Each patient: sorted dates
    for pid, group in evlog.groupby("ID"):
        assert group["date"].is_monotonic_increasing, f"Dates not sorted for {pid}"

    #  Start/end consistency
    for pid, group in evlog.groupby("ID"):
        n_start = (group["cycle"] == "start").sum()
        n_end = (group["cycle"] == "end").sum()
        assert n_start == n_end, f"Mismatch start/end for {pid}"

    # Compute fitness
    fitness_df = calculate_fitness(
        df=evlog,
        baseline=pd.Timestamp("2020-01-01"),
        endline=pd.Timestamp("2020-05-31"),
        pnml_file=str(pnml_file),
        initial_place="place100",
        final_place="place111",
        initial_event="INI",
        final_event="FIN",
        fixed_period_days=14,
    )

    # Fitness structure
    assert isinstance(fitness_df, pd.DataFrame)
    assert "fitness" in fitness_df.columns

    # Fitness values valid
    if not fitness_df.empty:
            assert fitness_df["fitness"].between(0.0, 1.0).all()


    # All patients included
    assert set(fitness_df["ID"]).issubset(set(evlog["ID"]))
    


# ---------------------------------------------------------------------
# Stability test (multiple seeds)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("seed", [5, 6, 7, 8])
def test_pipeline_stability(seed):
    treat = synthetic_prescriptions(seed)

    evlog = evlog_from_prescriptions(
        treat=treat,
        study_start_date="2020-01-01",
        study_end_date="2020-05-31",
        min_days_to_treatment_change=2
    )

    # Should not crash
    assert evlog is not None
