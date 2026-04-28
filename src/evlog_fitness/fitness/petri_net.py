# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:02:08 2026


Petri net loading utilities for conformance checking.

This module contains helper functions to load Petri net models
from PNML files and to construct the initial and final markings
required for alignment-based conformance fitness computation.

All interactions with PM4Py Petri net objects are isolated here
to keep the rest of the package independent of PM4Py internals.


@author: I. Oscoz Villanuevas
"""


import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking

def load_PetriNet(
        pnml_file: str,
        initial_place: str,
        final_place: str
) -> tuple[PetriNet, Marking, Marking]:
    """
    Load a Petri net from a PNML file and construct its markings.

    Parameters
    ----------
    pnml_file : str
        Path to the Petri net model in PNML format.

    initial_place : str
        Name of the place representing the initial marking.

    final_place : str
        Name of the place representing the final marking.

    Returns
    -------
    tuple (PetriNet, Marking, Marking)
        - net : PetriNet
            The loaded Petri net.
        - initial_marking : Marking
            Marking with one token in the initial place.
        - final_marking : Marking
            Marking with one token in the final place.

    Raises
    ------
    ValueError
        If the specified initial or final place does not exist
        in the Petri net.

    """
    
    # --------------------------------------------------
    # Load the Petri net from PNML
    # --------------------------------------------------
    net, _, _ = pm4py.read_pnml(pnml_file)
    # Build a lookup table for places by na
    place_by_name = {p.name: p for p in net.places}

    # --------------------------------------------------
    # Construct markings
    # --------------------------------------------------
    try:
        initial_marking = Marking({place_by_name[initial_place]: 1})
        final_marking = Marking({place_by_name[final_place]: 1})
    except KeyError as e:
        raise ValueError(f"Place not found in Petri Net: {e.args[0]}")
    
    return net, initial_marking, final_marking
