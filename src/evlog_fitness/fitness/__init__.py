# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:58:26 2026


Fitness computation submodule.

This package contains functions to compute alignment-based
fitness on event logs using Petri nets.


@author: I. Oscoz Villanueva
"""



from .fitness import calculate_fitness

__all__ = [
    "calculate_fitness",
]

