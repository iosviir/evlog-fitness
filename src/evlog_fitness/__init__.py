# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:32:42 2026


evlog-fitness

Treatment event-log construction and fitness (compliance with the guidelines) 
computation.


@author: I. Oscoz Villanueva
"""


__version__ = "0.1.0"

from .evlog.prescriptions import evlog_from_prescriptions
from .evlog.dispensations import evlog_from_dispensations
from .fitness.fitness import calculate_fitness

__all__ = [
    "evlog_from_prescriptions",
    "evlog_from_dispensations",
    "calculate_fitness",
]