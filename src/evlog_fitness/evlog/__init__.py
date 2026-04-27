# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:58:26 2026


Event-log construction submodule.

This package provides functions to build process-mining–ready
event logs from longitudinal treatment dat


@author: I. Oscoz Villanueva
"""


from .prescriptions import evlog_from_prescriptions
from .dispensations import evlog_from_dispensations

__all__ = [
    "evlog_from_prescriptions",
    "evlog_from_dispensations",
]
