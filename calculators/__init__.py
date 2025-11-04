# cardio-tool/app/calculators/__init__.py
"""
Cardio-Tool calculators package.

Contains evidence-based, rule-driven calculators for cardiovascular risk assessment,
blood pressure classification, and ECG interpretation.

All functions are for **Research Use Only (RUO)** and are not intended for
clinical diagnosis or treatment decisions.

This module implements models from:
- Goff et al. 2013 ACC/AHA Pooled Cohort Equations (ASCVD)
- Lip et al. CHA₂DS₂-VASc score (Chest 2010)
- Whelton et al. 2017 ACC/AHA BP Guidelines (Hypertension 2018)
- ACC/AHA/HRS ECG interpretation standards

Use cases: research, education, clinical trial support, and digital health prototyping.

Modules:
    ascvd: 10-year ASCVD risk (Pooled Cohort Equations)
    bp_category: ACC/AHA 2017 blood pressure classification
    cha2ds2vasc: Stroke risk in atrial fibrillation
    ecg_interpret: Rhythm + 12-lead ECG findings (RUO)
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Cerevia Inc."
__license__ = "Proprietary"
__email__ = "research@cerevia.ai"

# Import public functions
from .ascvd import ascvd_pce_risk
from .bp_category import bp_category
from .cha2ds2vasc import cha2ds2vasc
from .ecg_interpret import (
    interpret_rhythm,
    ecg_interpret_detailed,
    interpret_12lead_findings,
    interpret_ecg_comprehensive
)

# Define public API
__all__ = [
    # ASCVD Risk
    "ascvd_pce_risk",

    # BP Classification
    "bp_category",

    # Stroke Risk
    "cha2ds2vasc",

    # ECG Interpretation
    "interpret_rhythm",
    "ecg_interpret_detailed",
    "interpret_12lead_findings",
    "interpret_ecg_comprehensive"
]