"""
Cardio-Tool Input Validators

This package provides robust input validation for cardiovascular risk calculators,
ensuring all patient data meets clinical, mathematical, and guideline-based requirements
before being passed to risk models (e.g., ASCVD, CHA₂DS₂-VASc, BP, ECG).

All functions raise descriptive `ValueError` exceptions on failure, enabling
clear feedback in web apps, APIs, and research workflows.

Designed for Research Use Only (RUO) — not for clinical diagnosis.

Modules:
    - patient_input: Validation for patient demographics and risk factors
"""

# Import public validation functions
from .patient_input import (
    validate_ascvd_input,
    validate_bp_input,
    validate_cha2ds2vasc_input,
    validate_ecg_rhythm_input,
    validate_ecg_12lead_input
)

# Define public API
__all__ = [
    "validate_ascvd_input",
    "validate_bp_input",
    "validate_cha2ds2vasc_input",
    "validate_ecg_rhythm_input",
    "validate_ecg_12lead_input"
]

# Package metadata
__version__ = "0.1.0"
__author__ = "Cerevia Inc."
__license__ = "Proprietary"
__email__ = "research@cerevia.ai"