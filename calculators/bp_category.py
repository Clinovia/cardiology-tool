# cardio-tool/app/calculators/bp_category.py

from typing import Dict, Any
from validators.patient_input import validate_bp_input

def bp_category(sbp: int, dbp: int) -> str:
    """
    Classify blood pressure using the 2017 ACC/AHA guideline.

    Reference:
        Whelton PK, Carey RM, et al. 2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA
        Guideline for the Prevention, Detection, Evaluation, and Management of High Blood
        Pressure in Adults. Hypertension. 2018;71(6):e13–e115. doi:10.1161/HYP.0000000000000065

    Categories:
        - Normal
        - Elevated
        - Stage 1 Hypertension
        - Stage 2 Hypertension

    Args:
        sbp (int): Systolic blood pressure (mm Hg)
        dbp (int): Diastolic blood pressure (mm Hg)

    Returns:
        str: BP category with emoji for UI clarity
    """
    # ✅ Validate input first
    validate_bp_input(sbp, dbp)

    if sbp >= 140 or dbp >= 90:
        return "Stage 2 Hypertension"
    elif sbp >= 130 or dbp >= 80:
        return "Stage 1 Hypertension"
    elif sbp >= 120:
        return "Elevated"
    else:
        return "Normal"
