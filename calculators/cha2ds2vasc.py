# cardio-tool/app/calculators/cha2ds2vasc.py

from typing import Dict, Any
from validators.patient_input import validate_cha2ds2vasc_input

def cha2ds2vasc(data: Dict[str, int]) -> int:
    """
    Calculate CHA₂DS₂-VASc stroke risk score for atrial fibrillation patients.

    Reference:
        Lip GYH, Frison L, Halperin JL, Lane DA. Refining Clinical Risk Stratification
        for Predicting Stroke and Thromboembolism in Atrial Fibrillation Using a
        Novel Risk Factor-Based Approach: The CHA₂DS₂-VASc Score. Chest. 2010;137(4):996-1002.
        doi:10.1378/chest.10-1273

    Components:
        C = Congestive heart failure      → 1 point
        H = Hypertension                  → 1 point
        A = Age ≥75                       → 2 points
        D = Diabetes                      → 1 point
        S = Stroke/TIA/Thromboembolism    → 2 points
        V = Vascular disease              → 1 point
        A = Age 65–74                     → 1 point
        Sc = Sex category (Female)        → 1 point

    Args:
        data (dict): Patient risk factors as binary (0/1) or boolean

    Returns:
        int: Total score (0–9)
    """
    validate_cha2ds2vasc_input(data)

    score = (
        data.get('chf', 0) * 1 +
        data.get('hypertension', 0) * 1 +
        data.get('age_ge_75', 0) * 2 +
        data.get('diabetes', 0) * 1 +
        data.get('stroke', 0) * 2 +
        data.get('vascular', 0) * 1 +
        data.get('age_65_74', 0) * 1 +
        data.get('female', 0) * 1
    )
    return score