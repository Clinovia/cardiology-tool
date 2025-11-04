# cardio-tool/app/calculators/ecg_interpret.py
"""
Rule-based ECG interpretation for research and education.

This module provides deterministic methods for:
- Rhythm analysis (rate, regularity, P-waves)
- 12-lead ECG findings (ST elevation, QTc, LVH, Q waves)

⚠️ WARNING: This is for **Research Use Only (RUO)** and **not for clinical diagnosis**.
It is not a substitute for expert ECG interpretation.

Clinical Basis:
- Rhythm logic aligns with ACC/AHA/HRS guidelines
- 12-lead criteria based on:
  - Goff DC et al. (2013) ACC/AHA Risk Guidelines (for risk context)
  - ACC/AHA/ESC ECG interpretation standards
  - UpToDate: "ECG tutorial", "Approach to arrhythmias"
  - Surawicz B, et al. AHA/ACCF/HRS Recommendations for the Standardization of Electrocardiography

Use this in teaching, simulation, or research workflows — never in patient care.
"""

from typing import Dict, Any, List
from validators.patient_input import validate_ecg_rhythm_input, validate_ecg_12lead_input

# -----------------------------
# 1. RHYTHM INTERPRETATION
# -----------------------------

def interpret_rhythm(rate: int, regular: bool, p_waves_present: bool) -> str:
    """
    Interpret ECG rhythm based on rate, regularity, and P-wave presence.

    Args:
        rate (int): Heart rate in beats per minute (bpm)
        regular (bool): Whether the R-R intervals are regular
        p_waves_present (bool): Whether P-waves are present and morphologically normal

    Returns:
        str: Rhythm interpretation (e.g., "Sinus bradycardia", "Atrial fibrillation")

    Examples:
        >>> interpret_rhythm(75, True, True)
        'Normal sinus rhythm'

        >>> interpret_rhythm(110, False, False)
        'Atrial fibrillation'

        >>> interpret_rhythm(50, True, True)
        'Sinus bradycardia'
    """
    if rate < 60:
        return "Sinus bradycardia"
    elif rate > 100:
        if regular and p_waves_present:
            return "Sinus tachycardia"
        elif not regular and not p_waves_present:
            return "Atrial fibrillation"
        else:
            return "Tachycardia, uncertain rhythm"
    else:
        if regular and p_waves_present:
            return "Normal sinus rhythm"
        elif not regular:
            return "Possible AFib or arrhythmia"
        else:
            return "Undetermined"

def ecg_interpret_detailed(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Extended version with structured output for API use.

    Args:
        data (dict): Keys: 'rate', 'regular', 'p_waves_present'

    Returns:
        dict: With keys 'rhythm', 'confidence', 'notes'
    """
    try:
        validate_ecg_rhythm_input(data)
    except ValueError as e:
        return {
            "rhythm": "Invalid input",
            "confidence": "Low",
            "notes": str(e)
        }

    rate = data.get('rate')
    regular = data.get('regular', False)
    p_waves_present = data.get('p_waves_present', False)

    # Note: validation already passed, so rate should be valid
    rhythm = interpret_rhythm(rate, regular, p_waves_present)

    confidence = "High" if rhythm in [
        "Sinus bradycardia",
        "Normal sinus rhythm",
        "Sinus tachycardia",
        "Atrial fibrillation"
    ] else "Low"

    notes = {
        "Sinus bradycardia": "Assess for athletic training, medications, or conduction disease.",
        "Sinus tachycardia": "Evaluate for fever, pain, anemia, or hyperthyroidism.",
        "Atrial fibrillation": "Confirm with 12-lead ECG; consider anticoagulation.",
        "Tachycardia, uncertain rhythm": "Further analysis needed (e.g., V1 lead).",
        "Possible AFib or arrhythmia": "Irregular rhythm without clear P-waves.",
        "Undetermined": "Insufficient data for interpretation."
    }.get(rhythm, "No specific notes.")

    return {
        "rhythm": rhythm,
        "confidence": confidence,
        "notes": notes
    }

# -----------------------------
# 2. 12LEAD FINDINGS
# -----------------------------

def interpret_12lead_findings(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interpret key 12-lead ECG findings for risk stratification and research.
    """
    try:
        validate_ecg_12lead_input(data)
    except ValueError as e:
        return {
            "findings": [],
            "risk_level": "Low",
            "recommendations": [f"Invalid input: {str(e)}"]
        }

    findings: List[str] = []

    # 1. ST Elevation (STEMI criteria)
    if data.get('st_elevation') and data.get('st_elevation_leads'):
        leads = data['st_elevation_leads']
        findings.append(f"ST elevation in {leads} → Possible anterior/inferior STEMI")

    # 2. QTc Prolongation (Bazett's formula)
    qt = data.get('qt_interval_ms')
    rr = data.get('rr_interval_ms')
    if qt and rr:
        try:
            rr_seconds = rr / 1000.0
            qtcor = round(qt / (rr_seconds ** 0.5))
            if qtcor > 500:
                findings.append(f"QTc = {qtcor} ms → High risk of Torsades de Pointes")
            elif qtcor > 450:
                findings.append(f"QTc = {qtcor} ms → Monitor for drug-induced prolongation")
        except (ZeroDivisionError, ValueError):
            pass

    # 3. Left Ventricular Hypertrophy (LVH)
    if data.get('lvh_criteria_met'):
        findings.append("LVH by voltage criteria (e.g., Sokolow-Lyon) → ASCVD risk enhancer")

    # 4. Pathological Q Waves (Prior MI)
    if data.get('pathological_q_waves') and data.get('q_wave_leads'):
        leads = data['q_wave_leads']
        findings.append(f"Pathological Q waves in {leads} → Possible prior MI")

    # 5. PR Interval (First-degree AV block)
    pr = data.get('pr_interval_ms')
    if pr and pr > 200:
        findings.append(f"PR = {pr} ms → First-degree AV block")

    # 6. T-Wave Inversion (Ischemia/Strain)
    if data.get('t_wave_inversion'):
        findings.append("T-wave inversion → Possible ischemia or strain pattern")

    # Risk Level
    risk_level = "Low"
    high_risk_terms = ["STEMI", "Torsades", "prior MI"]
    moderate_risk_terms = ["prolongation", "LVH", "AV block"]

    if any(any(term in f for term in high_risk_terms) for f in findings):
        risk_level = "High"
    elif any(any(term in f for term in moderate_risk_terms) for f in findings):
        risk_level = "Moderate"

    return {
        "findings": findings,
        "risk_level": risk_level,
        "recommendations": [
            "Correlate with clinical picture",
            "Consider troponin if ST changes",
            "Review medication list if QTc > 500ms",
            "Assess for symptoms if high-risk findings present"
        ]
    }


# -----------------------------
# 3. COMBINED INTERPRETATION
# -----------------------------

def interpret_ecg_comprehensive(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine rhythm and 12-lead findings into a unified ECG summary.

    Useful for research reports and educational tools.

    Args:
        data (dict): Combined rhythm and 12-lead inputs

    Returns:
        dict: Full ECG interpretation
    """
    rhythm_data = {
        'rate': data.get('rate'),
        'regular': data.get('regular', False),
        'p_waves_present': data.get('p_waves_present', False)
    }
    rhythm_result = ecg_interpret_detailed(rhythm_data)
    twelve_lead_result = interpret_12lead_findings(data)

    return {
        "rhythm": {
            "interpretation": rhythm_result["rhythm"],
            "confidence": rhythm_result["confidence"],
            "notes": rhythm_result["notes"]
        },
        "12_lead_findings": twelve_lead_result["findings"],
        "overall_risk": twelve_lead_result["risk_level"],
        "recommendations": twelve_lead_result["recommendations"]
    }

