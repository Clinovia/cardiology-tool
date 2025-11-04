# cardio-tool/app/validators/patient_input.py
from typing import Dict, Any

# -----------------------------
# GENERAL HELPERS
# -----------------------------
def validate_required_fields(data: Dict[str, Any], fields: list, context: str = "") -> None:
    missing = [f for f in fields if f not in data or data[f] is None]
    if missing:
        ctx = f" in {context}" if context else ""
        raise ValueError(f"Missing required fields{ctx}: {missing}")

def validate_type(value, field, expected_type, context=""):
    if not isinstance(value, expected_type):
        if expected_type is bool:
            raise ValueError(f"{field} must be True or False")
        raise ValueError(f"{field} must be of type {expected_type.__name__} for {context}")

def validate_range(value: float, field: str, min_val: float, max_val: float, inclusive: bool = True, context: str = "") -> None:
    ctx = f" in {context}" if context else ""
    if inclusive:
        if not (min_val <= value <= max_val):
            raise ValueError(f"{field}{ctx} out of range: {value} (expected between {min_val} and {max_val})")
    else:
        if not (min_val < value < max_val):
            raise ValueError(f"{field}{ctx} out of range: {value} (expected >{min_val} and <{max_val})")

def validate_in_set(value: Any, field: str, valid_set: set, context: str = "") -> None:
    if value not in valid_set:
        ctx = f" in {context}" if context else ""
        raise ValueError(f"Invalid value for '{field}'{ctx}: '{value}'. Must be one of {sorted(valid_set)}")


# -----------------------------
# ASCVD INPUT VALIDATION
# -----------------------------
def validate_ascvd_input(data: Dict[str, Any]) -> None:
    required_fields = [
        'age', 'sex', 'race',
        'total_cholesterol', 'hdl',
        'sbp',
        'smoker', 'diabetes',
        'on_htn_meds'
    ]
    validate_required_fields(data, required_fields, "ASCVD")

    # Age 40-79
    validate_type(data['age'], 'age', (int, float), "ASCVD")
    validate_range(data['age'], 'age', 40, 79, context="ASCVD")

    # Sex
    validate_type(data['sex'], 'sex', str, "ASCVD")
    validate_in_set(data['sex'].strip().lower(), 'sex', {'male', 'female'}, "ASCVD")

    # Race
    validate_type(data['race'], 'race', str, "ASCVD")
    validate_in_set(data['race'].strip().lower(), 'race', {'white', 'black'}, "ASCVD")

    # Cholesterol (mg/dL)
    validate_type(data['total_cholesterol'], 'total_cholesterol', (int, float), "ASCVD")
    validate_range(data['total_cholesterol'], 'total_cholesterol', 100, 400, context="ASCVD")

    # HDL
    validate_type(data['hdl'], 'hdl', (int, float), "ASCVD")
    validate_range(data['hdl'], 'hdl', 10, 150, context="ASCVD")

    # Systolic BP (mmHg)
    validate_type(data['sbp'], 'sbp', (int, float), "ASCVD")
    validate_range(data['sbp'], 'sbp', 70, 250, context="ASCVD")

    # Binary flags
    for field in ['smoker', 'diabetes', 'on_htn_meds']:
        val = data[field]
        if not isinstance(val, (bool, int)) or val not in [0, 1, True, False]:
            raise ValueError(f"{field} must be 0 or 1")


# -----------------------------
# BP CATEGORY VALIDATION
# -----------------------------
def validate_bp_input(sbp: float, dbp: float) -> None:
    """
    Validate systolic and diastolic blood pressure for classification.
    Uses field names 'sbp' and 'dbp' consistently.
    """
    # Reject booleans first (they are instances of int/float!)
    if isinstance(sbp, bool):
        raise ValueError("sbp must be a number (int or float), but bool is not allowed")
    if isinstance(dbp, bool):
        raise ValueError("dbp must be a number (int or float), but bool is not allowed")

    # Now check if they are numeric (int or float)
    if not isinstance(sbp, (int, float)):
        raise ValueError("sbp must be a number (int or float)")
    if not isinstance(dbp, (int, float)):
        raise ValueError("dbp must be a number (int or float)")

    # Validate ranges
    if not (50 <= sbp <= 250):
        raise ValueError(f"sbp must be between 50 and 250, but got {sbp}")
    if not (40 <= dbp <= 150):
        raise ValueError(f"dbp must be between 40 and 150, but got {dbp}")

    # Logical check
    if sbp <= dbp:
        raise ValueError(f"sbp ({sbp}) must be greater than dbp ({dbp})")


# -----------------------------
# CHA₂DS₂-VASc VALIDATION
# -----------------------------
def validate_cha2ds2vasc_input(data: Dict[str, Any]) -> None:
    required_fields = [
        'chf', 'hypertension', 'age_ge_75',
        'diabetes', 'stroke', 'vascular',
        'age_65_74', 'female'
    ]
    validate_required_fields(data, required_fields, "CHA2DS2VASc")

    for field in required_fields:
        val = data[field]
        if not isinstance(val, (bool, int)) or val not in [0, 1, True, False]:
            raise ValueError(f"Field '{field}' must be 0/1 or True/False in CHA2DS2VASc")

    if data["age_ge_75"] and data["age_65_74"]:
        raise ValueError("age_ge_75 and age_65_74 cannot both be 1")

# -----------------------------
# ECG VALIDATION
# -----------------------------

def validate_ecg_rhythm_input(data: dict) -> None:
    """
    Validate ECG rhythm input data passed as a dictionary.

    Required:
        - rate (int): Heart rate in bpm, 20–250

    Optional:
        - regular (bool or 0/1): Whether rhythm is regular
        - p_waves_present (bool or 0/1): Whether P-waves are present

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValueError("ECG rhythm input must be a dictionary")

    # -------------------------------
    # 1. Validate: 'rate' (required)
    # -------------------------------
    rate = data.get('rate')
    if rate is None:
        raise ValueError("Missing required field: 'rate'")

    if not isinstance(rate, int):
        raise ValueError(f"rate must be an integer, got {type(rate).__name__}: {rate!r}")
    if rate <= 0:
        raise ValueError(f"rate must be a positive integer, got {rate}")
    if rate < 20 or rate > 250:
        raise ValueError(f"rate in ECG Rhythm out of range: {rate} (expected between 20 and 250)")

    # -----------------------------------------
    # 2. Validate: 'regular' (optional, if present)
    # -----------------------------------------
    regular = data.get('regular')
    if regular is not None:
        if not isinstance(regular, (bool, int)) or regular not in (0, 1):
            raise ValueError(f"regular must be True, False, 0, or 1, got {regular!r}")

    # ---------------------------------------------------
    # 3. Validate: 'p_waves_present' (optional, if present)
    # ---------------------------------------------------
    p_waves_present = data.get('p_waves_present')
    if p_waves_present is not None:
        if not isinstance(p_waves_present, (bool, int)) or p_waves_present not in (0, 1):
            raise ValueError(f"p_waves_present must be True, False, 0, or 1, got {p_waves_present!r}")

def validate_ecg_12lead_input(data: Dict[str, Any]) -> None:
    """
    Validate 12-lead ECG input data.

    Args:
        data (dict): Keys may include:
            - qt_interval_ms (int/float): 300–700 ms
            - rr_interval_ms (int/float): 300–1200 ms
            - pr_interval_ms (int/float): 120–300 ms
            - st_elevation (bool)
            - st_elevation_leads (str): required if st_elevation is True
            - lvh_criteria_met (bool)
            - pathological_q_waves (bool)
            - q_wave_leads (str): required if pathological_q_waves is True
            - t_wave_inversion (bool)

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValueError("12-lead ECG input must be a dictionary")

    # Validate numeric ranges
    for key, minv, maxv in [
        ('qt_interval_ms', 300, 700),
        ('rr_interval_ms', 300, 1200),
        ('pr_interval_ms', 120, 300)
    ]:
        if key in data:
            validate_type(data[key], key, (int, float), "12-lead ECG")
            validate_range(data[key], key, minv, maxv, context="12-lead ECG")

    # Validate boolean fields
    boolean_fields = [
        'st_elevation',
        'lvh_criteria_met',
        'pathological_q_waves',
        't_wave_inversion'
    ]
    for field in boolean_fields:
        if field in data:
            validate_type(data[field], field, bool, "12-lead ECG")

    # --- Validate lead fields: type first (if provided) ---
    st_leads = data.get('st_elevation_leads')
    if 'st_elevation_leads' in data:
        if not isinstance(st_leads, str):
            raise ValueError("st_elevation_leads must be a string")

    q_leads = data.get('q_wave_leads')
    if 'q_wave_leads' in data:
        if not isinstance(q_leads, str):
            raise ValueError("q_wave_leads must be a string")

    # --- Validate conditional presence ---
    if data.get('st_elevation') is True and not st_leads:
        raise ValueError("st_elevation_leads is required when st_elevation is True")
    if data.get('pathological_q_waves') is True and not q_leads:
        raise ValueError("q_wave_leads is required when pathological_q_waves is True")