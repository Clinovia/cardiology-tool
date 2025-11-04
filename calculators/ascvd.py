# cardio-tool/app/calculators/ascvd.py

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Any

from validators.patient_input import validate_ascvd_input

"""
Calculate 10-year ASCVD risk using the 2013 ACC/AHA Pooled Cohort Equations.

Reference:
    Goff DC Jr, Lloyd-Jones DM, et al. 2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk.
    Circulation. 2014;129(25_suppl_1):S49-S73. doi:10.1161/01.cir.0000437741.48606.98

Notes:
    - Valid for ages 40–79
    - Race groups: White, Black
    - Uses separate coefficients for treated vs. untreated systolic BP

Args:
    data (dict): Patient data with keys:
        - age (int)
        - sex (str): "Male" or "Female"
        - race (str): "White" or "Black"
        - total_cholesterol (mg/dL)
        - hdl (mg/dL)
        - systolic_bp (mmHg)
        - on_htn_meds (bool)
        - smoker (bool)
        - diabetes (bool)

Returns:
    float: 10-year ASCVD risk as percentage (0.0–100.0), or None if invalid input
"""

def common_terms(x: Dict[str, Any]) -> Dict[str, Any]:
    ln_age = math.log(x["age"])
    return {
        "ln_age": ln_age,
        "ln_age_sq": ln_age * ln_age,
        "ln_tc": math.log(x["total_cholesterol"]),
        "ln_hdl": math.log(x["hdl"]),
        "ln_sbp": math.log(x["sbp"]),
        "trt": 1 if x.get("on_htn_meds", False) else 0,
        "smoker": 1 if x.get("smoker", False) else 0,
        "diabetes": 1 if x.get("diabetes", False) else 0,
    }

def terms_white_female(x: Dict[str, Any]) -> Dict[str, float]:
    base = common_terms(x)
    terms = {
        "ln_age": base["ln_age"],
        "ln_age_sq": base["ln_age_sq"],
        "ln_tc": base["ln_tc"],
        "ln_age*ln_tc": base["ln_age"] * base["ln_tc"],
        "ln_hdl": base["ln_hdl"],
        "ln_age*ln_hdl": base["ln_age"] * base["ln_hdl"],
        "smoker": base["smoker"],
        "ln_age*smoker": base["ln_age"] * base["smoker"],
        "diabetes": base["diabetes"],
    }
    if base["trt"]:
        terms["ln_sbp_trt"] = base["ln_sbp"]
    else:
        terms["ln_sbp_untrt"] = base["ln_sbp"]
    return terms


def terms_black_female(x: Dict[str, Any]) -> Dict[str, float]:
    base = common_terms(x)
    terms = {
        "ln_age": base["ln_age"],
        "ln_tc": base["ln_tc"],
        "ln_hdl": base["ln_hdl"],
        "ln_age*ln_hdl": base["ln_age"] * base["ln_hdl"],
        "smoker": base["smoker"],
        "diabetes": base["diabetes"],
    }
    if base["trt"]:
        terms["ln_sbp_trt"] = base["ln_sbp"]
        terms["ln_age*ln_sbp_trt"] = base["ln_age"]*base["ln_sbp"]
    else:
        terms["ln_sbp_untrt"] = base["ln_sbp"]
        terms["ln_age*ln_sbp_untrt"] = base["ln_age"]*base["ln_sbp"]
    return terms

def terms_white_male(x: Dict[str, Any]) -> Dict[str, float]:
    base = common_terms(x)
    terms = {
        "ln_age": base["ln_age"],
        "ln_tc": base["ln_tc"],
        "ln_age*ln_tc": base["ln_age"] * base["ln_tc"],
        "ln_hdl": base["ln_hdl"],
        "ln_age*ln_hdl": base["ln_age"] * base["ln_hdl"],
        "smoker": base["smoker"],
        "ln_age*smoker": base["ln_age"] * base["smoker"],
        "diabetes": base["diabetes"],
    }
    if base["trt"]:
        terms["ln_sbp_trt"] = base["ln_sbp"]
    else:
        terms["ln_sbp_untrt"] = base["ln_sbp"]
    return terms

def terms_black_male(x: Dict[str, Any]) -> Dict[str, float]:
    base = common_terms(x)
    terms = {
        "ln_age": base["ln_age"],
        "ln_tc": base["ln_tc"],
        "ln_hdl": base["ln_hdl"],
        "smoker": base["smoker"],
        "diabetes": base["diabetes"],
    }
    if base["trt"]:
        terms["ln_sbp_trt"] = base["ln_sbp"]
    else:
        terms["ln_sbp_untrt"] = base["ln_sbp"]

    return terms

# ---------- Parameters per group (paste official numbers) ----------

@dataclass(frozen=True)
class PCEParams:
    S0: float           # baseline 10-yr survival for the group
    mean_lp: float      # mean linear predictor in derivation cohort
    betas: Dict[str, float]  # {term_name: coefficient}

TERM_BUILDERS: Dict[Tuple[str, str], Callable[[Dict[str, Any]], Dict[str, float]]] = {
    ("female", "white"): terms_white_female,
    ("male", "white"):   terms_white_male,
    ("female", "black"): terms_black_female,
    ("male", "black"):   terms_black_male,
}

# Fill these from the official tables (names must match the term builders)
PCE_CONSTANTS: Dict[Tuple[str, str], PCEParams] = {
    ("female", "white"): PCEParams(
        S0=0.9665,  # e.g., 0.9665
        mean_lp=-29.18,  # published mean LP for white females
        betas={
            "ln_age": -29.799,
            "ln_age_sq": 4.884,
            "ln_tc": 13.540,
            "ln_age*ln_tc": -3.114,
            "ln_hdl": -13.578,
            "ln_age*ln_hdl": 3.149,
            "ln_sbp_trt": 2.019,
            "ln_sbp_untrt": 1.957,
            "smoker": 7.574,
            "ln_age*smoker": -1.665,
            "diabetes": 0.661,
        },
    ),
    ("female", "black"): PCEParams(
        S0=0.9533,  # e.g., 0.9665
        mean_lp=86.61,  # published mean LP for white females
        betas={
            "ln_age": 17.114,
            "ln_tc": 0.940,
            "ln_hdl": -18.920,
            "ln_age*ln_hdl": 4.475,
            "ln_sbp_trt": 29.291,
            "ln_age*ln_sbp_trt": -6.432,
            "ln_sbp_untrt": 27.820,
            "ln_age*ln_sbp_untrt": -6.087,
            "smoker": 0.691,
            "diabetes": 0.874,
        },
    ),
    ("male", "white"): PCEParams(
        S0=0.9144,  # e.g., 0.9665
        mean_lp=61.18,  # published mean LP for white females
        betas={
            "ln_age": 12.344,
            "ln_tc": 11.853,
            "ln_age*ln_tc": -2.664,
            "ln_hdl": -7.990,
            "ln_age*ln_hdl": 1.769,
            "ln_sbp_trt": 1.797,
            "ln_sbp_untrt": 1.764,
            "smoker": 7.837,
            "ln_age*smoker": -1.795,
            "diabetes": 0.658,
        },
    ),
    ("male", "black"): PCEParams(
        S0=0.8954,  # e.g., 0.9665
        mean_lp=19.54,  # published mean LP for white females
        betas={
            "ln_age": 2.469,
            "ln_tc": 0.302,
            "ln_hdl": -0.307,
            "ln_sbp_trt": 1.916,
            "ln_sbp_untrt": 1.809,
            "smoker": 0.549,
            "diabetes": 0.645,
        },
    ),
}

# ---------- Risk function that adapts per group ----------

def ascvd_pce_risk(patient: Dict[str, Any]) -> float:
    """
    Calculate 10-year ASCVD risk.
    Returns risk as percentage (0.0 to 100.0).
    """
    # ✅ Add 'sbp' BEFORE validation
    x = patient.copy()  # Avoid mutating input
    #x["sbp"] = x["systolic_bp"]

    # ✅ Validate the modified dict that has all required fields
    validate_ascvd_input(x)

    sex = x["sex"].strip().lower()
    race = x["race"].strip().lower()
    race = "black" if race == "black" else "white"

    params = PCE_CONSTANTS[(sex, race)]
    terms = TERM_BUILDERS[(sex, race)](x)

    relevant_betas = {
        k: v for k, v in params.betas.items() if k in terms
    }

    missing = [k for k in relevant_betas.keys() if k not in terms]
    if missing:
        raise ValueError(f"Missing term(s) for group {sex}/{race}: {missing}")

    lp = sum(relevant_betas[name] * terms[name] for name in relevant_betas)
    deviation = lp - params.mean_lp

    if deviation > 709:
        return 100.0
    if deviation < -709:
        return 0.0

    survival = params.S0 ** math.exp(deviation)
    risk = 1.0 - survival
    return round(max(0.0, min(100.0, risk * 100.0)), 2)

def ascvd(data: Dict[str, Any]) -> float:
    """
    Calculate 10-year ASCVD risk using the 2013 ACC/AHA Pooled Cohort Equations.

    Args:
        data (dict): Patient data with keys:
            - age (int): 40–79
            - sex (str): "Male" or "Female"
            - race (str): "White" or "Black"
            - total_cholesterol (float): mg/dL
            - hdl (float): mg/dL
            - systolic_bp (float): mmHg
            - on_htn_meds (bool)
            - smoker (bool)
            - diabetes (bool)

    Returns:
        float: 10-year ASCVD risk as a decimal (e.g., 0.078 = 7.8%)

    Raises:
        ValueError: If input is invalid
    """
    risk_percent = ascvd_pce_risk(data)
    return risk_percent / 100.0  # Convert percentage to decimal