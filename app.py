import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.io import read_video
import os
import logging
import uuid
import requests
from io import BytesIO
from video_converter import convert_to_avi

from calculators.ascvd import ascvd
from calculators.bp_category import bp_category
from calculators.cha2ds2vasc import cha2ds2vasc
from calculators.ecg_interpret import (
    interpret_rhythm,
    interpret_12lead_findings,
    interpret_ecg_comprehensive,
)
from validators.patient_input import (
    validate_ascvd_input,
    validate_bp_input,
    validate_cha2ds2vasc_input,
    validate_ecg_rhythm_input,
    validate_ecg_12lead_input,
)
from model import EF3DCNN

# -----------------------------
# Logger setupfrom video_converter import convert_to_avi


def run_echonet_ef(video_path, patient_id="N/A"):
    if ef_model is None:
        return "❌ EF model not loaded."

    try:
        video_path = convert_to_avi(video_path.name)  # Convert any upload to .avi
        video_tensor = preprocess_video(video_path)
        with torch.no_grad():
            ef_pred = ef_model(video_tensor.to(device)).item()
        ef_pred = max(0, min(100, ef_pred))

        category = (
            "Normal (≥55%) ✅" if ef_pred >= 55 else
            "Mildly Reduced (45–54%) ⚠️" if ef_pred >= 45 else
            "Moderately Reduced (30–44%) ❗️" if ef_pred >= 30 else
            "Severely Reduced (<30%) 🚨"
        )
        result_full = f"Predicted EF: {ef_pred:.1f}%\nCategory: {category}"
        log_usage("echonet_ef", {"patient_id": patient_id}, result_full)
        return result_full
    except Exception as e:
        log_usage("echonet_ef", {"error": str(e)}, "Error")
        return f"❌ Error processing video: {str(e)}"


# -----------------------------
logging.basicConfig(
    filename="usage.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

def log_usage(tool_name, inputs, outputs=None):
    session_id = str(uuid.uuid4())[:8]
    logging.info(f"session={session_id} tool={tool_name} inputs={inputs} outputs={outputs}")

# -----------------------------
# Load Echonet EF model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EF_MODEL_PATH = "ef3dcnn_epoch17.pth"

# Public S3 URL (replace with your own if needed)
EF_MODEL_URL = "https://s3.us-east-1.amazonaws.com/clinovia.ai/ef3dcnn_epoch17.pth"

# Download model if it doesn't exist
if not os.path.exists(EF_MODEL_PATH):
    try:
        print(f"⬇️ Downloading EF model from {EF_MODEL_URL}...")
        response = requests.get(EF_MODEL_URL, stream=True)
        response.raise_for_status()
        with open(EF_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded EF model to {EF_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to download EF model: {e}")

# Load model
try:
    ef_model = EF3DCNN(in_channels=1, T=32, H=112, W=112).to(device)
    state_dict = torch.load(EF_MODEL_PATH, map_location=device)
    ef_model.load_state_dict(state_dict)
    ef_model.eval()
    print(f"✅ Echonet EF model loaded successfully from {EF_MODEL_PATH}")
except FileNotFoundError:
    ef_model = None
    print(f"⚠️ EF model not found — please ensure it exists at {EF_MODEL_PATH}")
except Exception as e:
    ef_model = None
    print(f"⚠️ Error loading EF model: {e}")

# -----------------------------
# Echonet EF Prediction
# -----------------------------
def preprocess_video(video_path, num_frames=32, resize=(112, 112)):
    """
    Preprocess uploaded echo video:
    - Convert to grayscale
    - Resize to 112x112
    - Sample/pad to num_frames
    - Return tensor [1, 1, T, H, W]
    """
    video, _, _ = read_video(video_path, pts_unit='sec')  # [T, H, W, C]
    video = video.float() / 255.0
    video = video.mean(dim=-1, keepdim=True)  # convert to grayscale [T, H, W, 1]
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
    video = F.interpolate(video, size=resize, mode='bilinear', align_corners=False)

    T_current = video.shape[0]
    if T_current >= num_frames:
        indices = torch.linspace(0, T_current - 1, steps=num_frames).long()
        video = video[indices]
    else:
        pad = num_frames - T_current
        video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)

    return video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 1, T, H, W]

def run_echonet_ef(video_path, patient_id="N/A"):
    if ef_model is None:
        return "❌ EF model not loaded. Please upload ef3dcnn_epoch17.pth."

    try:
        video_tensor = preprocess_video(video_path)
        with torch.no_grad():
            ef_pred = ef_model(video_tensor.to(device)).item()
        ef_pred = max(0, min(100, ef_pred))

        category = (
            "Normal (≥55%) ✅" if ef_pred >= 55 else
            "Mildly Reduced (45–54%) ⚠️" if ef_pred >= 45 else
            "Moderately Reduced (30–44%) ❗️" if ef_pred >= 30 else
            "Severely Reduced (<30%) 🚨"
        )
        result_full = f"Predicted Ejection Fraction (EF): {ef_pred:.1f}%\nCategory: {category}"
        log_usage("echonet_ef", {"patient_id": patient_id}, result_full)
        return result_full
    except Exception as e:
        log_usage("echonet_ef", {"error": str(e)}, "Error")
        return f"❌ Error processing video: {str(e)}"

# -----------------------------
# Other calculators (ASCVD, BP, CHA2DS2-VASc, ECG)
# -----------------------------
def run_ascvd(age, sex, race, total_chol, hdl, sbp, on_htn_meds, smoker, diabetes, patient_id="N/A"):
    data = {"age": age, "sex": sex, "race": race,
            "total_cholesterol": total_chol, "hdl": hdl,
            "sbp": sbp, "on_htn_meds": on_htn_meds,
            "smoker": smoker, "diabetes": diabetes}
    try:
        validate_ascvd_input(data)
        risk = ascvd(data)
        category = ("High" if risk >= 0.20 else
                    "Intermediate" if risk >= 0.075 else
                    "Borderline" if risk >= 0.05 else
                    "Low")
        result_text = f"10-Year ASCVD Risk: {risk:.1%} ({category})"
        log_usage("ascvd", data, result_text)
        return result_text
    except ValueError as e:
        return f"❌ Input Error: {str(e)}"

def run_bp(sbp, dbp, patient_id="N/A"):
    data = {"sbp": sbp, "dbp": dbp}
    try:
        validate_bp_input(sbp, dbp)
        category = bp_category(sbp, dbp)
        result_text = f"Blood Pressure Category: {category}"
        log_usage("bp_category", data, result_text)
        return result_text
    except ValueError as e:
        return f"❌ Input Error: {str(e)}"

def run_cha2ds2vasc(chf, htn, age75, diabetes, stroke, vascular, age65_74, female, patient_id="N/A"):
    data = {"chf": int(chf), "hypertension": int(htn),
            "age_ge_75": int(age75), "diabetes": int(diabetes),
            "stroke": int(stroke), "vascular": int(vascular),
            "age_65_74": int(age65_74), "female": int(female)}
    try:
        validate_cha2ds2vasc_input(data)
        score = cha2ds2vasc(data)
        result_text = f"CHA₂DS₂-VASc Score: {score}"
        log_usage("cha2ds2vasc", data, result_text)
        return result_text
    except ValueError as e:
        return f"❌ Input Error: {str(e)}"

def run_ecg(rate, regular, p_waves_present, st_elev, st_elev_leads, qt, rr, lvh, q_waves, q_leads, t_inversion, pr, patient_id="N/A"):
    rhythm_data = {"rate": rate, "regular": regular, "p_waves_present": p_waves_present}
    lead_data = {
        "st_elevation": st_elev,
        **({"st_elevation_leads": st_elev_leads} if st_elev and st_elev_leads else {}),
        "qt_interval_ms": qt, "rr_interval_ms": rr,
        "lvh_criteria_met": lvh, "pathological_q_waves": q_waves,
        **({"q_wave_leads": q_leads} if q_waves and q_leads else {}),
        "t_wave_inversion": t_inversion, "pr_interval_ms": pr,
    }
    try:
        validate_ecg_rhythm_input(rhythm_data)
        validate_ecg_12lead_input(lead_data)
        rhythm = interpret_rhythm(rate, regular, p_waves_present)
        findings = interpret_12lead_findings(lead_data)
        comprehensive = interpret_ecg_comprehensive({**rhythm_data, **lead_data})
        findings_text = findings["findings"] or "No 12-lead abnormalities detected"
        summary = f"Rhythm: {rhythm}\nFindings: {findings_text}\nOverall Risk: {comprehensive['overall_risk']}"
        log_usage("ecg_interpret", {**rhythm_data, **lead_data}, summary)
        return summary
    except ValueError as e:
        return f"❌ Input Error: {str(e)}"

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Cardio-Tool (RUO)") as demo:
    gr.Markdown("## ⚠️ Cardio-Tool — Research Use Only\nNot for clinical diagnosis.")

    with gr.Tabs():
        # ASCVD
        with gr.TabItem("ASCVD Risk"):
            age = gr.Slider(40, 79, value=60, label="Age")
            sex = gr.Dropdown(["Male", "Female"], label="Sex")
            race = gr.Dropdown(["White", "Black"], label="Race")
            total_chol = gr.Slider(100, 400, value=200, label="Total Cholesterol (mg/dL)")
            hdl = gr.Slider(20, 100, value=50, label="HDL (mg/dL)")
            sbp = gr.Slider(70, 250, value=120, label="Systolic BP (mmHg)")
            htn = gr.Checkbox(label="On HTN Meds?")
            smoker = gr.Checkbox(label="Smoker?")
            diabetes = gr.Checkbox(label="Diabetes?")
            patient_id = gr.Textbox(label="Patient ID", placeholder="Optional")
            out_text = gr.Textbox(label="Result")
            btn = gr.Button("Calculate ASCVD Risk")
            btn.click(run_ascvd,
                      [age, sex, race, total_chol, hdl, sbp, htn, smoker, diabetes, patient_id],
                      out_text)

        # Blood Pressure
        with gr.TabItem("Blood Pressure"):
            sbp2 = gr.Slider(50, 250, value=120, label="Systolic BP (mmHg)")
            dbp2 = gr.Slider(40, 150, value=80, label="Diastolic BP (mmHg)")
            out_text2 = gr.Textbox(label="Result")
            btn2 = gr.Button("Classify BP")
            btn2.click(run_bp, [sbp2, dbp2], out_text2)

        # CHA2DS2-VASc
        with gr.TabItem("CHA₂DS₂-VASc"):
            chf = gr.Checkbox(label="CHF")
            htn_cb = gr.Checkbox(label="Hypertension")
            age75 = gr.Checkbox(label="Age ≥75")
            diabetes2 = gr.Checkbox(label="Diabetes")
            stroke = gr.Checkbox(label="Stroke/TIA/Thromboembolism")
            vascular = gr.Checkbox(label="Vascular Disease")
            age65 = gr.Checkbox(label="Age 65–74")
            female_cb = gr.Checkbox(label="Female")
            out_text3 = gr.Textbox(label="Result")
            btn3 = gr.Button("Calculate CHA₂DS₂-VASc")
            btn3.click(run_cha2ds2vasc,
                       [chf, htn_cb, age75, diabetes2, stroke, vascular, age65, female_cb],
                       out_text3)

        # ECG Interpretation
        with gr.TabItem("ECG Interpretation"):
            rate = gr.Slider(20, 250, value=75, label="Heart Rate (bpm)")
            regular = gr.Checkbox(label="Rhythm Regular?")
            p_waves = gr.Checkbox(label="P Waves Present?")
            st_elev = gr.Checkbox(label="ST Elevation?")
            st_elev_leads = gr.Textbox(label="ST Elevation Leads")
            qt = gr.Slider(300, 700, value=400, label="QT Interval (ms)")
            rr = gr.Slider(300, 1200, value=800, label="RR Interval (ms)")
            lvh = gr.Checkbox(label="LVH Criteria Met?")
            q_waves = gr.Checkbox(label="Pathological Q Waves?")
            q_leads = gr.Textbox(label="Q Wave Leads")
            t_inv = gr.Checkbox(label="T-Wave Inversion?")
            pr = gr.Slider(120, 300, value=160, label="PR Interval (ms)")
            out_text4 = gr.Textbox(label="Result", lines=4)
            btn4 = gr.Button("Interpret ECG")
            btn4.click(run_ecg,
                       [rate, regular, p_waves, st_elev, st_elev_leads, qt, rr, lvh, q_waves, q_leads, t_inv, pr],
                       out_text4)

        # Echonet EF
        with gr.TabItem("Echonet EF"):
            gr.Markdown("### 🫀 Echonet EF Regression (Research Use Only)")
            video_input = gr.Video(label="Upload Echocardiogram Video")
            patient_id_ef = gr.Textbox(label="Patient ID", placeholder="Optional")
            ef_result = gr.Textbox(label="Predicted EF", lines=3)
            ef_button = gr.Button("Predict Ejection Fraction (EF)")
            ef_button.click(run_echonet_ef, [video_input, patient_id_ef], ef_result)

demo.launch(debug=True, share=True)
