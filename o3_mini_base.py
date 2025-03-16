import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Define the protocol schema using Pydantic for validation
class Protocol(BaseModel):
    CISS: str
    CE_1st_pass_Angio: str
    CSF_Drive: str
    CSF_PCA: str
    DIR: str
    DTI_32R: str
    DWI: str
    FLAIR: str
    FLAIR_post_contrast: str
    neck_angiography: str
    T1_Dixon: str
    T1_Dixon_post_contrast: str
    T1_MPRAGE: str
    T1_MPRAGE_post_contrast: str
    PWI: str
    SWI: str
    T1_BB: str
    T1_BB_post_contrast: str
    T1_SPIR: str
    T1_dynamic_contrast_enhanced: str
    T2: str
    T2_Dixon: str
    T2_SPAIR: str
    TOF_MRA: str
    TRAK: str
    TRANCE: str
    TOF_MRA_post_contrast: str
    Reasoning: str


# Define relative paths for data sources
DATA_DIR = "./data"
INPUT_FILE = os.path.join(DATA_DIR, "<INPUT_FILE_NAME>.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "<OUTPUT_FILE_NAME>.csv")

# Load case data
cases = pd.read_csv(INPUT_FILE)

# Combine case descriptions
full_case_desc = [
    f"Age range: {cases['age_range'][i]}\nSex: {cases['sex'][i]}\n"
    f"Medical history: {cases['medical_history'][i]}\n"
    f"Clinical question: {cases['clinical_question'][i]}"
    for i in range(len(cases))
]

# Define system instructions
system_instructions = (
    "You are a senior neuroradiologist tasked with defining a brain MRI protocol for a given clinical case. "
    "Consider the patient's demographics, medical history, and clinical question. Synthesize this information "
    "to define an MRI protocol. For each sequence, indicate 'yes' or 'no'. Adhere to the data schema provided to you. "
    "Include only clinically relevant sequences, avoid redundant or unnecessary sequences. Here is a brief explanation "
    "of the more uncommon sequences. CISS (Constructive Interference in Steady State): A 3D gradient-echo sequence "
    "with high spatial resolution. CE_1st_pass_Angio (Contrast-Enhanced First-Pass Angiography): A dynamic imaging "
    "technique using gadolinium contrast to visualize blood vessels during the first pass of contrast through circulation. "
    "CSF_Drive: A specialized sequence for assessing CSF flow dynamics. CSF_PCA (Phase Contrast Angiography): Measures "
    "pulsatile CSF flow velocities and directions. T1_BB (Black Blood Imaging): Suppresses blood signal for vessel wall imaging. "
    "TRAK_4D (4D Time-Resolved MR Angiography with Keyhole): A 4D (time-resolved) MR angiography technique that captures "
    "the dynamics of blood flow over time. TRANCE_4D (4D Time-Resolved Angiography using Non-Contrast Enhancement): A "
    "non-contrast-enhanced 4D MRA technique, primarily used for imaging vascular structures based on cardiac-triggered sequences. "
    "In the field 'reasoning', indicate your rationale for each sequence included."
)


# Function to query the model
def get_model_output(case_desc):
    response = client.beta.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": "Case Description: " + case_desc},
        ],
        response_format=Protocol,
    )
    return response.choices[0].message.parsed


# Ensure protocol and reasoning columns exist
if "o3_mini_base_protocol" not in cases.columns:
    cases["o3_mini_base_protocol"] = pd.NA
if "o3_mini_base_reasoning" not in cases.columns:
    cases["o3_mini_base_reasoning"] = pd.NA

# Process cases
for i in range(0, len(cases)):
    llm_output = get_model_output(full_case_desc[i])

    # Convert model output to dictionary
    output_dict = llm_output.model_dump()
    reasoning = output_dict.pop("Reasoning", "")

    # Store results in dataframe
    cases.at[i, "o3_mini_base_reasoning"] = reasoning
    cases.at[i, "o3_mini_base_protocol"] = json.dumps(output_dict)

    # Print outputs for verification
    print(f"Case {i} Protocol:", cases.at[i, "o3_mini_base_protocol"])
    print(f"Case {i} Reasoning:", cases.at[i, "o3_mini_base_reasoning"])

# Save updated data back to file
cases.to_csv(OUTPUT_FILE, index=False)
