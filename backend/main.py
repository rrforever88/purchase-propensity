import json
import random
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_DIR = Path(__file__).parent / "models"

ml_model = None
scaler = None
feature_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, scaler, feature_names
    ml_model = joblib.load(MODEL_DIR / "propensity_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "feature_config.json") as f:
        config = json.load(f)
    feature_names = config["feature_names"]
    yield


app = FastAPI(title="Purchase Propensity API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COMPANY_NAMES = [
    "Apex Solutions", "BlueStar Corp", "Cascade Tech", "Delta Dynamics",
    "Echo Enterprises", "Frontier Systems", "Global Networks", "Horizon Group",
    "Innovate Inc", "Jetstream LLC", "Keystone Partners", "Latitude Co",
    "Meridian Services", "Nexus Technologies", "Orbit Digital", "Pinnacle Ltd",
    "Quantum Data", "Radius Consulting", "Stellar Systems", "Titan Networks",
    "Ultra Solutions", "Vertex Corp", "Wavefront Tech", "Xcel Dynamics",
    "Yellowstone Group", "Zenith Enterprises", "Alphawave", "Bridgepoint",
    "Crestwood IT", "Deepfield Analytics", "Edgewise Networks", "Fulcrum Group",
    "GridPower Systems", "Harbinger Tech", "Integral Solutions", "Junction Co",
    "Knightbridge Ltd", "Lodestar Systems", "Momentum Corp", "Nordic Data",
    "Overture Networks", "Paragon Tech", "Quartermile LLC", "Redstone Group",
    "Summit Analytics", "Tidewater Corp", "Unified Systems", "Vanguard Tech",
    "Westfield Networks", "Zephyr Solutions",
]

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
PAYMENT_METHODS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]
TECH_SUPPORT_VALS = ["Yes", "No", "No internet service"]
ONLINE_SECURITY_VALS = ["Yes", "No", "No internet service"]


def build_feature_row(
    contract: str,
    internet: str,
    payment: str,
    tenure: int,
    monthly: float,
    total: float,
    gender: int,
    senior: int,
    partner: int,
    dependents: int,
    phone: int,
    multi_lines: int,
    online_security: int,
    online_backup: int,
    device_protection: int,
    tech_support: int,
    streaming_tv: int,
    streaming_movies: int,
    paperless: int,
) -> dict:
    return {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PaperlessBilling": paperless,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract_Month-to-month": int(contract == "Month-to-month"),
        "Contract_One year": int(contract == "One year"),
        "Contract_Two year": int(contract == "Two year"),
        "InternetService_DSL": int(internet == "DSL"),
        "InternetService_Fiber optic": int(internet == "Fiber optic"),
        "InternetService_No": int(internet == "No"),
        "PaymentMethod_Bank transfer (automatic)": int(payment == "Bank transfer (automatic)"),
        "PaymentMethod_Credit card (automatic)": int(payment == "Credit card (automatic)"),
        "PaymentMethod_Electronic check": int(payment == "Electronic check"),
        "PaymentMethod_Mailed check": int(payment == "Mailed check"),
    }


class Account(BaseModel):
    account_id: str
    company_name: str
    contract_type: str
    internet_service: str
    tenure_months: int
    monthly_charges: float
    total_charges: float
    tech_support: str
    online_security: str
    propensity_score: int
    risk_tier: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/accounts", response_model=list[Account])
def get_accounts():
    random.seed(42)
    np.random.seed(42)

    accounts = []

    for i in range(50):
        contract = random.choices(
            CONTRACT_TYPES, weights=[0.5, 0.25, 0.25]
        )[0]
        internet = random.choices(
            INTERNET_SERVICES, weights=[0.35, 0.45, 0.20]
        )[0]
        payment = random.choice(PAYMENT_METHODS)

        # Tenure correlated with contract type
        if contract == "Month-to-month":
            tenure = random.randint(1, 36)
        elif contract == "One year":
            tenure = random.randint(12, 48)
        else:
            tenure = random.randint(24, 72)

        # Charges correlated with internet service
        if internet == "Fiber optic":
            monthly = round(random.uniform(70, 110), 2)
        elif internet == "DSL":
            monthly = round(random.uniform(45, 75), 2)
        else:
            monthly = round(random.uniform(20, 45), 2)

        total = round(monthly * tenure * random.uniform(0.95, 1.05), 2)

        no_internet = internet == "No"
        tech_support_raw = (
            "No internet service" if no_internet else random.choice(["Yes", "No"])
        )
        online_security_raw = (
            "No internet service" if no_internet else random.choice(["Yes", "No"])
        )

        gender = random.randint(0, 1)
        senior = random.choices([0, 1], weights=[0.84, 0.16])[0]
        partner = random.randint(0, 1)
        dependents = random.randint(0, 1)
        phone = 1 if internet != "No" else random.randint(0, 1)
        multi_lines = random.randint(0, 1) if phone else 0
        online_security = int(online_security_raw == "Yes")
        online_backup = random.randint(0, 1) if not no_internet else 0
        device_protection = random.randint(0, 1) if not no_internet else 0
        tech_support = int(tech_support_raw == "Yes")
        streaming_tv = random.randint(0, 1) if not no_internet else 0
        streaming_movies = random.randint(0, 1) if not no_internet else 0
        paperless = random.randint(0, 1)

        row = build_feature_row(
            contract=contract,
            internet=internet,
            payment=payment,
            tenure=tenure,
            monthly=monthly,
            total=total,
            gender=gender,
            senior=senior,
            partner=partner,
            dependents=dependents,
            phone=phone,
            multi_lines=multi_lines,
            online_security=online_security,
            online_backup=online_backup,
            device_protection=device_protection,
            tech_support=tech_support,
            streaming_tv=streaming_tv,
            streaming_movies=streaming_movies,
            paperless=paperless,
        )

        df = pd.DataFrame([row], columns=feature_names)
        scaled = scaler.transform(df)
        prob = ml_model.predict_proba(scaled)[0][1]
        score = int(round(prob * 100))

        if score >= 70:
            tier = "High"
        elif score >= 40:
            tier = "Medium"
        else:
            tier = "Low"

        accounts.append(
            Account(
                account_id=str(uuid.UUID(int=random.getrandbits(128))),
                company_name=COMPANY_NAMES[i],
                contract_type=contract,
                internet_service=internet,
                tenure_months=tenure,
                monthly_charges=monthly,
                total_charges=total,
                tech_support=tech_support_raw,
                online_security=online_security_raw,
                propensity_score=score,
                risk_tier=tier,
            )
        )

    return accounts
