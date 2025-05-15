from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="Customer Segmentation API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# import os
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

static_path = os.path.join(BASE_PATH, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load models and scaler
# base_path = os.path.dirname(__file__)
scaler = joblib.load(os.path.join(BASE_PATH, 'model', 'scaler.pkl'))
kmeans = joblib.load(os.path.join(BASE_PATH, 'model', 'kmeans.pkl'))

class CustomerData(BaseModel):
    age: float
    income: float
    spending_score: float
    membership_years: float
    purchase_frequency: float
    last_purchase_amount: float
    gender: str
    preferred_category: str

@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    with open(os.path.join(static_path, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict")
def predict_cluster(customer: CustomerData):
    # Encode gender
    input_gender = customer.gender.strip().lower()
    gender_male = 1 if input_gender == 'male' else 0
    gender_other = 1 if input_gender == 'other' else 0

    # Encode preferred_category
    input_category = customer.preferred_category.strip().lower()
    category_dummies = [
        1 if input_category == 'fashion' else 0,
        1 if input_category == 'groceries' else 0,
        1 if input_category == 'home & garden' else 0,
        1 if input_category == 'sports' else 0
    ]

    # Combine features
    features = [
        customer.age,
        customer.income,
        customer.spending_score,
        customer.membership_years,
        customer.purchase_frequency,
        customer.last_purchase_amount,
        gender_male,
        gender_other,
        *category_dummies
    ]

    # Convert to DataFrame
    df = pd.DataFrame([features])

    # Scale numeric features
    df.iloc[:, :6] = scaler.transform(df.iloc[:, :6])

    # Predict cluster
    cluster = kmeans.predict(df)[0]

    return {"cluster": int(cluster)}