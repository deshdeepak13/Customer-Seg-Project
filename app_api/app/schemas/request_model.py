from pydantic import BaseModel

class UserData(BaseModel):
    age: float
    income: float
    spending_score: float
    membership_years: float
    purchase_frequency: float
    last_purchase_amount: float
    gender: str
    preferred_category: str
