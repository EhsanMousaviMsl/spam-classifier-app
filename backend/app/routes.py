# backend/app/routes.py

from fastapi import APIRouter
from pydantic import BaseModel
from .model import predict_spam

router = APIRouter()

class Message(BaseModel):
    text: str

@router.post("/predict")
def predict(message: Message):
    result = predict_spam(message.text)
    return {"label": result}
