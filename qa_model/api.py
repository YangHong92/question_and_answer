from fastapi import FastAPI, Depends
from pydantic import BaseModel

from .gpt3.model import Model, get_model

app = FastAPI()

class ModelRequest(BaseModel):
    query:str

class ModelResponse(BaseModel):
    answer: str

@app.post('/ask', response_model = ModelResponse)
def ask(request: ModelRequest, model: Model = Depends(get_model())): #dependence injection
    answer = model.answer(request.query)
    return ModelResponse(
        answer = answer
    )