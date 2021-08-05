import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Response(BaseModel):
    survived: bool
    status: int = 200
    message: str = "success"

app = FastAPI(
    title="API de sobrevivência no Titanic",
    description="Esta API realiza a predição da sobrevivência de uma pessoa no Titanic.",
    contact={
        "name": "Jessica Sousa",
        "url": "https://github.com/Treinamento-Stefa-IA-DevOps/treinamento-dia-04-08-JessicaSousa",
    }
)
@app.post("/model", response_model=Response)
def titanic(sex: int, age: float, lifeboat: int, p_class: int):
    """Prediz se alguém sobreviveria ao acidente do Titanic com base na:
    
    - **sex**: sexo da pessoa, sendo o valor 0 para masculino e 1 para feminino
    - **age**: idade informada em modo fracionário, exemplo: 10.6
    - **lifeboat**: número do barco salva vidas utilizado
    - **p_class**: classe no navio, sendo 1 primeira classe e 3 econômica
    """
    with open("codigo/model/Titanic.pkl", "rb") as fid: 
        clf = pickle.load(fid)
        pred = clf.predict([[sex, age, lifeboat, p_class]])
        return {
            "survived": bool(pred[0]),
            "status": 200,
            "message": "success",
        }
        

@app.get("/")
def get():
    return {"msg": "API de sobrevivência no Titanic"}