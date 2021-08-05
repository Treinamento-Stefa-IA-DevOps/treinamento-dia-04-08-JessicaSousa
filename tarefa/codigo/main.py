import pickle
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse


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
    },
)


@app.post("/model", response_model=Response)
# https://en.wikipedia.org/wiki/Lifeboats_of_the_Titanic
def titanic(
    sex: int = Query(..., alias="Sex", title="Sex", ge=0, le=1),
    age: float = Query(..., alias="Age", title="Age in years", ge=0, le=120),
    lifeboat: int = Query(
        ..., alias="Lifeboat", title="Lifeboat", ge=1, le=20
    ),
    p_class: int = Query(
        ..., alias="Pclass", title="Ticket class", ge=1, le=3
    ),
):
    """Prediz se alguém sobreviveria ao acidente do Titanic com base em:

    - **sex**: sexo da pessoa, sendo o valor 0 para masculino e 1 para feminino
    - **age**: idade informada em modo fracionário, exemplo: 10.6
    - **lifeboat**: número do barco salva vidas utilizado
    - **p_class**: classe no navio, sendo 1 primeira classe e 3 econômica
    """
    try:
        with open("codigo/model/Titanic.pkl", "rb") as fid:
            clf = pickle.load(fid)
            pred = clf.predict([[sex, age, lifeboat, p_class]])
            return {
                "survived": bool(pred[0]),
                "status": 200,
                "message": "success",
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"{e}",
                "status": 500,
                "message": "Internal Server Error.",
            },
        )


@app.get("/")
def get():
    return {"msg": "API de sobrevivência no Titanic"}
