# Obter imagem oficial do python no Docker
FROM python:3.9.6-slim

# Definir o work directory
WORKDIR /app

# Definir variáveis de ambiente do python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Instalar dependências
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 80

# Copiar o projeto
COPY . .

CMD ["uvicorn", "codigo.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]