FROM python:3.9-slim
EXPOSE 8501

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app/

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]