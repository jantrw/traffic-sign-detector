# Dockerfile for Streamlit Traffic Sign Detector

FROM python:3.10-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

# Abhängigkeiten kopieren
COPY requirements.txt requirements.txt

# Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# Code kopieren
COPY . .

# Streamlit-Port freigeben
EXPOSE 8501

# Streamlit App starten
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
