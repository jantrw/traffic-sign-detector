# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code kopieren
COPY . /app

WORKDIR /app

# Streamlit-Port freigeben
EXPOSE 8501

# Streamlit App starten
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]