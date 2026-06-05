FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to /code
WORKDIR /code

# Copy requirements first to leverage Docker cache
COPY ./backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy backend code
COPY ./backend /code/backend

# Set working directory to the backend folder so imports are resolved correctly
WORKDIR /code/backend

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
