# Use slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire spg_pred folder (including app.py and model/)
COPY spg_pred/ ./spg_pred/

# Set the command to run the app
CMD ["gunicorn", "spg_pred.app:app", "--bind", "0.0.0.0:8000"]
