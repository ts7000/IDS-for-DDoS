FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make sure the logs directory exists
RUN mkdir -p logs

# Create template directories for the dashboard
RUN mkdir -p templates static/css static/js

# Create a basic dashboard.html template if it doesn't exist
RUN if [ ! -f templates/dashboard.html ]; then \
    echo '<!DOCTYPE html><html><head><title>Network Attack Monitoring Dashboard</title></head><body><h1>Network Attack Monitoring Dashboard</h1><div id="stats"></div><script src="/static/js/dashboard.js"></script></body></html>' > templates/dashboard.html; \
    fi

# Create a basic dashboard.js file if it doesn't exist
RUN if [ ! -f static/js/dashboard.js ]; then \
    echo 'document.addEventListener("DOMContentLoaded", function() { document.getElementById("stats").innerHTML = "Dashboard is running. Connect to WebSocket for real-time updates."; });' > static/js/dashboard.js; \
    fi

# Environment variables
ENV PORT=8080
ENV MODEL_PATH="model.h5"
ENV SCALER_PATH="scaler.json"

# Expose the port that Cloud Run expects
EXPOSE 8080

# Command to run the application
CMD exec python app.py