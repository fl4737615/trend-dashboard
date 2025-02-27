# syntax=docker/dockerfile:1
FROM python:3.10.10-slim as base
ENV PYTHONUNBUFFERED=1

# Set environment variables for caching and data storage
ENV NLTK_DATA=/opt/app/nltk_data
ENV TRANSFORMERS_CACHE=/opt/app/huggingface

WORKDIR /app

# Create directories for persistent caching
RUN mkdir -p $NLTK_DATA $TRANSFORMERS_CACHE

# Copy only the requirements to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --use-pep517 --prefer-binary -r requirements.txt

# Download NLTK data (vader lexicon) into the designated directory
RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='$NLTK_DATA')"

# Copy the rest of your application code
COPY . .

# Expose the port (adjust if needed)
EXPOSE 8000

# Start the application via Gunicorn (using app:app.server so that Dash’s Flask server is served)
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "180", "app:app.server"]
