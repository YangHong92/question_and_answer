# Use python as base image
FROM python:3.6-stretch

# Use working directory /app/model
WORKDIR /app/model

# Copy and install required packages
COPY requirements.txt .
# Use --no-cache-dir to avoid running low on memory
RUN pip install --trusted-host pypi.python.org --no-cache-dir -r requirements.txt

# Copy all the content of current directory to working directory
COPY . .

# Set env variables for Cloud Run
ENV PORT 8000
ENV HOST 0.0.0.0

EXPOSE 8000:8000

# CMD to run uvicorn app after the container is started
CMD ["uvicorn", "qa_model.api:app", "--host", "0.0.0.0", "--port", "8000"]