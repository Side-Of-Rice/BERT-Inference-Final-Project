FROM python:3.14.3
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

EXPOSE 8080
CMD ["python", "inference/predict.py"]
