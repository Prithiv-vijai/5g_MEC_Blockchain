# Use a lightweight Python image
FROM python:3.10-slim

# Install libgomp1 for LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Set the working directory inside the container
WORKDIR /app

# Copy the application code and model to the container
COPY app.py ./
COPY model1.txt ./
COPY model2.txt ./
COPY mec_config.json .
COPY requirements.txt ./

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports that Flask and Prometheus will run on
EXPOSE 5000  
EXPOSE 8000  

# Run the Flask app
CMD ["python", "app.py"]