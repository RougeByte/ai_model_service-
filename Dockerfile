# Use the official lightweight Python image.
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port used by Flask
EXPOSE 8080

# Run the Flask app
CMD ["python", "ai_service.py"]
