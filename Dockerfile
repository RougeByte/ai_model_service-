# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Cloud Run expects
ENV PORT=8080
EXPOSE 8080   # 👈 This line is required so Flask binds correctly

# Start the Flask app
CMD ["python", "ai_service.py"]
