# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Default command to run all required Python scripts
CMD ["bash", "-c", "\
    echo 'Running CNN_model.py' && python CNN_model.py && \
    echo 'Running CNN_usage.py' && python CNN_usage.py  \
"]
