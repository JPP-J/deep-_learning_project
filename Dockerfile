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
    echo 'Running ANN_model_tf.py' && python ANN_model_tf.py && \
    echo 'Running ANN_usage_tf.py' && python ANN_usage_tf.py && \
    echo 'Running ANN_model_pt.py' && python ANN_model_pt.py \
"]
