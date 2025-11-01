# Docker is an environment -> Whats the first thing you set up in your env? The high level programming language you want to use
FROM python:3.12-slim

# Set the working directory in the container 
WORKDIR /app

# Install compilers and development libs needed to build and
# Install python packages (numpy, pandas, matplolib)
# with C/C++ extensions for ML and data science
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        python3-dev \
        libfreetype6-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

# Then for you code to run, you have dependencies/packages you use to run something, you need to install them here as well -> Use the reqs file and copy it in the directory here
COPY app-ml/requirements.txt .

# Now run the pip install of the requirements.txt to install the dependencies 
RUN pip install --no-cache-dir -r requirements.txt

# We have the env setup, but and the code to run? 
COPY app-ml/entrypoint ./entrypoint/
COPY app-ml/src/ ./src/
COPY common/ ./common/

# Use IF run Docker file separetely -> but we use docker-compose.yml (use docker volumes for data)
# COPY config/ /config/
# COPY data/ /data/
# COPY models/ /models/

# Set PYTHONPATH so Python can find modules in /app
# /common, and /app/src for imports across the projects
# ENV PYTHONPATH="/app:/common:$PYTHONPATH"
# ENV PYTHONPATH="/app:/src:$PYTHONPATH"
ENV PYTHONPATH="/app:/common:/app/src"

# Use IF run Docker file separetely -> but we use docker-compose.yml (run app from the docker compose)
# CMD ["python", "entrypoint/inference_api.py"]