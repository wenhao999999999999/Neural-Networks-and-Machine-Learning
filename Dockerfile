FROM python:3.10-slim

# Set workdir
WORKDIR /app

# System deps (optional; add as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python deps (CPU torch by default)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default runtime args (can be overridden)
ENV CONFIG_PATH=config/config.yaml \
    RUN_NAME=container_run \
    USE_FILL=1 \
    USE_SCALER=1

# Entrypoint: train+eval+predict (can be overridden by docker run ...)
CMD ["bash", "-lc", "python main_lstm.py --config $CONFIG_PATH --run_name $RUN_NAME --use_fill_calendar --use_scaler --do_train --do_eval --do_predict --do_plot"]

