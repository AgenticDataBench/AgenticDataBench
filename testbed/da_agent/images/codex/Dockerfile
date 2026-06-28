FROM python:3.10-bullseye

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install OpenAI Codex CLI
RUN npm install -g @openai/codex@0.80.0

# Upgrade pip and install Python packages
RUN pip install --upgrade pip
RUN pip install google-cloud-bigquery db-dtypes pandas matplotlib scikit-learn seaborn numpy scipy statsmodels xgboost lightgbm plotly tabulate snowflake-connector-python duckdb openpyxl pyyaml geopandas

# Create a non-root user named after the HOST user, with a uid matching the
# HOST uid (both passed as build args) so bind-mounted /workspace files are
# owned by the host user. Falls back to uid 1000 / name "agent".
ARG HOST_USER=agent
ARG HOST_UID=1000
RUN apt-get update && apt-get install -y sudo && \
    useradd -m -u ${HOST_UID} -s /bin/bash ${HOST_USER} && \
    echo "${HOST_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /home/${HOST_USER}/.codex && \
    chown -R ${HOST_USER}:${HOST_USER} /workspace /home/${HOST_USER} && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${HOST_USER}

CMD ["/bin/bash"]
