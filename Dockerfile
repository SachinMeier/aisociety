# ---------- Stage 1: Build frontend ----------
FROM node:22-slim AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (layer cache)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build the app
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: Python runtime ----------
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Install Python dependencies first (layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the modules needed for the server
COPY highsociety/domain/ ./highsociety/domain/
COPY highsociety/server/ ./highsociety/server/
COPY highsociety/app/ ./highsociety/app/
COPY highsociety/players/ ./highsociety/players/
COPY highsociety/__init__.py highsociety/interfaces.py ./highsociety/

# Copy only the server entry point
COPY scripts/play_local.py ./scripts/play_local.py

# Copy built frontend into the image
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["python", "scripts/play_local.py", "--host", "0.0.0.0", "--port", "8000"]
