# TREVIDEON hosted API (Railway). CPU image: uploads, sessions, scans, job queue.
# With TREVIDEON_EXECUTOR=modal the heavy processing runs on a Modal GPU (see modal_app.py);
# with TREVIDEON_EXECUTOR=local it runs on this container's CPU (slow but works).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 SAM2_BUILD_CUDA=0 PYTHONUNBUFFERED=1
COPY requirements.txt requirements-hosted.txt ./
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -r requirements-hosted.txt
RUN mkdir -p models \
    && curl -sSL -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
    && curl -sSL -o models/selfie_multiclass_256x256.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
COPY app.py modal_app.py ./
COPY evm ./evm

ENV TREVIDEON_MODE=hosted \
    TREVIDEON_TMP=/tmp/trevideon \
    TREVIDEON_TTL_MIN=60 \
    TREVIDEON_MAX_UPLOAD_MB=1024 \
    TREVIDEON_MAX_SEGMENT_S=60 \
    TREVIDEON_MAX_HEIGHT=1080 \
    TREVIDEON_EXECUTOR=local
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
