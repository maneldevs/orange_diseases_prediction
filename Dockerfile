FROM python:3.14-slim-trixie
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app:/home/user/app/scripts
RUN useradd -m -u 1000 user
WORKDIR $HOME/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user ./scripts/predict.py ./scripts/predict.py
COPY --chown=user ./scripts/serve.py ./scripts/serve.py
COPY --chown=user ./models/orange_diseases_model_1.onnx ./models/orange_diseases_model_1.onnx
COPY --chown=user ./models/orange_diseases_model_1.onnx.data ./models/orange_diseases_model_1.onnx.data
COPY --chown=user ./templates ./templates
EXPOSE 7860
CMD ["sh", "-c", "uvicorn scripts.serve:app --host 0.0.0.0 --port ${PORT:-7860}"]