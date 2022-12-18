FROM python:3.11-slim

COPY --chown=root:root src/app /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x run.py

CMD ["python", "run.py"]
