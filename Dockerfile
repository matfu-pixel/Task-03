FROM python:3.11-slim

COPY --chown=root:root src/app /root/app
COPY --chown=root:root src/ensembles /root/app/ensembles

WORKDIR /root/app

RUN pip install -r requirements.txt
RUN chmod +x run.py

ENV SECRET_KEY hello

CMD ["python", "run.py"]
