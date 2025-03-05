FROM python:3.10-slim

WORKDIR /app

# get args
ARG BACKEND_PORT

COPY . /app

RUN pip install -r /app/build/requirements.txt
RUN pip install fastapi[standard] uvicorn

EXPOSE ${BACKEND_PORT}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${BACKEND_PORT}"]