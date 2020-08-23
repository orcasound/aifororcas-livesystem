FROM python:3.7-alpine

COPY ../ /app

WORKDIR /app
RUN pip install -r inference_system/requirements.txt

CMD ["python", "inference_system/predict_and_aggregate.py"]
