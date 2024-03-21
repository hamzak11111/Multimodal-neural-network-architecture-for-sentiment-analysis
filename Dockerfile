FROM python:3.9-slim
WORKDIR /app
COPY app.py model.pt requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENV MODEL_PATH=model.pt
CMD ["python", "app.py"]