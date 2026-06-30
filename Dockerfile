FROM python:3.14-slim

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","rank.py"]
