FROM python:3.11-slim

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "rank.py", "--candidates", "sample_candidates.json", "--jd", "job_description.md"]
