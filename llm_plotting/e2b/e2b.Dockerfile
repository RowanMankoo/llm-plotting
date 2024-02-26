# You can use most of the Debian-based base images
FROM python:3.10.13-slim

COPY requirments.txt .
# Install plotly
RUN pip install -r requirments.txt
