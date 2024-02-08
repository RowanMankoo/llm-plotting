# You can use most of the Debian-based base images
FROM python:3.10.13-slim

# Install plotly
RUN pip install plotly pandas numpy seaborn matplotlib kaleido