FROM python:3.10.13-slim

WORKDIR /app
RUN pip install poetry

COPY pyproject.toml poetry.lock /app/
RUN poetry install

COPY . /app

EXPOSE 5000

CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.port", "5000"]

