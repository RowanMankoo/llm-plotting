# Define variables
SHELL := /bin/bash
MAX_LINE_SIZE := 120


make pre-lint:
	poetry run black --preview --line-length $(MAX_LINE_SIZE) .
	poetry run isort --profile=black --line-length $(MAX_LINE_SIZE) .
	poetry run flake8 --max-line-length $(MAX_LINE_SIZE) .

make lint:
	poetry run black --preview --line-length $(MAX_LINE_SIZE) --check .
	poetry run isort --profile=black --line-length $(MAX_LINE_SIZE) --check-only .
	poetry run flake8 --max-line-length=$(MAX_LINE_SIZE) .

start_app:
	streamlit run app.py 

use_buildx:
	docker buildx create --use

build_app: use_buildx
	docker buildx build --platform linux/amd64,linux/arm64 -t rowanmankoo/llm-plotting-app:latest . --push

run_app:
	docker run -p 5000:5000 llm-plotting-app:latest

run_tests:
	poetry run pytest tests -v 