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

build_app:
	docker build -t my-app .

run_app:
	docker run -p 5000:5000 my-app:latest

run_tests:
	poetry run pytest tests -v 