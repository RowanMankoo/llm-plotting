# Define variables
SHELL := /bin/bash

# TODO: set this up correctly
build_e2b:
	e2b build --name "my-agent-sandbox-test"

start_app:
	python -m uvicorn app:app --reload