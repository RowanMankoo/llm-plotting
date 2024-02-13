# Define variables
SHELL := /bin/bash

# TODO: set this up correctly
build_e2b:
	e2b build --name "my-agent-sandbox-test"

start_app:
	streamlit run app.py 

build_app:
	docker build -t my-app .

run_app:
	docker run -p 5000:5000 my-app:latest