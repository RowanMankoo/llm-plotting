# LLM-plotting

### Agent Prompt
Because OpenAI Function Calling is finetuned for tool usage, we hardly need any instructions on how to reason, or how to output format.

 - decision madew to join code execution and validation into one tool so we don't have to send raw image bytes through LLM to waste tokens

## Setup
- Create a python venv with version 3.10 using either virtualenv or conda
- Install dependencies with poetry install
- Create a .env file in root with `OPENAI_API_KEY=<Your api key>`
- Example dataset taken from https://www.kaggle.com/datasets/hummaamqaasim/jobs-in-data?resource=download


# TODO:

- add snapshot pytests


- can use flasks g to store image data during handlig of a request.