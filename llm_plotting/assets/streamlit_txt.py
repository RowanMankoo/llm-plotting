from llm_plotting.prompt_helper import parse_requirements

MAIN_INSTRUCTIONS = """
Please upload a CSV file and choose the agent settings you wish to initalise the agent with and \
then press confirm settings. Once the settings are confirmed, you will have initialized an LLM \
agent. This agent is capable of:

- Generating Python code to create a plot from the dataset.
- Executing and validating the generated code until it is certain the plot is legible.
- Answering your questions about the dataset you have inputted.

**Please Note**: If you alter the agent settings and press 'Confirm Settings' again, the \
previous chat history will be lost.
"""

TECHNICAL_INFO = f"""
The Agent is using python version 3.10.13 and the following packages to generate all the code:

{parse_requirements("llm_plotting/e2b/requirments.txt")}

Below is a diagram of the agent workflow:
"""
