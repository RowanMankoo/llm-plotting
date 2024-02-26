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

TECHNICAL_INFO_1 = """
Above we have a high level overview of how this Streamlit app functions. The application \
functions by taking two inputs: a dataframe and a natural language request from the user to \
either query or plot the dataframe. Upon receiving these inputs, the application proceeds to \
extract metadata from the dataframe. This includes information such as column names, shape, \
data types, missing values, and other statistical summaries. This metadata is then stored in a \
JSON format and combined with the natural language request as an input prompt to help gnereate \
the code. This approach was chosen instead of sending the entire dataset to the LLM, as gives \
the LLM a sense of the data without exceeding token limits.

With this input prompt the agent begins to generate code to fulfill the user's request, this \
generated code is then sent to a code validation tool. The agent iterates this process, \
generating and validating code until it produces an output that the validation tool approves. \
All the while the agent steps are being streamed to the user via the Streamlit app. For a more \
detailed view of this workflow, please refer to the diagram below.
"""

TECHNICAL_INFO_2 = f"""
The Agent is using python version 3.10.13 and the following packages to generate all the code:

{parse_requirements("llm_plotting/e2b/requirments.txt")}
"""
