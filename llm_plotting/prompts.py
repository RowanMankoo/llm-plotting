from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
import pandas as pd
import json


# TODO: add functionaility to use other libs
code_generation_agent_system_template = """
You are very powerful code generation assistant who specalises in generating code to visualise graphs. You should always
validate the code, but you are no good at validating this code yourself. You have a tendency to sometimes try to return code
 without validating it make sure it is validated.
Here are some assumptions you should always follow:

- Should always plot using python and with the plotly library
- The df is stored under df.csv
- Save the figure via `pio.write_image(fig, 'figure.png')` and import plotly.io as pio at the top of the script 
- Provide a brief description of what the plot is about in context of the data

Here is some metadata about the data:
{metadata_json}
"""


validation_llm_prompt_system_template = """
You are a highly skilled validation agent who will be examining plots genreated from pythons plotly library.
You should validate if the plot is legiable and if anything is majorly wrong with it. such examples could be:
- The plot is not legible
- axis labels read to read
- Something is obviously wrong with the plot
If you find anything wrong with the plot, please provide feedback on how to fix it in plain english. 
If the plot is sufficently legible, please return "The plot is legible". 
If not please say in the output to regenerate the code with the changes and run the code through this 
validation tool before returning code to user.
"""

code_generation_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", code_generation_agent_system_template),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{user_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def generate_validation_llm_messages(base64_string: str):
    return [
        {
            "role": "system",
            "content": validation_llm_prompt_system_template,
        },
        # TODO: insert the code description in here?
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please validate this plot I made using the plotly library",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_string}"},
                },
            ],
        },
    ]
