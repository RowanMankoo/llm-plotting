from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from llm_plotting.parsers import ValidationLLMOutput
from langchain.output_parsers import PydanticOutputParser

validation_llm_output_parser = PydanticOutputParser(pydantic_object=ValidationLLMOutput)

image_save_path = "figure.png"

# TODO: add functionaility to use other libs
code_generation_agent_system_template = f"""
You are very powerful code generation assistant who specalises in generating code to visualise graphs. You should always
validate the code, but you are no good at validating this code yourself. You have a tendency to sometimes try to return code
 without validating it make sure it is validated.
Here are some assumptions you should always follow:

- Should always plot using python and with the plotly library
- The df is stored under df.csv please load it in with `df = pd.read_csv('df.csv')`
- Provide a brief description of what the plot is about in context of the data

Here is some metadata about the data:
{{metadata_json}}
"""


validation_llm_prompt_system_template = """
You are a highly skilled validation agent who will be examining plots generated from pythons plotly library.
You should validate if the plot is legiable and if anything is majorly wrong with it. such examples could be:
- The plot is not legible
- axis labels read to read
- Something is obviously wrong with the plot
If you find anything wrong with the plot, please provide a details on how to fix it

If the plot is sufficently legible, please return "The plot is legible" in the feedback and return the current code without any changes.

If you modify the code please also say in the response to run this modified code through validation again
"""
#     + validation_llm_output_parser.get_format_instructions()
# )

code_generation_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", code_generation_agent_system_template),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{user_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def generate_validation_llm_messages(base64_string: str, description: str, code: str):
    return [
        {
            "role": "system",
            "content": validation_llm_prompt_system_template,
        },
        # TODO: explore prompt engineering on code description here
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Please validate this plot I made using the plotly library here is a breif description of the plot in context of the data: {description} along with the code to generate the plot: {code}""",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_string}"},
                },
            ],
        },
    ]


code_validation_tool_description = """
Must be called everytime you have python code which is responsible for generating an image
and you wish to validate it, this tool will validate the image and return a description of the observations
along with the code to fix the observations if any are found. 

Please not that any code returned by this tool should be validated again before being used as teh final output to the user
"""
