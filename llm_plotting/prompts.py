from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from llm_plotting.prompt_helper import parse_requirements


image_save_path = "figure.png"
validation_tool_acceptable_output_message = "The plot is legible"
avaliable_libraries = parse_requirements(
    "llm_plotting/e2b/requirments.txt", names_only=True
)

# TODO: clean up these prompts
code_generation_agent_system_template = f"""
You are very powerful code generation assistant who specalises in generating code to visualise graphs.
Whenever you generate code you must also validate the code however you are no good at this yourself so you must use the validation tool.
Your responses should always be some python code to generate a plot using the plotly library along with instructions on 
whether to use the validaiton tool or not. The only time you should return code is if you have used the validation tool and it says 
{validation_tool_acceptable_output_message}. You have a tendency to sometimes try to return code without validating it make sure it is validated so please validate it
unless you are sure it is legible.

Here are some assumptions you should always follow:

- Should always plot using python and with the plotly library
- The df is stored under df.csv please load it in with `df = pd.read_csv('df.csv')`
- Provide a brief description of what the plot is about in context of the data
- You have access to the following libraries: {avaliable_libraries}

Also when you regenerate the code after validaiton please ensure it is not the same as previous code you have generated

Here is some metadata about the data:
{{metadata_json}}
"""


validation_llm_prompt_system_template = f"""
You are a highly skilled validation agent who will be examining plots generated from pythons plotly library.
You will be given the code along with the image and a description of the plot in context of the data.
You should validate if the plot is legiable and if anything is majorly wrong with it. such examples could be:
- The plot is not legible
- axis labels read to read
- no text is overlapping
- Something is obviously wrong with the plot
If you find anything wrong with the plot, please provide a details on how to fix it in plain english and bullet point format
 and make sure not to return code.
If the plot is sufficently legible, please return {validation_tool_acceptable_output_message} in the feedback 
If from the description of the plot you think legibility issues cannot be resolved with simple adjustments due to the complexity and size of the dataset
please state so in the feedback
"""


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
