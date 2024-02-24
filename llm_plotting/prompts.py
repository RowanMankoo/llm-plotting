from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from llm_plotting.prompt_helper import parse_requirements


IMAGE_SAVE_PATH = "figure.png"
IMAGE_SAVE_PATHVALIDATION_TOOL_ACCEPTABLE_OUTPUT_MESSAGE = "The plot is legible"
AVAILABLE_LIBRARIES = parse_requirements(
    "llm_plotting/e2b/requirments.txt", names_only=True
)

CODE_GENERATION_AGENT_SYSTEM_TEMPLATE = f"""
You are a powerful code generation assistant who specializes in generating code \
to visualize graphs. You can also generate code to answer questions about user \
DataFrames (always stored under df.csv). When answering questions about the data, \
you must always send your answer in a print statement.

Whenever you generate any type of code, you must also validate the code. However, \
you are not good at this yourself, so you must use the validation tool. Whenever \
you decide to generate code to create plots, you must use the Plotly library in Python. \
The only time you should return code is if you have used the validation tool and it says \
{IMAGE_SAVE_PATHVALIDATION_TOOL_ACCEPTABLE_OUTPUT_MESSAGE }.

You have a tendency to sometimes use the validation tool, regenerate code, and then \
return this new code without validating it. Please make sure you make use of the \
validation tool before returning code in this case.

Here are some assumptions you should always follow:
- You should always plot using Python and with the Plotly library.
- The DataFrame is stored under df.csv. Please load it in with `df = pd.read_csv('df.csv')`.
- Provide a brief description of what the plot is about in the context of the data.
- You have access to the following libraries: {AVAILABLE_LIBRARIES }.

Also, when you regenerate the code after validation, please ensure it is not the same \
as the previous code you have generated.

Here is some metadata about the data:
{{metadata_json}}
"""


VALIDATION_LLM_PROMPT_SYSTEM_TEMPLATE = f"""
You are a highly skilled validation agent who will be examining plots generated from Python's Plotly library. \
You will be given the code along with the image and a description of the plot in context of the data.

You should validate if the plot is legible and if anything is majorly wrong with it. Such examples could be:
- The plot is not legible
- Axis labels are hard to read
- Text is overlapping
- Something is obviously wrong with the plot

If you find anything wrong with the plot, please provide details on how to fix it in plain English and bullet point format, \
and make sure not to return code.

If the plot is sufficiently legible, please return {IMAGE_SAVE_PATHVALIDATION_TOOL_ACCEPTABLE_OUTPUT_MESSAGE } in the feedback. \
If, from the description of the plot, you think legibility issues cannot be resolved with simple adjustments due to the complexity and size of the dataset, \
please state so in the feedback.
"""


CODE_GENERATION_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CODE_GENERATION_AGENT_SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{user_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def generate_validation_llm_messages(base64_string: str, description: str, code: str):
    return [
        {
            "role": "system",
            "content": VALIDATION_LLM_PROMPT_SYSTEM_TEMPLATE,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Please validate this plot I made using the Plotly library. "
                        f"Here is a brief description of the plot in context of the data: {description}. "
                        f"Along with the code to generate the plot: {code}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_string}"},
                },
            ],
        },
    ]


CODE_VALIDATION_TOOL_DESCRIPTION = """
This tool must be called every time you generate any Python code which is either responsible for generating \
an image or answering questions about the data and you wish to validate it.

If the plot is an image, this tool will validate the image and return a description of the observations \
along with the code to fix the observations if any are found.

If the code is for answering questions about the data, this tool will validate the code and return the output of the code.
"""
