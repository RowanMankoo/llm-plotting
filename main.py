import json
import logging

import pandas as pd
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

from llm_plotting.prompts import code_generation_agent_prompt
from llm_plotting.settings import Settings
from llm_plotting.tools import CodeValidationTool
from llm_plotting.utils import extract_metadata

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ==================== User setup/input ====================
    settings = Settings()
    user_input = "make a plot of the average salary of every job, and validate it"

    df = pd.read_csv("jobs_in_data.csv")
    metadata_json = extract_metadata(df)

    # ==================== Agent setup ====================

    code_validation_tool = CodeValidationTool(df=df, settings=settings)
    tools = [code_validation_tool]

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", api_key=settings.openai_api_key, temperature=0
    )

    agent = create_openai_functions_agent(llm, tools, code_generation_agent_prompt)

    # TODO: turn handle_parsing_errors off whilst developing
    # TODO: think about config for testing and acc deployment?
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=2,
    )

    # ==================== Run agent ====================
    output = agent_executor.invoke(
        {"user_input": user_input, "metadata_json": metadata_json}
    )

    code_validation_tool = [
        tool for tool in agent_executor.tools if isinstance(tool, CodeValidationTool)
    ][0]
    image_in_bytes_history = code_validation_tool.image_in_bytes_history

    print("done")
