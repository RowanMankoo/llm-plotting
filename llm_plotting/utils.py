import json
import streamlit as st
import textwrap
import logging
import pdb
import base64
from io import BytesIO


import pandas as pd
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel, Field, validator

from llm_plotting.prompts import code_generation_agent_prompt
from llm_plotting.tools import CodeValidationTool
from llm_plotting.settings import Settings

Logger = logging.Logger(__name__)


class MyStreamingCallback(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.content = ""
        self.final_answer = False

    async def on_lm_new_token(self, token):
        self.content += token
        if "final_answer" in self.content:
            self.final_answer = True
            self.content = ""
        if self.final_answer:
            print(token)


# TODO: should try not to have a utils.py
def extract_metadata(df: pd.DataFrame, col_limit=150):
    if df.shape[1] > col_limit:
        raise ValueError(f"Dataframe exceeds col limit of {col_limit}")

    df = df.copy()
    metadata_dict = {
        "column_names": df.columns.tolist(),
        "data_dimensions": df.shape,
        "data_types_per_column": df.dtypes.apply(lambda x: x.name).to_dict(),
        "statistical_summary": df.describe().to_dict(),
        "missing_values_per_column": df.isnull().sum().to_dict(),
    }
    return json.dumps(metadata_dict)


class AgentSettings(BaseModel):
    max_iterations: int = 5
    code_generation_llm_temperature: float = 0


def setup_agent_executor(
    settings: Settings, agent_settings: AgentSettings, df: pd.DataFrame
):

    code_validation_tool = CodeValidationTool(df=df, settings=settings)
    tools = [code_validation_tool]

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        api_key=settings.openai_api_key,
        temperature=agent_settings.code_generation_llm_temperature,
        streaming=True,
        callbacks=[MyStreamingCallback()],
    )

    # TODO: add memory to agent
    agent = create_openai_functions_agent(llm, tools, code_generation_agent_prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=False,
        handle_parsing_errors=True,
        max_iterations=agent_settings.max_iterations,
    )


# TODO: seperate out logic from presentation of webpage?
class STAgentInterface:
    def __init__(
        self, agent_settings: AgentSettings, user_input: str, uploaded_file: BytesIO
    ):
        df = pd.read_csv(uploaded_file)

        self.user_input = user_input
        self.metadata_json = extract_metadata(df)
        self.agent_executor = setup_agent_executor(Settings(), agent_settings, df)

    @property
    def code_validation_tool(self):
        return [
            tool
            for tool in self.agent_executor.tools
            if isinstance(tool, CodeValidationTool)
        ][0]

    async def invoke(self):
        chunks = []

        async for chunk in self.agent_executor.astream(
            {"user_input": self.user_input, "metadata_json": self.metadata_json}
        ):
            chunks.append(chunk)
            self.process_chunk(chunk)
            st.write("---")

        return chunks

    def process_chunk(self, chunk):
        if "actions" in chunk:
            self._process_agent_action(chunk)
        elif "steps" in chunk:
            self._process_agent_observation(chunk)
        elif "output" in chunk:
            self._process_final_result(chunk)
        else:
            # TODO: think about how to handle this
            raise ValueError("Chunk type not recognized")

    def _process_agent_action(self, chunk):
        action = chunk["actions"][0]
        if action.tool == "CodeValidationTool":
            st.write(f"Calling Tool: `{action.tool}` with input:")
            st.code(textwrap.indent(action.tool_input["code"], "    "))
        else:
            st.write(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")

    def _process_agent_observation(self, chunk):
        observation = chunk["steps"][0].observation
        try:
            prior_tool_name = chunk["messages"][0].name
            if prior_tool_name == "CodeValidationTool":
                image_in_base64 = self.code_validation_tool.image_in_base64_history[-1]
                image_in_bytes = base64.b64decode(image_in_base64)
                st.image(image_in_bytes, caption="fsf")
        except Exception as e:
            pass

        st.write(f"Tool Result: `{observation}`")

    def _process_final_result(self, chunk):
        st.write(f'Final Output: {chunk["output"]}')
