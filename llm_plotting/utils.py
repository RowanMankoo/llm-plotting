import json
import streamlit as st
import textwrap
import logging
import pdb
import base64
from io import BytesIO
from typing import List, Callable, Dict, Optional
import pdb


import pandas as pd
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel, Field, validator

from llm_plotting.prompts import code_generation_agent_prompt
from llm_plotting.tools import CodeValidationTool
from llm_plotting.settings import Settings, AgentSettings

Logger = logging.Logger(__name__)


class STFuncRepr(BaseModel):
    st_func: Callable
    args: List = []
    kwargs: Dict = {}


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
        self,
        settings: Settings,
        agent_settings: AgentSettings,
        user_input: str,
        uploaded_file: BytesIO,
        execute_st_funcs: bool = True,
    ):
        df = pd.read_csv(uploaded_file)

        self.user_input = user_input
        self.metadata_json = extract_metadata(df)
        self.agent_executor = setup_agent_executor(settings, agent_settings, df)
        self.execute_st_funcs = execute_st_funcs

    @property
    def code_validation_tool(self):
        return [
            tool
            for tool in self.agent_executor.tools
            if isinstance(tool, CodeValidationTool)
        ][0]

    # TODO: turn this into a yield
    async def invoke(self):
        chunks = []

        async for chunk in self.agent_executor.astream(
            {"user_input": self.user_input, "metadata_json": self.metadata_json}
        ):
            list_of_st_func_reprs = self.process_chunk(chunk)
            if self.execute_st_funcs:
                for st_func_repr in list_of_st_func_reprs:
                    st_func_repr.st_func(*st_func_repr.args, **st_func_repr.kwargs)
                st.write("---")
            chunks.append({"chunk": chunk, "st_func_reprs": list_of_st_func_reprs})

        return chunks

    def process_chunk(self, chunk) -> List[STFuncRepr]:
        try:
            if "actions" in chunk:
                list_of_st_func_reprs = self._process_agent_action(chunk)
            elif "steps" in chunk:
                list_of_st_func_reprs = self._process_agent_observation(chunk)
            elif "output" in chunk:
                list_of_st_func_reprs = self._process_final_result(chunk)
            return list_of_st_func_reprs
        except:
            # TODO: think about how to handle this
            return []

    def _process_agent_action(self, chunk) -> List[STFuncRepr]:
        action = chunk["actions"][0]
        if action.tool != "CodeValidationTool":
            raise ValueError("Tool not recognized")

        return [
            STFuncRepr(
                st_func=st.write,
                args=[f"Calling Tool: `{action.tool}` with input:"],
            ),
            STFuncRepr(
                st_func=st.code,
                args=[textwrap.indent(action.tool_input["code"], "    ")],
                kwargs={"language": "python"},
            ),
        ]

    def _process_agent_observation(self, chunk) -> List[STFuncRepr]:
        observation = chunk["steps"][0].observation
        prior_tool_name = chunk["messages"][0].name
        if prior_tool_name == "CodeValidationTool":
            image_in_base64 = self.code_validation_tool.image_in_base64_history[-1]
            image_in_bytes = base64.b64decode(image_in_base64)

            return [
                STFuncRepr(st_func=st.image, kwargs={"image": image_in_bytes}),
                STFuncRepr(st_func=st.write, args=[f"Tool Result: `{observation}`"]),
            ]

    def _process_final_result(self, chunk) -> List[STFuncRepr]:
        return [STFuncRepr(st_func=st.write, args=[f'Final Output: {chunk["output"]}'])]


def execute_st_func_repr(st_func_repr: STFuncRepr):
    st_func_repr.st_func(*st_func_repr.args, **st_func_repr.kwargs)
