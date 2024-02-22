import base64
import textwrap
from io import BytesIO
from typing import Callable, Dict, List

import pandas as pd
import streamlit as st
from pydantic import BaseModel


from llm_plotting.settings import AgentSettings, Settings
from llm_plotting.tools import CodeValidationTool
from llm_plotting.agent import setup_agent_executor
from llm_plotting.prompt_helper import extract_metadata


class STFuncRepr(BaseModel):
    st_func: Callable
    args: List = []
    kwargs: Dict = {}


class STAgentInterface:
    def __init__(
        self,
        settings: Settings,
        agent_settings: AgentSettings,
        uploaded_file: BytesIO,
        execute_st_funcs: bool = True,
    ):
        df = pd.read_csv(uploaded_file)

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

    async def invoke(self, user_input: str):
        chunks = []
        # with st.chat_message("assistant"):
        #     st.write("---")

        async for chunk in self.agent_executor.astream(
            {"user_input": user_input, "metadata_json": self.metadata_json}
        ):
            list_of_st_func_reprs = self.process_chunk(chunk)
            if self.execute_st_funcs:
                for st_func_repr in list_of_st_func_reprs:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "st_func": st_func_repr.st_func,
                            "args": st_func_repr.args,
                            "kwargs": st_func_repr.kwargs,
                        }
                    )
                    with st.chat_message("assistant"):
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
                st_func=st.subheader,
                args=[f"Calling Tool:"],
            ),
            STFuncRepr(
                st_func=st.write,
                args=[f"`{action.tool}` with inputs:"],
            ),
            STFuncRepr(
                st_func=st.markdown,
                args=[f"**Code:**"],
            ),
            STFuncRepr(
                st_func=st.code,
                args=[textwrap.indent(action.tool_input["code"], "    ")],
                kwargs={"language": "python"},
            ),
            STFuncRepr(
                st_func=st.markdown,
                args=[f"**Description:**"],
            ),
            STFuncRepr(
                st_func=st.write,
                args=[f"{action.tool_input['description']}"],
            ),
        ]

    def _process_agent_observation(self, chunk) -> List[STFuncRepr]:
        observation = chunk["steps"][0].observation
        prior_tool_name = chunk["messages"][0].name
        if prior_tool_name == "CodeValidationTool":
            image_in_base64 = self.code_validation_tool.image_in_base64_history[-1]
            image_in_bytes = base64.b64decode(image_in_base64)

            return [
                STFuncRepr(st_func=st.subheader, args=["Tool Result:"]),
                STFuncRepr(st_func=st.image, kwargs={"image": image_in_bytes}),
                STFuncRepr(st_func=st.write, args=[f"{observation}"]),
            ]

    def _process_final_result(self, chunk) -> List[STFuncRepr]:
        return [
            STFuncRepr(st_func=st.subheader, args=["Final Output:"]),
            STFuncRepr(st_func=st.write, args=[f'{chunk["output"]}']),
        ]
