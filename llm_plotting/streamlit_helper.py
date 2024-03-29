import base64
import logging
import textwrap
from io import BytesIO
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pydantic import BaseModel
from streamlit_modal import Modal

from llm_plotting.agent import setup_agent_executor
from llm_plotting.assets.streamlit_txt import TECHNICAL_INFO_1, TECHNICAL_INFO_2
from llm_plotting.prompt_helper import extract_metadata
from llm_plotting.settings import AgentSettings, Settings
from llm_plotting.tools import CodeValidationTool

Logger = logging.getLogger(st.__name__)


def display_and_get_agent_settings():
    with st.sidebar.expander("Agent Settings"):
        max_iterations = st.number_input("max_iterations", min_value=1, max_value=10, value=4, step=1)
        code_generation_llm_temperature = st.slider(
            "code_generation_llm_temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )

        image_validation_llm_temperature = st.slider(
            "image_validation_llm_temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )
        return AgentSettings(
            max_iterations=max_iterations,
            code_generation_llm_temperature=code_generation_llm_temperature,
            image_validation_llm_temperature=image_validation_llm_temperature,
        )


def display_popup_message():

    modal = Modal(key="modal key", title="Information")

    img_agent_workflow = Image.open("llm_plotting/assets/agent_workflow.png")
    img_agent_workflow_array = np.array(img_agent_workflow)

    img_agent_setup = Image.open("llm_plotting/assets/high_level_agent_setup.png")
    img_agent_setup_array = np.array(img_agent_setup)

    info_button = st.button(label="technical info", key="info_button")
    if info_button:
        with modal.container():
            st.image(
                img_agent_setup_array,
                caption="High Level Agent Setup",
                use_column_width=True,
            )
            st.markdown(TECHNICAL_INFO_1)
            st.image(
                img_agent_workflow_array,
                caption="Agent Workflow",
                use_column_width=True,
            )
            st.markdown(TECHNICAL_INFO_2)
        st.write("done")


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

    @staticmethod
    def store_and_display_message(st_func: Callable, args: List = [], kwargs: Dict = {}, role: str = None):
        st.session_state.messages.append(
            {
                "role": role,
                "st_func": st_func,
                "args": args,
                "kwargs": kwargs,
            }
        )

        STAgentInterface.display_message(st_func, args, kwargs, role)

    @staticmethod
    def display_message(st_func: Callable, args: List = [], kwargs: Dict = {}, role: str = None):
        if role is None:
            st_func(*args, **kwargs)
        else:
            with st.chat_message(role):
                st_func(*args, **kwargs)

    @property
    def code_validation_tool(self):
        return [tool for tool in self.agent_executor.tools if isinstance(tool, CodeValidationTool)][0]

    async def invoke(self, user_input: str):
        chunks = []

        async for chunk in self.agent_executor.astream({"user_input": user_input, "metadata_json": self.metadata_json}):
            list_of_st_func_reprs = self.process_chunk(chunk)
            if self.execute_st_funcs:
                for st_func_repr in list_of_st_func_reprs:
                    STAgentInterface.store_and_display_message(
                        st_func_repr.st_func,
                        st_func_repr.args,
                        st_func_repr.kwargs,
                    )
                STAgentInterface.store_and_display_message(st.write, args=["---"])
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
        except Exception as e:
            Logger.error(f"Error processing chunk: {e}")
            return [STFuncRepr(st_func=st.error, args=["Error rendering chunk"])]

    def _process_agent_action(self, chunk) -> List[STFuncRepr]:
        action = chunk["actions"][0]
        if action.tool != "CodeValidationTool":
            raise ValueError("Tool not recognized")

        return [
            STFuncRepr(
                st_func=st.subheader,
                args=["Calling Tool:"],
            ),
            STFuncRepr(
                st_func=st.write,
                args=[f"`{action.tool}` with inputs:"],
            ),
            STFuncRepr(
                st_func=st.markdown,
                args=["**Code:**"],
            ),
            STFuncRepr(
                st_func=st.code,
                args=[textwrap.indent(action.tool_input["code"], "    ")],
                kwargs={"language": "python"},
            ),
            STFuncRepr(
                st_func=st.markdown,
                args=["**Description:**"],
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
            plotting_code = chunk["steps"][0].action.tool_input["plotting_code"]
            if plotting_code:
                image_in_base64 = self.code_validation_tool.image_in_base64_history[-1]
                image_in_bytes = base64.b64decode(image_in_base64)

                return [
                    STFuncRepr(st_func=st.subheader, args=["Tool Result:"]),
                    STFuncRepr(st_func=st.image, kwargs={"image": image_in_bytes}),
                    STFuncRepr(st_func=st.write, args=[f"{observation}"]),
                ]
            else:
                return [
                    STFuncRepr(st_func=st.subheader, args=["Tool Result:"]),
                    STFuncRepr(st_func=st.write, args=[f"{observation}"]),
                ]

    def _process_final_result(self, chunk) -> List[STFuncRepr]:
        return [
            STFuncRepr(st_func=st.subheader, args=["Final Output:"]),
            STFuncRepr(st_func=st.write, args=[f'{chunk["output"]}']),
        ]
