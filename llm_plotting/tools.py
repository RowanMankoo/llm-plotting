# Import things that are needed generically
import base64
import logging
from io import StringIO
from typing import Optional, Type, List, Union

import pandas as pd
import requests
from e2b import Sandbox
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from llm_plotting.settings import Settings, AgentSettings
from llm_plotting.prompts import generate_validation_llm_messages
from llm_plotting.prompts import image_save_path, code_validation_tool_description
import streamlit as st

Logger = logging.getLogger(st.__name__)


class NamedStringIO(StringIO):
    """Custom StringIO class with name attribute to circumvent e2b .upload_file()
    method needing file to be saved locally.
    """

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop("name", "untitled")
        super().__init__(*args, **kwargs)


class CodeValidationToolInput(BaseModel):
    code: str = Field(description="python code to generate plots")
    description: str = Field(
        description="description of the plot in context of the data"
    )


class SandboxExecutionError(Exception):
    pass


# TODO: add logging
# TODO: add pydantic validation as method here
class CodeValidationTool(BaseTool):
    name = "CodeValidationTool"
    description = code_validation_tool_description
    args_schema: Type[BaseModel] = CodeValidationToolInput

    binary = "python3"
    filepath = "/index.py"
    image_in_base64_history: List[str] = []

    settings = Settings()
    temperature = 0.0
    df = pd.DataFrame()

    def _run(
        self,
        code: str,
        description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        try:
            image_in_base64 = self._execute_code(code)
        except SandboxExecutionError:
            return "Error: code failed to execute"

        self.image_in_base64_history.append(image_in_base64)
        return self._validate_image(image_in_base64, description, code)

    def _execute_code(self, code: str) -> str:
        # TODO: test case where this fails

        with Sandbox(
            template="my-agent-sandbox-test", api_key=self.settings.e2b_api_key
        ) as sandbox:
            self._upload_df_to_sandbox(self.df, sandbox)
            code = self._modify_code(code)
            sandbox.filesystem.write(self.filepath, code)

            output = sandbox.process.start_and_wait(
                cmd=f"{self.binary} {self.filepath}"
            )

            if output.exit_code != 0:
                Logger.info(f"E2B Sandbox failed to execute code: {code}")
                Logger.info(f"E2B Sandbox stderr: {output.stderr}")
                raise SandboxExecutionError

            Logger.info(f"E2B Sandbox stdout: {output.stdout}")
            Logger.info(f"E2B Sandbox stderr: {output.stderr}")

            image_in_bytes = sandbox.download_file(
                "/home/user/{}".format(image_save_path)
            )
            image_in_base64 = base64.b64encode(image_in_bytes).decode("utf-8")

            return image_in_base64

    def _modify_code(self, code):

        # line to remove df.csv after reading it in
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "df = pd.read_csv('df.csv')":
                # Insert new lines after the current line
                lines[i : i + 1] = [line, "import os", "os.remove('df.csv')"]
                break

        # line to save the figure
        lines.extend(
            [
                "import plotly.io as pio",
                f"pio.write_image(fig, '{image_save_path}')",
            ]
        )

        return "\n".join(lines)

    def _upload_df_to_sandbox(self, df, sandbox):
        # TODO: add functionailty to upload multiple df's
        # Convert DataFrame to CSV format in memory
        csv_buffer = NamedStringIO(name="df.csv")
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Go to the start of the StringIO object

        # Upload CSV to sandbox
        sandbox.upload_file(csv_buffer)

    def _validate_image(self, image_in_base64: str, description: str, code: str) -> str:

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.openai_api_key}",
        }

        # TODO: figure out max_tokens
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": generate_validation_llm_messages(
                image_in_base64, description, code
            ),
            "max_tokens": 300,
            "temperature": self.temperature,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _arun(
        self,
        code: str,
        description: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """operations of tool are sequential and depend on eachother so no async implmentation."""
        output = self._run(code, description)

        return output
