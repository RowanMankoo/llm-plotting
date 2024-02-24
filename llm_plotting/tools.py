# Import things that are needed generically
import base64
import logging
from io import StringIO
from typing import List, Optional, Type

import pandas as pd
import requests
import streamlit as st
from e2b import Sandbox
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from llm_plotting.prompts import CODE_VALIDATION_TOOL_DESCRIPTION, IMAGE_SAVE_PATH, generate_validation_llm_messages
from llm_plotting.settings import Settings

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
    description: str = Field(description="description of the plot in context of the data")
    plotting_code: bool = Field(
        description="if True, the code is for plotting, if False, the code is for answering questions about the data"
    )


class SandboxExecutionError(Exception):
    pass


class CodeValidationTool(BaseTool):
    name = "CodeValidationTool"
    description = CODE_VALIDATION_TOOL_DESCRIPTION
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
        plotting_code: bool,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        try:
            code_output = self._execute_code(code, plotting_code)
            if not plotting_code:
                return code_output
        except SandboxExecutionError as e:
            return str(e)

        self.image_in_base64_history.append(code_output)
        return self._validate_image(code_output, description, code)

    def _execute_code(self, code: str, plotting_code: bool) -> str:

        with Sandbox(template="my-agent-sandbox-test", api_key=self.settings.e2b_api_key) as sandbox:
            self._upload_df_to_sandbox(self.df, sandbox)
            code = self._modify_code(code) if plotting_code else code
            sandbox.filesystem.write(self.filepath, code)

            output = sandbox.process.start_and_wait(cmd=f"{self.binary} {self.filepath}")

            Logger.info(f"E2B Sandbox stdout: {output.stdout}")
            Logger.info(f"E2B Sandbox stderr: {output.stderr}")

            if output.exit_code != 0:
                raise SandboxExecutionError(f"Failed to execute code with error: {output.stderr}")

            return self._handle_execute_code_output(output, plotting_code, sandbox)

    def _handle_execute_code_output(self, output, plotting_code, sandbox):
        if plotting_code:
            image_in_bytes = sandbox.download_file("/home/user/{}".format(IMAGE_SAVE_PATH))
            return base64.b64encode(image_in_bytes).decode("utf-8")
        else:
            return output.stdout

    def _modify_code(self, code):

        lines = code.split("\n")

        lines.extend(
            [
                "import plotly.io as pio",
                f"pio.write_image(fig, '{IMAGE_SAVE_PATH }')",
            ]
        )

        return "\n".join(lines)

    def _upload_df_to_sandbox(self, df, sandbox):
        csv_buffer = NamedStringIO(name="df.csv")
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Go to the start of the StringIO object

        sandbox.upload_file(csv_buffer)

    def _validate_image(self, image_in_base64: str, description: str, code: str) -> str:

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.openai_api_key}",
        }

        # TODO: figure out max_tokens
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": generate_validation_llm_messages(image_in_base64, description, code),
            "max_tokens": 300,
            "temperature": self.temperature,
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _arun(
        self,
        code: str,
        description: str,
        plotting_code: bool,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """operations of tool are sequential and depend on eachother so no async implmentation."""
        output = self._run(code, description, plotting_code)

        return output
