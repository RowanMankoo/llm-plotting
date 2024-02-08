# Import things that are needed generically
import base64
import logging
from io import StringIO
from typing import Optional, Type, List

import pandas as pd
import requests
from e2b import Sandbox
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from llm_plotting.settings import Settings
from llm_plotting.prompts import generate_validation_llm_messages

Logger = logging.Logger(__name__)


class NamedStringIO(StringIO):
    """Custom StringIO class with name attribute to circumvent e2b .upload_file()
    method needing file to be saved locally.
    """

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop("name", "untitled")
        super().__init__(*args, **kwargs)


class CodeValidationToolInput(BaseModel):
    # TODO: figure out right definnition for this so agent can recognise
    code: str = Field(description="python code to generate plots")


# TODO: add logging
# TODO: add pydantic validation as method here
# TODO: if code doesn't execute capture it and return error message
class CodeValidationTool(BaseTool):
    name = "CodeValidationTool"
    description = """
    Must be called everytime you have python code which is responsible for generating an image
    and you wish to validate it
    """
    args_schema: Type[BaseModel] = CodeValidationToolInput

    settings = Settings()
    binary = "python3"
    filepath = "/index.py"
    df = pd.DataFrame()
    image_in_bytes: List[str] = []

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        image_in_bytes = self._execute_code(code)
        return self._validate_image(image_in_bytes)

    def _execute_code(self, code: str) -> str:
        # TODO: handle case where this fails
        sandbox = Sandbox(
            template="Python3-DataAnalysis",
        )
        self._upload_df_to_sandbox(self.df, sandbox)
        # TODO: figure this out adds to much time make docker image?
        package_install = sandbox.process.start("pip install -U kaleido")
        package_install.wait()

        sandbox.filesystem.write(self.filepath, code)

        output = sandbox.process.start_and_wait(cmd=f"{self.binary} {self.filepath}")

        Logger.info(f"E2B Sandbox stdout: {output.stdout}")
        Logger.info(f"E2B Sandbox stderr: {output.stderr}")

        # TODO: ensure this is only defined once when doing prompting on save file name
        image_in_bytes = sandbox.download_file("/home/user/figure.png")

        sandbox.close()

        return image_in_bytes

    def _upload_df_to_sandbox(self, df, sandbox):
        # TODO: add functionailty to upload multiple df's
        # Convert DataFrame to CSV format in memory
        csv_buffer = NamedStringIO(name="df.csv")
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Go to the start of the StringIO object

        # Upload CSV to sandbox
        sandbox.upload_file(csv_buffer)

    def _validate_image(self, image_in_bytes: str) -> str:

        base64_string = base64.b64encode(image_in_bytes).decode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.openai_api_key}",
        }

        # TODO: figure out max_tokens
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": generate_validation_llm_messages(base64_string),
            "max_tokens": 300,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # TODO: figure out how to define this
    async def _arun(
        self, code: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
