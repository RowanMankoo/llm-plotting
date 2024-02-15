from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class ValidationLLMOutput(BaseModel):
    code: str = Field(description="code to generate the plots")
    validation_description: str = Field(
        description="brief description of validation observations"
    )

    # @validator("code")
    # def code_must_be_python(cls, v):
    #     if not v.startswith("```python"):
    #         raise ValueError("code must be python")
    #     return v



# TODO: add parsers in agent steps or tool outputs?
