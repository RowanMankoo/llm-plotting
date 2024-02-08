from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class CodeAndDescription(BaseModel):
    code: str = Field(description="code to generate the plots")
    description: str = Field(description="brief description of the code and plots")

    # @validator("code")
    # def code_must_be_python(cls, v):
    #     if not v.startswith("```python"):
    #         raise ValueError("code must be python")
    #     return v


code_generation_output_parser = PydanticOutputParser(pydantic_object=CodeAndDescription)
