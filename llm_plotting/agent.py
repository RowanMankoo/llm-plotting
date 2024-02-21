import pandas as pd
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

from llm_plotting.prompts import code_generation_agent_prompt
from llm_plotting.settings import AgentSettings, Settings
from llm_plotting.tools import CodeValidationTool


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


def setup_agent_executor(
    settings: Settings, agent_settings: AgentSettings, df: pd.DataFrame
):

    code_validation_tool = CodeValidationTool(df=df, settings=settings)
    tools = [code_validation_tool]

    llm = ChatOpenAI(
        model_name="gpt-4-0125-preview",
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
