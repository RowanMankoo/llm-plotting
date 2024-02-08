"""Chain that takes in an input and produces an action and action input."""
from __future__ import annotations

import logging
from typing import Any, List, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import Callbacks
from langchain.agents import BaseSingleActionAgent
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)
class CustomRunnableAgent(BaseSingleActionAgent):
    """Agent powered by runnables."""

    runnable: Runnable[dict, Union[AgentAction, AgentFinish]]
    """Runnable to call to get agent action."""
    input_keys_arg: List[str] = []
    return_keys_arg: List[str] = []

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def return_values(self) -> List[str]:
        """Return values of the agent."""
        return self.return_keys_arg

    @property
    def input_keys(self) -> List[str]:
        return self.input_keys_arg

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, **{"intermediate_steps": intermediate_steps}}
        # Use streaming to make sure that the underlying LLM is invoked in a streaming
        # fashion to make it possible to get access to the individual LLM tokens
        # when using stream_log with the Agent Executor.
        # Because the response from the plan is not a generator, we need to
        # accumulate the output into final output and return that.
        final_output: Any = None
        for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
            if final_output is None:
                final_output = chunk
            else:
                final_output += chunk

        return final_output

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[
        AgentAction,
        AgentFinish,
    ]:
        """Based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, **{"intermediate_steps": intermediate_steps}}
        final_output: Any = None
        # Use streaming to make sure that the underlying LLM is invoked in a streaming
        # fashion to make it possible to get access to the individual LLM tokens
        # when using stream_log with the Agent Executor.
        # Because the response from the plan is not a generator, we need to
        # accumulate the output into final output and return that.
        async for chunk in self.runnable.astream(
            inputs, config={"callbacks": callbacks}
        ):
            if final_output is None:
                final_output = chunk
            else:
                final_output += chunk
        return final_output