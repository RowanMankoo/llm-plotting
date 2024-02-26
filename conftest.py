import pandas as pd
import pytest

from llm_plotting.settings import AgentSettings, Settings


@pytest.fixture
def df():
    return pd.read_csv("jobs_in_data.csv")


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def agent_settings():
    return AgentSettings(
        max_iterations=4,
        code_generation_llm_temperature=0.0,
    )
