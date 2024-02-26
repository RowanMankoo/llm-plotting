import asyncio
from io import BytesIO

import pytest

from llm_plotting.streamlit_helper import STAgentInterface


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e(df, settings, agent_settings):
    # turn df into BytesIO
    df_simulated_upload = BytesIO()
    df.to_csv(df_simulated_upload, index=False)
    df_simulated_upload.seek(0)

    user_input = "Please make a plot of the average salary of the top 20 jobs"

    st_agent_interface = STAgentInterface(
        settings,
        agent_settings,
        df_simulated_upload,
        execute_st_funcs=False,
    )
    _ = await st_agent_interface.invoke(user_input)
    assert True
