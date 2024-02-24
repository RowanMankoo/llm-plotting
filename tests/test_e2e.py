import asyncio
from io import BytesIO

import pytest

from llm_plotting.streamlit_helper import STAgentInterface


@pytest.mark.e2e
def test_e2e(example_code, df, settings, agent_settings):
    # turn df into BytesIO
    df_simulated_upload = BytesIO()
    df.to_csv(df_simulated_upload, index=False)
    df_simulated_upload.seek(0)

    st_agent_interface = STAgentInterface(
        settings,
        agent_settings,
        example_code,
        df_simulated_upload,
        execute_st_funcs=False,
    )
    _ = asyncio.run(st_agent_interface.invoke())
    assert True
