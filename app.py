import asyncio
import logging
import warnings
from io import BytesIO

import nest_asyncio
import pandas as pd
import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_modal import Modal

from llm_plotting.settings import Settings
from llm_plotting.streamlit_helper import STAgentInterface, display_and_get_agent_settings

# TODO: figure this out?
nest_asyncio.apply()
Logger = logging.getLogger(st.__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit_extras.dataframe_explorer")


def main():
    settings = Settings()
    st_agent_interface = None

    st.title("LLM-Plotting Tool")
    st.write(
        """
    Please upload a CSV file and provide a description of the plot you want to create from the dataset.
    The tool will then perform the following steps:

    1. Generate the necessary code to construct the desired plot.
    2. Execute this code within a secure, sandboxed environment to produce the plot.
    3. Validate the quality of the plot by sending the resulting image to a vision-based LLM.

    This process will continue in a loop until the vision-based LLM approves of the plot or until the maximum number
    of iterations is reached
    """
    )
    modal = Modal(key="modal key", title="Information")

    info_button = st.button(label="technical info", key="info_button")
    if info_button:
        with modal.container():
            st.markdown("testtesttesttesttesttesttesttest")
        st.write("done")

    agent_settings = display_and_get_agent_settings()
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

    if st.sidebar.button("Confirm Settings"):
        if uploaded_file is None:
            st.error("You must upload a CSV file before confirming the settings.")
        else:
            st.session_state.uploaded_file = uploaded_file.getvalue()

            uploaded_file.seek(0)
            st_agent_interface = STAgentInterface(settings, agent_settings, uploaded_file)
            st.session_state.st_agent_interface = st_agent_interface
            st.session_state.messages = []

    st.markdown("---")
    st.subheader("Dataset Explorer")
    st.write("Explore the dataset to understand its structure and contents.")

    uploaded_file = st.session_state.get("uploaded_file", None)
    if uploaded_file is not None:
        uploaded_file = BytesIO(uploaded_file)
        filtered_df = dataframe_explorer(pd.read_csv(uploaded_file), case=False)
        st.dataframe(filtered_df)
    else:
        st.markdown("The dataset will be displayed below once you upload a CSV file and confirm the settings.")

    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = message.get("role", None)
        st_func = message["st_func"]
        args = message.get("args", [])
        kwargs = message.get("kwargs", {})

        STAgentInterface.display_message(st_func, args, kwargs, role)

    if user_input := st.chat_input(
        # "Describe the plot you wish to construct out of the dataset",
        "Please describe the plot you wish to construct out of the dataset or ask any questions about the data.",
    ):
        if uploaded_file is not None:
            try:
                STAgentInterface.store_and_display_message(
                    st.markdown,
                    args=[f":green[{user_input}]"],
                    role="user",
                )
                Logger.info("Starting LLM-Plotting Tool")

                if st.session_state.get("st_agent_interface") is not None:
                    st_agent_interface = st.session_state.st_agent_interface
                    with st.spinner("Running Agent..."):
                        asyncio.run(st_agent_interface.invoke(user_input))
                else:
                    st.error("You must confirm the settings before generating the plot.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("You must upload a CSV file.")


if __name__ == "__main__":
    main()
