import asyncio
import logging
import warnings
from io import BytesIO

import nest_asyncio
import pandas as pd
import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer

from llm_plotting.assets.streamlit_txt import MAIN_INSTRUCTIONS
from llm_plotting.settings import Settings
from llm_plotting.streamlit_helper import STAgentInterface, display_and_get_agent_settings, display_popup_message

nest_asyncio.apply()
Logger = logging.getLogger(st.__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit_extras.dataframe_explorer")


def main():
    settings = Settings()
    st_agent_interface = None

    st.title("LLM-Plotting Tool")
    st.write(MAIN_INSTRUCTIONS)
    display_popup_message()

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
    st.write(
        "Explore the dataset to understand its structure and contents. \
        Please note you can also ask the agent directly questions about the dataset."
    )

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
        "Please enter request here",
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
