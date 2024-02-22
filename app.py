import asyncio
import logging
import sys

import streamlit as st

from llm_plotting.settings import AgentSettings, Settings
from llm_plotting.streamlit_helper import STAgentInterface
import nest_asyncio

# Apply the patch at the beginning of your script
# TODO: figure this out?
nest_asyncio.apply()
Logger = logging.getLogger(st.__name__)


def main():
    settings = Settings()
    st_agent_interface = None  # Initialize st_agent_interface

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

    with st.sidebar.expander("Agent Settings"):
        max_iterations = st.number_input(
            "max_iterations", min_value=1, max_value=10, value=4, step=1
        )
        code_generation_llm_temperature = st.slider(
            "code_generation_llm_temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )

        image_validation_llm_temperature = st.slider(
            "image_validation_llm_temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )
        agent_settings = AgentSettings(
            max_iterations=max_iterations,
            code_generation_llm_temperature=code_generation_llm_temperature,
            image_validation_llm_temperature=image_validation_llm_temperature,
        )

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

    if st.sidebar.button("Confirm Settings"):
        st_agent_interface = STAgentInterface(settings, agent_settings, uploaded_file)
        st.session_state.st_agent_interface = st_agent_interface
        st.session_state.messages = []

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            message.get("st_func")(
                *message.get("args", []), **message.get("kwargs", {})
            )

    if user_input := st.chat_input(
        # "Describe the plot you wish to construct out of the dataset",
        "Make a plot of the average salary of every job",
    ):
        if uploaded_file is not None:
            try:
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "st_func": st.markdown,
                        "args": [f":green[{user_input}]"],
                    }
                )
                with st.chat_message("user"):
                    st.markdown(f":green[{user_input}]")
                Logger.info("Starting LLM-Plotting Tool")

                if (
                    st.session_state.get("st_agent_interface") is not None
                ):  # Check if st_agent_interface is not None
                    st_agent_interface = st.session_state.st_agent_interface
                    with st.spinner("Generating plot..."):
                        asyncio.run(st_agent_interface.invoke(user_input))
                else:
                    st.error(
                        "You must confirm the settings before generating the plot."
                    )
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("You must upload a CSV file.")


if __name__ == "__main__":
    main()
