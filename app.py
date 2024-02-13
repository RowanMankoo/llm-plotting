import streamlit as st
from llm_plotting.utils import STAgentInterface, AgentSettings
import asyncio


def main():
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
        agent_settings = AgentSettings(
            max_iterations=max_iterations,
            code_generation_llm_temperature=code_generation_llm_temperature,
        )

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    user_input = st.sidebar.text_area(
        "Describe the plot you wish to construct out of the dataset",
        "make a plot of the average salary of every job, and validate it",
        height=200,
    )

    if st.sidebar.button("Generate Plot"):
        if uploaded_file is not None:
            try:
                st_agent_interface = STAgentInterface(
                    agent_settings, user_input, uploaded_file
                )
                with st.spinner("Generating plot..."):
                    asyncio.run(st_agent_interface.invoke())
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("You must upload a CSV file.")


if __name__ == "__main__":
    main()
