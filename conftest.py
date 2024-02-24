import pandas as pd
import pytest

from llm_plotting.settings import AgentSettings, Settings


@pytest.fixture
def example_code():
    return (
        "import pandas as pd\n"
        "import plotly.express as px\n"
        "import plotly.io as pio\n\n"
        "# Load the data\n"
        "df = pd.read_csv('df.csv')\n\n"
        "# Calculate the average salary for each job\n"
        "avg_salary = df.groupby('job_title')['salary'].mean().reset_index()\n\n"
        "# Create the plot\n"
        "fig = px.bar(avg_salary, x='job_title', y='salary', title='Average Salary by Job')\n\n"
        "# Save the figure\n"
        "pio.write_image(fig, 'figure.png')"
    )


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
