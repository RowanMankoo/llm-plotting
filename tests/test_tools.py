import pytest

from llm_plotting.prompts import IMAGE_SAVE_PATH
from llm_plotting.tools import CodeValidationTool


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


def test_CodeValidationTool___modify_code(settings, example_code, df):

    code_validation_tool = CodeValidationTool(settings=settings, df=df)
    modified_code = code_validation_tool._modify_code(example_code)

    assert f"import plotly.io as pio\npio.write_image(fig, '{IMAGE_SAVE_PATH }')" in modified_code
