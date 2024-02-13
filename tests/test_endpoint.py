from llm_plotting.tools import CodeValidationTool


def test_sandbox_file_removal(settings, example_code, df):

    code_validation_tool = CodeValidationTool(settings=settings, df=df)
    modified_code = code_validation_tool._modify_code(example_code)

    assert "df = pd.read_csv('df.csv')\nimport os\nos.remove('df.csv')" in modified_code
