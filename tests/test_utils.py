from unittest.mock import mock_open, patch

from llm_plotting.prompt_helper import parse_requirements


def test_parse_requirements():
    mock_file_content = "plotly==5.18\npandas==2.2\nnumpy==1.26\nkaleido==0.2.1\n"

    mock_open_func = mock_open(read_data=mock_file_content)

    # Test the function with names_only=False
    with patch("builtins.open", mock_open_func):
        result = parse_requirements("requirements.txt")
    expected = 'plotly = "5.18"\npandas = "2.2"\nnumpy = "1.26"\nkaleido = "0.2.1"\n'
    assert result == expected

    # Test the function with names_only=True
    with patch("builtins.open", mock_open_func):
        result = parse_requirements("requirements.txt", names_only=True)
    expected = "plotly, pandas, numpy, kaleido"
    assert result == expected
