import dellve.config
import json
import os
import pytest
import tempfile


@pytest.fixture
def json_file():
    return tempfile.mkstemp(suffix=".json")


@pytest.fixture
def fail_file():
    return tempfile.mkstemp(suffix=".fail")


def test_load(json_file):
    """Tests dellve.config.load with JSON config file.

    Args:
        json_file (file):  JSON config file fixture.
    """
    file_handle, file_name = json_file
    with open(file_name, 'w') as file:
        json.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        dellve.config.load(file)
        assert dellve.config.get('key') == 'value'
    os.remove(file_name)


def test_load_fail(fail_file):
    """Tests dellve.config.load with non-JSON config file.

    Args:
        fail_file (file): File with extension other than .json.
    """
    file_handle, file_name = fail_file
    with open(file_name, 'w') as file:
        json.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        with pytest.raises(IOError):
            dellve.config.load(file)
