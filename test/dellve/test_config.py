import dellve.config
import json, yaml
import os
import pytest
import tempfile

@pytest.fixture
def json_file():
    return tempfile.mkstemp(suffix=".json")

@pytest.fixture(scope="module",
                params=['.yaml', '.yml'])
def yaml_file(request):
    yield tempfile.mkstemp(suffix=request.param)

@pytest.fixture
def fail_file():
    return tempfile.mkstemp(suffix=".fail")

def test_load_json(json_file):
    """
    @brief      Tests dellve.config._load_json function.

    @param        JSON config file fixture.
    """
    file_handle, file_name = json_file
    with open(file_name, 'w') as file:
        json.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        dellve.config._load_json(file)
        assert dellve.config.get('key') == 'value'
    os.remove(file_name)

def test_load_yaml(yaml_file):
    """
    @brief      Tests dellve.config._load_yaml function.

    @param      json_file  YAML config file fixture.
    """
    file_handle, file_name = yaml_file
    with open(file_name, 'w') as file:
        yaml.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        dellve.config._load_yaml(file)
        assert dellve.config.get('key') == 'value'
    os.remove(file_name)

def test_load_generic_json(json_file):
    """
    @brief      Tests dellve.config.load with JSON config file.

    @param      json_file  JSON config file fixture.
    @param      yaml_file  YAML config file fixture.
    """
    file_handle, file_name = json_file
    with open(file_name, 'w') as file:
        json.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        dellve.config.load(file)
        assert dellve.config.get('key') == 'value'
    os.remove(file_name)

def test_load_generic_yaml(yaml_file):
    """
    @brief      Tests dellve.config.load with YAML config file.

    @param      yaml_file  YAML config file fixture.
    """
    file_handle, file_name = yaml_file
    with open(file_name, 'w') as file:
        json.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        dellve.config.load(file)
        assert dellve.config.get('key') == 'value'

def test_load_generic_fail(fail_file):
    file_handle, file_name = fail_file
    with open(file_name, 'w') as file:
        json.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        with pytest.raises(IOError):
            dellve.config.load(file)
