import dellve.config
import json, yaml
import os
import pytest
import tempfile

def test_load_json():
    """
    @brief      Tests dellve.config.load_json function.

    @param        JSON config file fixture.
    """
    file_handle, file_name = tempfile.mkstemp(suffix=".json")
    with open(file_name, 'w') as file:
        json.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        dellve.config.load_json(file)
        assert dellve.config.get('key') == 'value'
    os.remove(file_name)

def test_load_yaml():
    """
    @brief      Tests dellve.config.load_yaml function.

    @param      json_file  YAML config file fixture.
    """
    file_handle, file_name = tempfile.mkstemp(suffix=".yaml")
    with open(file_name, 'w') as file:
        yaml.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        dellve.config.load_yaml(file)
        assert dellve.config.get('key') == 'value'
    os.remove(file_name)

def test_load_generic_json():
    """
    @brief      Tests dellve.config.load with JSON config file.

    @param      json_file  JSON config file fixture.
    @param      yaml_file  YAML config file fixture.
    """
    file_handle, file_name = tempfile.mkstemp(suffix=".json")
    with open(file_name, 'w') as file:
        json.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        dellve.config.load(file)
        assert dellve.config.get('key') == 'value'
    os.remove(file_name)

def test_load_generic_yaml():
    """
    @brief      Tests dellve.config.load with YAML config file.

    @param      yaml_file  YAML config file fixture.
    """
    file_handle, file_name = tempfile.mkstemp(suffix=".yaml")
    with open(file_name, 'w') as file:
        json.dump({'key': 'value'}, file)
    with open(file_name, 'r') as file:
        dellve.config.load(file)
        assert dellve.config.get('key') == 'value'

