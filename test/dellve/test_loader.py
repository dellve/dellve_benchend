import dellve.loader
import pytest

class ConfigModuleStub:

    def __init__(self, data):
        self._data = {}

        self._data.update(data)


    def get(self, name):
        return self._data[name]

    def set(self, name, value):
        self._data[name] = value

test_data = {

}

def test_load_plugins():






