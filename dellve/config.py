import os
import os.path
__config = {}
__cache = {}

def load(file):
    """
    @brief      Loads a configuration file with JSON or YAML encoding.

    @param      file  Configuration file object.
    """
    if file.name.endswith('.json'):
        load_json(file)
    elif file.name.endswith(('.yml', '.yaml')):
        load_yaml(file)
    else:
        raise IOError('Couldn\'t load configuration file: %s' % file.name)

def load_yaml(file):
    """
    @brief      Loads a YAML-encoded configuration file.

    @param      file  Configuration file object.

    @return     { description_of_the_return_value }
    """
    import yaml
    data = yaml.load(file)
    if data:
        __config.update(data)

    # Cache directory of configuration file
    __cache['config-file-dir'] = os.path.dirname(os.path.realpath(file.name))

def load_json(file):
    """
    @brief      Loads a JSON-encoded configuration file.

    @param      file  Configuration file object.

    @return     { description_of_the_return_value }
    """
    import json
    data = json.load(file)
    if data:
        __config.update(data)

    # Cache directory of configuration file
    __cache['config-file-dir'] = os.path.realpath(file.name)

def get(name):
    """
    @brief      Gets value of a configuration parameter.

    @param      name   Parameter's name

    @return     Parameter's value
    """
    return __config[name]

def get_path(name):
    path = get(name)
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(__cache['config-file-dir'], path)


