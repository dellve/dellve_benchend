
import jinja2
import os


class Template(object):
    def __init__(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data')

        with open(os.path.join(data_path, 'jinja2/setup.py.j2'), 'r') as f:
            self._setup_py_template = jinja2.Template(f.read())
        with open(os.path.join(data_path, 'jinja2/module.py.j2'), 'r') as f:
            self._module_py_template = jinja2.Template(f.read())
        with open(os.path.join(data_path, 'jinja2/__init__.py.j2'), 'r') as f:
            self._init_py_template = jinja2.Template(f.read())

    def render(self, dir_name, package_name, benchmark_name):
        dir_save = os.getcwd()

        package_dir = os.path.join(dir_name, package_name)
        if not os.path.exists(package_dir):
            os.makedirs(package_dir)
        os.chdir(package_dir)

        with open('setup.py', 'w') as f:
            f.write(self._setup_py_template.\
                render( package_name=package_name,
                        benchmark_name=benchmark_name ))

        package_source_dir = os.path.join('./', package_name)
        if not os.path.exists(package_source_dir):
            os.makedirs(package_source_dir)
        os.chdir(package_source_dir)

        with open('__init__.py', 'w') as f:
            f.write(self._init_py_template.\
                render( package_name=package_name,
                        benchmark_name=benchmark_name ))

        with open('benchmark.py', 'w') as f:
            f.write(self._module_py_template.\
                render( package_name=package_name,
                        benchmark_name=benchmark_name ))

        os.chdir(dir_save)
