import click.testing
import dellve.benchmark
import dellve.cli
import dellve.config
import dellve.config
import pytest
import tempfile
import time
import uuid
import yaml

# def test_hello_world():
    # @click.command()
    # @click.argument('name')
    # def hello(name):
    #     click.echo('Hello %s!' % name)

    # runner = CliRunner()
    # result = runner.invoke(hello, ['Peter'])
    # assert result.exit_code == 0
    # assert result.output == 'Hello Peter!\n'

# @click.group()
# @click.option('--config-file', 'config_file',
#     help='Configuration file name.', type=click.File('r'))
# def cli(config_file):
#     """DELLve benchmark command line interface.

#     Type 'dellve COMMAND --help' to see help for commands listed below.
#     """
#     config.load(config_file) # load DELLve configuration

# def test_cat():
#     @click.command()
#     @click.argument('f', type=click.File())
#     def cat(f):
#         click.echo(f.read())

#     runner = CliRunner()
#     with runner.isolated_filesystem():
#         with open('hello.txt', 'w') as f:
#             f.write('Hello World!')

#         result = runner.invoke(cat, ['hello.txt'])
#         assert result.exit_code == 0
#         assert result.output == 'Hello World!\n'

@pytest.fixture(scope="module",
                params=[0, 1, 2, 10])
def benchmarks(request):
    benchmarks = []

    for count in range(0, request.param):
        class Benchmark(dellve.benchmark.Benchmark):

            name = 'Benchmark%d' % count

            def routine(self):
                for p in range(0, 101):
                    self.progress = p
                    time.sleep(0.05)

        benchmarks.append(Benchmark)

    yield benchmarks

@pytest.fixture()
def runner():
    return click.testing.CliRunner()

@pytest.fixture(scope="module",
                params=['.yaml', '.yml'])
def yaml_file(request):
    yield tempfile.mkstemp(suffix=request.param)

# def test_cli_yaml(runner, yaml_file):
#     with runner.isolated_filesystem():
#         file_handle, file_name = yaml_file
#         config_key   = uuid.uuid1()
#         config_value = uuid.uuid1()
#         with open(file_name, 'w') as file:
#             yaml.dump({str(config_key): str(config_value)}, file)
#         print file_name
#         result = runner.invoke(dellve.cli.cli, ['--config-file=%s' % file_name])
#         # assert result.exit_code == 0
#         assert dellve.config.get(str(config_key)) == str(config_value)

def test_ls(runner, benchmarks):
    """Tests CLI 'ls' command

    Args:
        runner (click.testing.CliRunner): Click CLI runner fixture.
        benchmarks (list): Benchmarks fixture.
    """

    # Load fixture benchmarks
    dellve.config.set('benchmarks', benchmarks)

    # Invoke CLI command and get its output
    result = runner.invoke(dellve.cli.ls)
    assert result.exit_code == 0

    # Compare output
    if len(benchmarks):
        # Note: we don't care about the order in which the benchmarks are listed,
        #       but 'ls' command should print all of the benchmarks (no more, no less)
        output_benchmarks_set = {l.strip() for l in result.output.split()}
        assert  output_benchmarks_set == {b.name for b in benchmarks}
    else:
        # Note: we shouldn't get any ouput if there's no installed benchmarks
        assert result.output == ''



