import click.testing
import dellve.benchmark
import dellve.cli
import dellve.config
import dellve.config
import pytest
import tempfile
import time
import uuid

@pytest.fixture(scope="module",
                params=[0, 1, 2, 10])
def benchmarks(request):
    benchmarks = []

    for count in range(0, request.param):
        class Benchmark(dellve.benchmark.Benchmark):

            name = 'Benchmark%d' % count
            config = dellve.benchmark.BenchmarkConfig([])
            def routine(self):
                for p in range(0, 101):
                    self.progress = p
                    time.sleep(0.05)

        benchmarks.append(Benchmark)

    yield benchmarks

@pytest.fixture()
def runner():
    return click.testing.CliRunner()

def test_start_status_stop(runner, benchmarks):
    """Tests CLI 'start' and 'stop' command"""

    # Load fixture benchmarks
    dellve.config.set('benchmarks', benchmarks)

    # Invoke CLI 'start' command
    result = runner.invoke(dellve.cli.start)
    assert result.output == ''
    assert result.exit_code == -1

    # Invoke CLI 'status' command
    result = runner.invoke(dellve.cli.status)
    assert result.output == ''
    assert result.exit_code == -1

    # Invoke CLI 'stop' command
    result = runner.invoke(dellve.cli.stop)
    assert result.output == ''
    assert result.exit_code == -1

'''
TODO: rewrite tests to use HTTP API to query benchmarks.
Possibly have a mock API.
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
'''
