import click
import dellve.config
import dellve.util
import json
import pytest
import responses


@pytest.fixture()
def host():
    return 'dellve.com'


@pytest.fixture()
def port():
    return 1234


def test_api_url(host, port):
    url = dellve.util.api_url('%s/%d/url', 'test', 0, host=host, port=port)
    assert url == 'http://%s:%d/test/0/url' % (host, port)


@responses.activate
def test_api_get_error(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    with pytest.raises(click.ClickException):
        dellve.util.api_get('thisendpointdoesntexist')


@responses.activate
def test_api_post_error(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    with pytest.raises(click.ClickException):
        dellve.util.api_post('thisendpointdoesntexist')


@responses.activate
def test_api_get_benchmark(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    data = {
        'config': {},
        'id': 0,
        'name': 'TestBenchmark'
    }

    # Monkey-patch HTTP server
    responses.add(responses.GET, 'http://%s:%d/benchmark' % (host, port),
                  body=json.dumps(data), content_type='application/json',
                  status=200)

    assert dellve.util.api_get('benchmark').json() == data


@responses.activate
def test_api_get_benchmark_error(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    data = {
        'config': {},
        'id': 0,
        'name': 'TestBenchmark'
    }

    # Monkey-patch broken HTTP server
    responses.add(responses.GET, 'http://%s:%d/benchmark' % (host, port),
                  body=json.dumps(data), content_type='application/json',
                  status=500)

    with pytest.raises(click.ClickException):
        dellve.util.api_get('benchmark')


@responses.activate
def test_api_get_benchmark_progress(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    data = {
        'id': 0,
        'name': 'TestBenchmark',
        'progress': 50,
        'output': []
    }

    # Monkey-patch HTTP server
    responses.add(responses.GET,
                  'http://%s:%d/benchmark/progress' % (host, port),
                  body=json.dumps(data), content_type='application/json',
                  status=200)

    assert dellve.util.api_get('benchmark/progress').json() == data


@responses.activate
def test_api_get_benchmark_progress_error(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    data = {
        'id': 0,
        'name': 'TestBenchmark',
        'progress': 50,
        'output': []
    }

    # Monkey-patch broken HTTP server
    responses.add(responses.GET,
                  'http://%s:%d/benchmark/progress' % (host, port),
                  body=json.dumps(data), content_type='application/json',
                  status=500)

    with pytest.raises(click.ClickException):
        dellve.util.api_get('benchmark/progress')


@responses.activate
def test_api_get_benchmark_start(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    data = {}

    # Monkey-patch HTTP server
    responses.add(responses.POST,
                  'http://%s:%d/benchmark/0/start' % (host, port),
                  body=json.dumps(data), content_type='application/json',
                  status=200)

    assert dellve.util.api_post('benchmark/%d/start', 0).json() == data


@responses.activate
def test_api_get_benchmark_start_error(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    data = {}

    # Monkey-patch broken HTTP server
    responses.add(responses.POST,
                  'http://%s:%d/benchmark/0/start' % (host, port),
                  body=json.dumps(data), content_type='application/json',
                  status=500)

    with pytest.raises(click.ClickException):
        dellve.util.api_post('benchmark/%d/start', 0)


@responses.activate
def test_api_get_benchmark_stop(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    data = {}

    # Monkey-patch HTTP server
    responses.add(responses.POST,
                  'http://%s:%d/benchmark/0/stop' % (host, port),
                  body=json.dumps(data), content_type='application/json',
                  status=200)

    assert dellve.util.api_post('benchmark/%d/stop', 0).json() == data


@responses.activate
def test_api_get_benchmark_stop_error(host, port):
    # Monkey-patch configuration
    dellve.config.set('http-host', host)
    dellve.config.set('http-port', port)

    data = {}

    # Monkey-patch broken HTTP server
    responses.add(responses.POST,
                  'http://%s:%d/benchmark/0/stop' % (host, port),
                  body=json.dumps(data), content_type='application/json',
                  status=500)

    with pytest.raises(click.ClickException):
        dellve.util.api_post('benchmark/%d/stop', 0)


# TODO: add test for dellve.util.DebugLoggingFilter
# TODO: add test for dellve.util.ClickLoggingHandler
