import dellve.api
import dellve.benchmark
import falcon.testing
import json
import multiprocessing as mp
import pytest
import time
import ctypes

start_counter = mp.Value(ctypes.c_int32, 0)
stop_counter = mp.Value(ctypes.c_int32, 0)

class MockBenchmark(dellve.benchmark.Benchmark):
    """
    @brief      Benchmark stuf for unit testing purposes.
    """
    name = 'MockBenchmark'

    config = dellve.benchmark.BenchmarkConfig([('key', 'value')])
    schema = {
        'type': 'object',
        'properties': {
            'key': {
                'description': 'Test config option',
                'type': 'string',
            }
        },
        'required': ['key']
    }
    def routine(self):
        try:
            start_counter.value += 1
            self.progress = 50
            while True:
                time.sleep(1)
        except dellve.benchmark.BenchmarkInterrupt:
            stop_counter.value += 1

def test_get_benchmark():
    """
    @brief      Tests GET '/benchmark' API endpoint.
    """

    # Create REST API client
    dellve_api = dellve.api.HttpAPI([MockBenchmark])
    client = falcon.testing.TestClient(dellve_api)

    # Define expected 'reference' output
    ref = [{
        u'config': MockBenchmark.config,
        u'id': 0,
        u'name': MockBenchmark.name,
        u'schema': MockBenchmark.schema,
    }]

    # Get actual output from API
    res = client.simulate_get('/benchmark')

    # Compare outputs
    assert res.json == ref

def test_get_benchmark_bid_start_stop():
    """
    @brief      Tests GET '/benchmark/{benchmark_id}/[start|stop]' API endpoints
    """

    # Create REST API client
    dellve_api = dellve.api.HttpAPI([MockBenchmark])
    client = falcon.testing.TestClient(dellve_api)

    # Create reference counter
    #
    # Note: every time our mock benchmark is started, it updates 'start_counter'
    #
    start_counter_reference = start_counter.value
    stop_counter_reference = stop_counter.value

    for _ in range(0, 5):
        # Start benchmark through REST API (no body)
        res = client.simulate_post('/benchmark/0/start')
        time.sleep(1) # give benchmark some time to start

        # Update reference counter
        start_counter_reference += 1

        # Make sure benchmark[0] indeed started
        assert start_counter.value == start_counter_reference

        # Stop benchmark through REST API
        res = client.simulate_post('/benchmark/0/stop')
        time.sleep(1) # give benchmark some time to stop

        # Update reference counter
        stop_counter_reference += 1

        # Make sure benchmark[0] indeed stopped
        assert stop_counter.value == stop_counter_reference

    for _ in range(0, 5):
        # Start benchmark through REST API (json body)
        res = client.simulate_post('/benchmark/0/start',
            body=json.dumps({'key': 'value'}))
        time.sleep(1) # give benchmark some time to start

        # Update reference counter
        start_counter_reference += 1

        # Make sure benchmark[0] indeed started
        assert start_counter.value == start_counter_reference

        # Stop benchmark through REST API
        res = client.simulate_post('/benchmark/0/stop')
        time.sleep(1) # give benchmark some time to stop

        # Update reference counter
        stop_counter_reference += 1

        # Make sure benchmark[0] indeed stopped
        assert stop_counter.value == stop_counter_reference

def test_get_benchmark_progress():
    """
    @brief      Tests GET '/benchmark/progress' API endpoint
    """

    # Create REST API client
    dellve_api = dellve.api.HttpAPI([MockBenchmark])
    client = falcon.testing.TestClient(dellve_api)

    # Get progress before benchmark is running
    res = client.simulate_get('/benchmark/progress')
    assert res.json == {
        'id':           None,
        'name':         None,
        'progress':     None,
        'output':       None,
        'running':      None
    }

    # Start benchmark through REST API
    res = client.simulate_post('/benchmark/0/start',
        body=json.dumps({'key': 'value'}))
    time.sleep(1) # give benchmark some time to start

    # Get progress after benchmark is running
    res = client.simulate_get('/benchmark/progress')
    assert res.json['id'] == 0
    assert res.json['name'] == 'MockBenchmark'
    assert res.json['progress'] == 50
    assert res.json['running'] == True

    # Stop benchmark through REST API
    res = client.simulate_post('/benchmark/0/stop')
    time.sleep(1) # give benchmark some time to stop

