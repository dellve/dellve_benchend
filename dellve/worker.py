import config
import executor
import falcon
import gevent
import gevent.event
import gevent.pywsgi
import helper
import json
import random
import re


class DELLveWorker(object):

    class BenchmarkListRoute:

        url = '/benchmark/'

        def __init__(self, worker):
            self._worker = worker

        def on_get(self, req, res):
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            res.body = json.dumps([{
                'id': 0,
                'name': "Benchmark0"
            }])

    class BenchmarkStartRoute:

        url = '/benchmark/{bid}/start'

        def __init__(self, worker):
            self._worker = worker

        def on_get(self, req, res, bid):
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            res.body = json.dumps({})

    class BenchmarkStopRoute:

        url = '/benchmark/{bid}/stop'

        def __init__(self, worker):
            self._worker = worker

        def on_get(self, req, res, bid):
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            res.body = json.dumps({})

    class BenchmarkProgressRoute:

        url = '/benchmark/progress'

        def __init__(self, worker):
            self._worker = worker

        def on_get(self, req, res):
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            res.body = json.dumps({
                'id': 0,
                'progress': random.randint(0, 100)
            })

    def __init__(self, port):
        # Create Falcon API
        api = falcon.API()

        # Load benchmarks
        self._benchmarks = helper.load_benchmarks()

        # Create benchmark executor
        self._executor = None

        # Add REST API routes
        for route in self._get_routes():
            api.add_route(route.url, route(self))

        # Create WSWGI server using gevent
        self._server = gevent.pywsgi.WSGIServer(('', port), api)

        # Create event (flag) to pend on
        self._stop = gevent.event.Event()

    def start(self):
        print 'Starting dellve worker ... OK'

        # Create executor
        self._executor = executor.Executor(self._benchmarks)

        # Start executor
        self._executor.start()

        # Start server
        self._server.start()

        # Wait forever...
        self._stop.wait()

        # Stop benchmark
        self._executor.stop_benchmark()

        # Join executor process
        self._executor.join()

        # Stop server
        self._server.stop()


    def stop(self, *args):
        print 'Stopping dellve worker ... '
        self._stop.set()
        print '\bOK'

    @property
    def pidfile(self):
        return '.dellve/dellve.pid'

    @property
    def workdir(self):
        return '.dellve'

    @classmethod
    def _get_routes(cls):
        return map(lambda name: getattr(cls, name),
            [name for name in dir(cls) if name.endswith('Route')])
