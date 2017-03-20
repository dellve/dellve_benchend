import config
import executor
import gevent
import gevent.event
import gevent.pywsgi
import re
import falcon
import json
import helper


class DELLveWorker(object):

    class BenchmarkListRoute:

        url = '/benchmarks/'

        def __init__(self, worker):
            self._worker = worker

        def on_get(self, req, res):
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            data = []
            bechmarks = self._worker._benchmarks
            for _id, benchmark in enumerate(bechmarks):
                data.append({
                    'id': _id,
                    'name': benchmark.name
                })
            res.body = json.dumps(data)

    class BenchmarkStartRoute:

        url = '/benchmarks/{bid}/start'

        def __init__(self, worker):
            self._worker = worker

        def on_get(self, req, res, bid):
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            benchmark = self._worker._benchmarks[int(bid)]
            benchmark.start()
            res.body = json.dumps({})

    class BenchmarkStopRoute:

        url = '/benchmarks/{bid}/stop'

        def __init__(self, worker):
            self._worker = worker

        def on_get(self, req, res, bid):
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            benchmark = self._worker._benchmarks[int(bid)]
            benchmark.stop()
            res.body = json.dumps({})

    def __init__(self, port):
        # Create Falcon API
        api = falcon.API()

        # Load benchmarks
        self._benchmarks = map(lambda b: b(), helper.load_benchmarks())

        # Add REST API routes
        for route in self._get_routes():
            api.add_route(route.url, route(self))

        # Create WSWGI server using gevent
        self._server = gevent.pywsgi.WSGIServer(('', port), api)

        # Create event (flag) to pend on
        self._stop = gevent.event.Event()

    def start(self):
        print 'Starting dellve worker ... OK'

        # Start server
        self._server.start()

        # Wait forever...
        self._stop.wait()

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
