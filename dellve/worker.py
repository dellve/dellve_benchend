import config
import executor
import gevent
import gevent.event
import gevent.pywsgi
import re
import falcon


class DELLveWorker(object):
    class BenchmarkStartRoute:

        url = '/benchmark/{bid}/start'

        def on_get(self, req, res, bid):
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            res.body = '{"message": "Yo, let\'s start this benchmark # %d"}' % int(bid)
            # TODO: actually start this benchmark

    class BenchmarkStopRoute:

        url = '/benchmark/{bid}/stop'

        def on_get(self, req, res, bid):
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            res.body = '{"message": "Yo, let\'s stop this benchmark # %d"}' % int(bid)
            # TODO: actually start this benchmark


    def __init__(self, port):
        # Create Falcon API
        api = falcon.API()

        # Add REST API routes
        for route in self._get_routes():
            api.add_route(route.url, route())

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
