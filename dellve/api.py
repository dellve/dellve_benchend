import config
import copy
import falcon
import json
import gevent
import gevent.lock
import logging

class HttpRoute(object):
    """DELLve REST API route base-class.

    Attributes:
        api (HttpAPI): Description
    """
    def __init__(self, api):
        """Summary

        Args:
            api (TYPE): Description
        """
        self.api = api

class HttpAPI(falcon.API):
    """DELLve REST HTTP API class.
    """

    class BenchmarkRoute(HttpRoute):
        url = '/benchmark'
        def on_get(self, req, res):
            return self.api._get_benchmark_info(req, res)

    class BenchmarkStartRoute(HttpRoute):
        url = '/benchmark/{bid}/start'
        def on_post(self, req, res, bid):
            return self.api._post_benchmark_start(req, res, int(bid))

    class BenchmarkStopRoute(HttpRoute):
        url = '/benchmark/{bid}/stop'
        def on_post(self, req, res, bid):
            return self.api._post_benchmark_stop(req, res, int(bid))

    class BenchmarkProgressRoute(HttpRoute):
        url = '/benchmark/progress'
        def on_get(self, req, res):
            return self.api._get_benchmark_progress(req, res)

    class HeartbeatRoute(HttpRoute):
        url = '/heartbeat'
        def on_get(self, req, res):
            res.status = falcon.HTTP_200

    def __init__(self, benchmarks=config.get('benchmarks')):
        """Constructs a new HttpAPI instance.

        Args:
            benchmarks (list, optional): List of benchmark classes.
        """

        # Construct parent class
        falcon.API.__init__(self, middleware=[RequestLoggingMiddleware()])

        # Create synchronization semaphore
        self._lock = gevent.lock.Semaphore()

        # Save benchmark classes
        self._benchmarks = benchmarks
        for b in benchmarks:
            b.init_config()

        # Create default state
        self._current_benchmark = None
        self._current_benchmark_id = None

        # Register REST API routes
        for item in [getattr(self.__class__, n) for n in dir(self.__class__)]:
            if isinstance(item, type) and issubclass(item, HttpRoute):
                self.add_route(item.url, item(self))

    def _get_benchmark_info(self, req, res):
        """Summary

        Args:
            req (TYPE): Description
            res (TYPE): Description

        Returns:
            TYPE: Description
        """
        res.status = falcon.HTTP_200
        res.content_type = 'application/json'
        res.body = json.dumps([{
            'config': benchmark.config,
            'id': benchmark_id,
            'name': benchmark.name,
            'schema': benchmark.schema,
        } for benchmark_id, benchmark in enumerate(self._benchmarks)])

    def _post_benchmark_start(self, req, res, bid):
        """Summary

        Args:
            req (TYPE): Description
            res (TYPE): Description
            bid (TYPE): Description

        Returns:
            TYPE: Description
        """
        with self._lock:
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            if not self._current_benchmark or \
                not self._current_benchmark.is_running():
                try: # Load config from request body
                    config = dict(json.load(req.stream))
                except ValueError:
                    config = {}
                self._current_benchmark = self._benchmarks[bid](config)
                self._current_benchmark.start()
                self._current_benchmark_id = bid

    def _post_benchmark_stop(self, req, res, bid):
        """Summary

        Args:
            req (TYPE): Description
            res (TYPE): Description
            bid (TYPE): Description

        Returns:
            TYPE: Description
        """
        with self._lock:
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'
            if self._current_benchmark and self._current_benchmark.is_running():
                self._current_benchmark.stop()

    def _get_benchmark_progress(self, req, res):
        """Summary

        Args:
            req (TYPE): Description
            res (TYPE): Description

        Returns:
            TYPE: Description
        """
        with self._lock:
            res.status = falcon.HTTP_200
            res.content_type = 'application/json'

            if self._current_benchmark is not None:
                res.body = json.dumps({
                    'id':           self._current_benchmark_id,
                    'name':         self._current_benchmark.name,
                    'progress':     self._current_benchmark.progress,
                    'output':       self._current_benchmark.output,
                    'running':      self._current_benchmark.is_running()
                })
            else:
                res.body = json.dumps({
                    'id':           None,
                    'name':         None,
                    'progress':     None,
                    'output':       None,
                    'running':      None
                })

"""HTTP API logger"""
logger = logging.getLogger('http-api-logger')

class RequestLoggingMiddleware(object):
    def process_request(self, req, resp):
        logger.info('{0} {1} {2}'
            .format(req.method, req.relative_uri, resp.status[:3]))
