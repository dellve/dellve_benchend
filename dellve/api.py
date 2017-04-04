"""Summary
"""
import config
import falcon
import json
import gevent
import gevent.lock

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
        def on_get(self, req, res, bid):
            return self.api._get_benchmark_start(req, res, int(bid))

    class BenchmarkStopRoute(HttpRoute):
        url = '/benchmark/{bid}/stop'
        def on_get(self, req, res, bid):
            return self.api._get_benchmark_stop(req, res, int(bid))

    class BenchmarkProgressRoute(HttpRoute):
        url = '/benchmark/progress'
        def on_get(self, req, res):
            return self.api._get_benchmark_progress(req, res)

    def __init__(self, benchmarks=config.get('benchmarks')):
        """Constructs a new HttpAPI instance.

        Args:
            benchmarks (list, optional): List of benchmark classes.
        """

        # Construct parent class
        falcon.API.__init__(self)

        # Create synchronization semaphore
        self._lock = gevent.lock.Semaphore()

        # Save benchmark classes
        self._benchmarks = benchmarks

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
            'id': benchmark_id,
            'name': benchmark.name
        } for benchmark_id, benchmark in enumerate(self._benchmarks)])

    def _get_benchmark_start(self, req, res, bid):
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
                self._current_benchmark = self._benchmarks[bid]()
                self._current_benchmark.start()
                self._current_benchmark_id = bid

    def _get_benchmark_stop(self, req, res, bid):
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
                benchmark_id = self._current_benchmark_id
                benchmark_progress = self._current_benchmark.progress
            else:
                benchmark_id = None
                benchmark_progress = None

            res.body = json.dumps({
                'id': benchmark_id,
                'progress': benchmark_progress
            })
