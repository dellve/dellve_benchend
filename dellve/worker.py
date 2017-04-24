import abc
import api
import config
import gevent
import gevent.event
import gevent.pywsgi
import logging
import sys
import traceback

# DELLve logger
logger = logging.getLogger('dellve-logger')


class WorkerAPI(object):
    """Abstract DELLve background worker interface"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def pidfile(self):
        """The name of PID-file for the worker"""

    @abc.abstractmethod
    def start(self):
        """Starts DELLve worker"""

    @abc.abstractmethod
    def stop(self):
        """Stops DELLve worker.

        Args:
            *args: Variable arguments list (for internal use only)
        """

    @abc.abstractproperty
    def workdir(self):
        """The name of working directory for the worker"""


class Worker(WorkerAPI):
    """DELLve background worker"""

    def __init__(self, port=config.get('http-port')):
        """
        @brief      Constructs a new DELLve worker.

        @param      port  HTTP API port

        Args:
            port (TYPE, optional): Description
        """

        # Create DELLve API instance
        dellve_api = api.HttpAPI()

        # Create WSWGI server using DELLve API
        self._server = gevent.pywsgi.WSGIServer(('', port), dellve_api)

        # Create event to wait for on exit
        self._exit = gevent.event.Event()

        # Create event to notify on exit
        self._exited = gevent.event.Event()

    def start(self):
        """Starts DELLve worker"""
        # Let the user worker started via STDOUT
        #
        # TODO: this can be eliminated if we configure logging system
        #       to print INFO messages to STDOUT in config.py logging setup
        #
        logger.info('Started dellve worker')

        # Note: we could use self._server.serve_forever() here, but since we
        #       need to notify user about successful server start, we rely
        #       on event self._exit / self._exited, and server methods
        #       self._server.start() / self._server.stop() instead.

        try: # Start server...
            self._server.start()
        except Exception: # Report unsuccessful server start
            logger.exception('Couldn\'t start dellve HTTP server')
            return # there's nothing we can do without server running
        else: # Report successful server start
            logger.info('Started dellve HTTP server')

        # Wait for exit...
        self._exit.wait()
        # Notify handler!
        self._exited.set()

    def stop(self, *args):
        """Stops DELLve worker.

        Args:
            *args: Variable arguments list (for internal use only)
        """

        # Note: this method is called from within gevent run loop, so we can't
        #       call any blocking functions here; instead, we create new
        #       greenlet that is going to do all the work for us; this greenlet
        #       won't exit before other greenlets, thus ensuring graceful exit

        def stop_server():
            try: # Stop server...
                self._server.stop()
            except Exception: # Report unsuccessful server stop
                logger.exception('Couldn\'t stop dellve HTTP server')
                return # there's nothing we can do with server running
            else: # Report successful server stop
                logger.info('Stopped dellve HTTP server')

            # Request exit!
            self._exit.set()
            # Wait for exit...
            self._exited.wait()
            # Let the user worker stopped via STDOUT
            #
            # TODO: this can be eliminated if we configure logging system
            #       to print INFO messages to STDOUT in config.py logging setup
            logger.info('Stopped dellve worker')

        # Wait for server to stop...
        gevent.spawn(stop_server)

    @property
    def pidfile(self):
        """The name of PID-file for the worker.

        Returns:
            string: PID-file name
        """
        return config.get('pid-file')

    @property
    def workdir(self):
        """The name of working directory for the worker.

        Returns:
            string: Working directory name
        """
        return config.get('app-dir')
