import abc
import api
import config
import gevent
import gevent.event
import gevent.pywsgi


class WorkerAPI(object):

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

    def start(self):
        """Starts DELLve worker"""
        print 'Starting dellve worker ... OK'

        # Start server
        self._server.start()

        # Wait forever...
        self._exit.wait()

        # Stop server
        self._server.stop()

    def stop(self, *args):
        """Stops DELLve worker.

        Args:
            *args: Variable arguments list (for internal use only)
        """
        print 'Stopping dellve worker ... OK'
        self._exit.set()
        gevent.sleep()

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
