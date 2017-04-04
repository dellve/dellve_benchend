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
        """Summary

        Returns:
            TYPE: Description
        """

    @abc.abstractmethod
    def start(self):
        """Summary

        Returns:
            TYPE: Description
        """

    @abc.abstractmethod
    def stop(self):
        """Summary

        Returns:
            TYPE: Description
        """

    @abc.abstractproperty
    def workdir(self):
        """Summary

        Returns:
            TYPE: Description
        """


class Worker(WorkerAPI):
    """
    @brief      DELLve background worker.
    """

    def __init__(self, port=config.get('http-port')):
        """
        @brief      Constructs a new DELLve worker.

        @param      port  HTTP API port
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
        """
        @brief      The name of PID-file for the worker.

        @return     PID-file name
        """
        return '.dellve/dellve.pid'


    @property
    def workdir(self):
        """
        @brief      The name of working directory for the worker.

        @return     Working directory name
        """
        return '.dellve'
