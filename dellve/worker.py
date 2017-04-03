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


class DELLveWorker(WorkerAPI):
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

        # Create event (flag) to pend on
        self._stop = gevent.event.Event()

    def start(self):
        """
        @brief      Starts DELLve worker.
        """
        print 'Starting dellve worker ... OK'

        # Start server
        self._server.start()

        # Wait forever...
        self._stop.wait()

        # Stop server
        self._server.stop()


    def stop(self, *args):
        """
        @brief      Stops DELLve worker.

        @param      args  Variable arguments list (for internal use)
        """
        print 'Stopping dellve worker ... '
        self._stop.set() # notify worker to stop
        print '\bOK'

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
