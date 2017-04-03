import config
import daemonocle as daemon
import os
import worker

class DELLveService(object):
    """Back-end DELLve deamon service"""

    def __init__(self, daemon_worker=worker.DELLveWorker(), debug=False):
        """Constructs a new DELLveService object.

        Args:
            debug (bool, optional): Flag indicating if service should be run in 'debug' mode
        """

        # Create worker instance
        if not isinstance(daemon_worker, worker.WorkerAPI):
            raise TypeError() # TODO: come up with meaningful error message
        dameon_worker = daemon_worker

        # Create working directory if one doesn't exit
        if not os.path.exists(dameon_worker.workdir):
            os.makedirs(dameon_worker.workdir)

        # Note:     Background deamon is created using daemonocle;
        #           Please refer https://pypi.python.org/pypi/daemonocle

        dameon_config = {
            'worker':               dameon_worker.start,
            'shutdown_callback':    dameon_worker.stop,
            'pidfile':              dameon_worker.pidfile,
            'workdir':              dameon_worker.workdir,
            'detach':               debug == False,
        }

        # Create background daemon
        self._daemon = daemon.Daemon(**dameon_config)

    def start(self):
        """Starts background daemon.
        """
        self._daemon.do_action('start')

    def stop(self):
        """Starts background daemon.
        """
        self._daemon.do_action('stop')

    def status(self):
        """Prints background daemon's status.
        """
        self._daemon.do_action('status')
