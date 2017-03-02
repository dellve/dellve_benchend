import daemonocle as daemon
import zmq
import worker

class DELLveService(object):

    def __init__(self):

        dameon_worker_config = {

        }

        dameon_worker = worker.DELLveWorker(**dameon_worker_config)

        dameon_config = {
            'worker':               dameon_worker.start,
            'shutdown_callback':    dameon_worker.stop,
            'pidfile':              dameon_worker.pidfile,
            'workdir':              dameon_worker.workdir,
        }

        self._daemon = daemon.Daemon(**dameon_config)

    def start(self):
        # daemonocle.Daemon.do_action
        self._daemon.do_action('start')

    def stop(self):
        self._daemon.do_action('stop')

    def status(self):
        self._daemon.do_action('status')
