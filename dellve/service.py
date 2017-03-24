import config
import daemonocle as daemon
import os
import worker

class DELLveService(object):

    def __init__(self, debug=False):

        dameon_worker = worker.DELLveWorker(config.get('http-port'))

        if not os.path.exists(dameon_worker.workdir):
            os.makedirs(dameon_worker.workdir)

        dameon_config = {
            'worker':               dameon_worker.start,
            'shutdown_callback':    dameon_worker.stop,
            'pidfile':              dameon_worker.pidfile,
            'workdir':              dameon_worker.workdir,
            'detach':               debug == False,
        }

        self._daemon = daemon.Daemon(**dameon_config)

    def start(self):
        # daemonocle.Daemon.do_action
        self._daemon.do_action('start')

    def stop(self):
        self._daemon.do_action('stop')

    def status(self):
        self._daemon.do_action('status')
