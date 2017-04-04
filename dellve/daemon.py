import config
import daemonocle
import os
import worker


class Daemon(daemonocle.Daemon):
    """DELLve background worker daemon.
    """

    def __init__(self, debug=False, daemon_worker=worker.Worker()):
        # Create working directory if one doesn't exit
        if not os.path.exists(daemon_worker.workdir):
            os.makedirs(daemon_worker.workdir)

        # Note:     Background deamon is created using daemonocle;
        #           Please refer https://pypi.python.org/pypi/daemonocle

        dameon_config = {
            'worker':               daemon_worker.start,
            'shutdown_callback':    daemon_worker.stop,
            'pidfile':              daemon_worker.pidfile,
            'workdir':              daemon_worker.workdir,
            'detach':               debug == False,
        }

        # Construct background daemon instance
        daemonocle.Daemon.__init__(self, **dameon_config)
