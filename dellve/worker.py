import config
import executor
import gevent
import gevent.event
import gevent.queue
import loader
import sys
import time
import zmq.green as zmq
import re

class DELLveWorkerThread(gevent.Greenlet):
    def __init__(self):
        gevent.Greenlet.__init__(self, self.run)
        self.__stop = gevent.event.Event()

    def pre_run(self):
        pass

    def post_run(self):
        pass

    def when_run(self):
        raise NotImplementedError()

    def run(self):
        self.pre_run()
        while not self.__stop.is_set():
            self.when_run()
        self.post_run()

    def stop(self):
        self.__stop.set()

class DELLveWorkerMainThread(DELLveWorkerThread):

    def __init__(self, server_id, host, port, socket):
        DELLveWorkerThread.__init__(self)
        self._host = str(host)
        self._port = int(port)
        self._server_id = server_id
        self._socket = socket
        self._command_queue = gevent.queue.Queue() # command-queue
        self._command_re = re.compile('(?P<server_id>\d+)(?:\s+)'      + \
                                      '(?P<command_name>\w+)(?:\s+)'   + \
                                      '(?P<command_data>.*$)')

    def pre_run(self):
        self._socket.connect('tcp://%s:%d' % (self._host, self._port))
        self._socket.setsockopt(zmq.SUBSCRIBE, str(self._server_id))

    def post_run(self):
        self._socket.disconnect()

    def when_run(self):
        message = self._socket.recv()
        match   = self._command_re.match(message)
        if match is None:
            raise NotImplementedError()
            # TODO: handle this gracefully

        command = match.groupdict()

        executor.execute(command['command_name'], command['command_data'])

class DELLveWorker(object):

    def __init__(self):
        self.thread = None

    def start(self):
        print 'Starting dellve worker ... OK'

        # Create ZeroMQ context
        context = zmq.Context()

        # Create main thread
        self.thread = DELLveWorkerMainThread(config.get('server-id'),
                                             config.get('zmq-sub-host'),
                                             config.get('zmq-sub-port'),
                                             context.socket(zmq.SUB))

        # Start main thread
        self.thread.start()

        # Wait for main thread to finish
        self.thread.join()

    def stop(self, *args):
        print 'Starting dellve worker ... '
        self.thread.stop()
        print '\bOK'

    @property
    def pidfile(self):
        return '.dellve/dellve.pid'

    @property
    def workdir(self):
        return '.dellve'
