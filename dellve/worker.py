import config
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

    def step(self):
        raise NotImplementedError()

    def run(self):
        self.pre_run()
        while not self.__stop.is_set():
            self.step()
        self.post_run()

    def stop(self):
        self.__stop.set()

class DELLveWorker(object):

    class CommandServerThread(DELLveWorkerThread):
        def __init__(self, server_id, host, port, socket):
            DELLveWorkerThread.__init__(self)
            self._host = str(host)
            self._port = int(port)
            self._server_id = server_id
            self._socket = socket
            self._command_queue = gevent.queue.Queue() # command-queue
            self._command_re = re.compile('(?P<server_id>\d+)(?:\s+)'      + \
                                          '(?P<command_type>\w+)(?:\s+)'   + \
                                          '(?P<command_data>.*$)')

        def process_message(self, message):
            match = self._command_re.match(message)
            if match is None:
                raise NotImplementedError()
                # TODO: handle this gracefully
            self._command_queue.put(match.groupdict())

        def pre_run(self):
            self._socket.connect('tcp://%s:%d' % (self._host, self._port))
            self._socket.setsockopt(zmq.SUBSCRIBE, str(self._server_id))

        def step(self):
            message = self._socket.recv()
            self.process_message(message)

        def post_run(self):
            self._socket.disconnect()

        def get_command(self, block=True, timeout=None):
            return self._command_queue.get(block, timeout)

    class CommandProcessorThread(DELLveWorkerThread):

        dispatch = {
            'startBenchmark': 'command_start_benchmark',
            'stopBenchmark': 'command_stop_benchmark',
            'startMetricStream': 'command_start_metric_stream',
            'stopMetricStream': 'command_stop_metric_stream'
        }

        def __init__(self, command_server=None):
            DELLveWorkerThread.__init__(self)
            self._command_server = command_server

        def command_start_benchmark(self, server_id, command_data):
            print 'Starting benchmark: ', str(server_id), str(command_data)

        def command_stop_benchmark(self, server_id, command_data):
            print 'Stopping benchmark: ', str(server_id), str(command_data)

        def command_start_metric_stream(self, server_id, command_data):
            print 'Starting metric stream: ', str(server_id), str(command_data)

        def command_stop_metric_stream(self, server_id, command_data):
            print 'Stopping metric stream: ', str(server_id), str(command_data)

        def step(self):
            command = self._command_server.get_command()

            server_id = command['server_id']
            command_type = command['command_type']
            command_data = command['command_data']

            getattr(self, self.dispatch[command_type])(server_id, command_data)

    def __init__(self, server_id):
        self.server_id = server_id
        self.threads   = []

    def start(self):
        print 'Starting dellve worker ... OK'

        # Create ZeroMQ context
        context = zmq.Context()

        # Create & register command SERVER sub-thread
        command_server = DELLveWorker.CommandServerThread(
            config.get('server-id'),
            config.get('zmq-sub-host'),
            config.get('zmq-sub-port'),
            context.socket(zmq.SUB))
        self.threads.append(command_server)

        # Create & register command PROCESSOR sub-thread
        command_processor = DELLveWorker.CommandProcessorThread(command_server)
        self.threads.append(command_processor)

        # Start sub-threads created above
        map(DELLveWorkerThread.start, self.threads)

        # Wait for sub-threads to finish
        gevent.joinall(self.threads)

    def stop(self, *args):
        print 'Starting dellve worker ... '
        map(DELLveWorkerThread.stop, self.threads)
        print '\bOK'

    @property
    def pidfile(self):
        return '.dellve/dellve.pid'

    @property
    def workdir(self):
        return '.dellve'
