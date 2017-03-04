import config
import gevent
import loader
import sys
import time
import zmq.green as zmq
import re
import stringcase

class DELLveTooDeepException(Exception): pass

class DELLveWorker(object):

    def __init__(self):
        self.plugins = loader.load_plugins()
        print self.plugins

    def command(self, server_id, command_type, command_data):
        command_name = '_'.join(['command', stringcase.snakecase(command_type)])
        method = getattr(self, command_name)(server_id, command_data)

    def command_start_benchmark(self, server_id, command_data):
        print 'Starting benchmark: ', str(server_id), str(command_data)

    def command_stop_benchmark(self, server_id, command_data):
        print 'Stopping benchmark: ', str(server_id), str(command_data)

    def start(self):
        # TODO: CLEAN THIS FUNCTION UP

        context = zmq.Context()
        socket = context.socket(zmq.SUB)

        command_re = re.compile('(?P<server_id>\d+)(?:\s+)'     + \
                                '(?P<command_type>\w+)(?:\s+)'  + \
                                '(?P<command_data>.*$)')

        print 'Connecting to backend via ZeroMQ...'

        print 'tcp://%s:%d' % \
            (config.get('zmq-sub-host'), config.get('zmq-sub-port'))

        socket.connect ('tcp://%s:%d' % \
            (config.get('zmq-sub-host'), config.get('zmq-sub-port')))

        socket.setsockopt(zmq.SUBSCRIBE, '1') # WE NEED THIS LINE

        def serve(socket):
            while True:
                gevent.sleep(1)
                match = command_re.match(socket.recv())

                # TODO: deal with exceptions...

                if not match:
                    raise DELLveTooDeepException(':/')
                else:
                    match_groupdict = match.groupdict()

                    try:
                        self.command(**match_groupdict)
                    except AttributeError e:
                        raise NotImplementedError(':/')

                # Note: match may not work out, be careful

        server = gevent.spawn(serve, socket)

        while True:
            gevent.sleep(1)


    def stop(self):
        self.terminate = True

    @property
    def pidfile(self):
        return '.dellve/dellve.pid'

    @property
    def workdir(self):
        return '.dellve'
