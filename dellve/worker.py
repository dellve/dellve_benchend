import config
import gevent
import loader
import sys
import time
import zmq.green as zmq

class DELLveTooDeepException(Exception): pass

class DELLveWorker(object):

    def __init__(self):
        self.plugins = loader.load_plugins()
        print self.plugins

    def start(self):
        # Socket to talk to server
        context = zmq.Context()
        socket = context.socket(zmq.SUB)

        print "Connecting to backend via ZeroMQ..."

        print "tcp://%s:%d" % \
            (config.get('zmq-sub-host'), config.get('zmq-sub-port'))

        socket.connect ("tcp://%s:%d" % \
            (config.get('zmq-sub-host'), config.get('zmq-sub-port')))

        # topicfilter = "1"
        # socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

        fifo = []

        def serve(socket):
            while True:
                fifo.append(socket.recv())
                # print "Received request from backend: ", message

        server = gevent.spawn(serve, socket)

        while True:
            for item in fifo:
                print item


    def stop(self):
        self.terminate = True

    @property
    def pidfile(self):
        return '.dellve/dellve.pid'

    @property
    def workdir(self):
        return '.dellve'
