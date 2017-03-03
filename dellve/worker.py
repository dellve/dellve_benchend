import config
import sys
import time

class DELLveTooDeepException(Exception): pass

class DELLveWorker(object):

    def __init__(self):
        self.terminate = False

    def start(self):
        while True:
            time.sleep(1)

    def stop(self):
        self.terminate = True

    @property
    def pidfile(self):
        return '.dellve/dellve.pid'

    @property
    def workdir(self):
        return '.dellve'
