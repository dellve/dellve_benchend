
class DELLveTooDeepException(Exception): pass

class DELLveWorker(object):

    def start(self):
        raise DELLveTooDeepException()

    def stop(self):
        raise DELLveTooDeepException()

    @property
    def pidfile(self):
        raise DELLveTooDeepException()

    @property
    def workdir(self):
        raise DELLveTooDeepException()
