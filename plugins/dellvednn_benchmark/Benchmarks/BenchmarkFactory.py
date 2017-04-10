from abc import ABCMeta, abstractmethod
import time

from dellve import Benchmark, BenchmarkInterrupt

class BenchmarkFactory(Benchmark):
    __metclass__ = ABCMeta

    @abstractmethod
    def get_controller(self): pass

    @abstractmethod
    def get_problem_set(self): pass
    
    def routine(self):
        control_constructor = self.get_controller()
        problem_set = self.get_problem_set()
        problem_set_size = len(problem_set)

        for problem_number, problem in enumerate(problem_set):
            self.controller = control_constructor(*problem)

            try:
                self.controller.start(1, 50)

                while (not self.complete()):
                    self.update_progress(problem_number, problem_set_size)
                    time.sleep(0.25)


            except BenchmarkInterrupt:
                print '\nStopping current benchmark'
                self.controller.stop()
                break


    def complete(self):
        return self.controller.get_progress() == 1.0

    def update_progress(self, problem_number, problem_set_size):
        p = self.controller.get_progress()
        if (p > 0):
            self.progress = (problem_number * 100. / problem_set_size) \
                          + (p * 100 / problem_set_size)
