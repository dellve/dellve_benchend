from dellve import Benchmark, BenchmarkInterrupt
import dellve_cudnn_benchmark as dcb
import time
from abc import ABCMeta, abstractmethod

class BenchmarkFactory(Benchmark):
    __metclass__ = ABCMeta

    @abstractmethod
    def get_problem_set(self): pass

    @abstractmethod
    def get_controller(self): pass

    def routine(self):
        control_constructor = self.get_controller()
        problem_set = self.get_problem_set()
        problem_set_size = len(problem_set)

        for problem_number, problem in enumerate(problem_set):
            controller = control_constructor(*problem)
            try:
                controller.start(1, 50)

                p = 0.0
                while (p != 1.0):
                    p = controller.get_progress()
                    if (p > 0):
                        self.progress = (problem_number * 100. / problem_set_size) + (p * 100 / problem_set_size)

                    time.sleep(0.5)

                print 'Problem {:2}/{} took {:5} microseconds'.format(problem_number + 1, problem_set_size, controller.get_avg_time_micro())

            except BenchmarkInterrupt:
                print 'Stopping current benchmark'
                controller.stop()
                break
