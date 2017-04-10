from abc import ABCMeta, abstractmethod
import time

from dellve import Benchmark, BenchmarkInterrupt

class BenchmarkFactory(Benchmark):
    __metclass__ = ABCMeta

    @abstractmethod
    def get_controller(self): pass

    @abstractmethod
    def get_problem_set(self): pass

    @abstractmethod
    def get_problem_header(self): pass

    def routine(self):
        control_constructor = self.get_controller()
        problem_set = self.get_problem_set()
        problem_set_size = len(problem_set)

        results = []

        for problem_number, problem in enumerate(problem_set):
            self.controller = control_constructor(*problem)

            try:
                self.controller.start(1, 50)

                while (not self.complete()):
                    self.update_progress(problem_number, problem_set_size)
                    time.sleep(0.25)

                self.update_progress(problem_number, problem_set_size)
                results.append(self.controller.get_avg_time_micro())

            except BenchmarkInterrupt:
                print '\nStopping current benchmark'
                self.controller.stop()
                break

        self.generate_report(problem_set, results)

    def complete(self):
        return self.controller.get_progress() == 1.0

    def update_progress(self, problem_number, problem_set_size):
        p = self.controller.get_progress()
        if (p > 0):
            self.progress = (problem_number * 100. / problem_set_size) \
                          + (p * 100 / problem_set_size)

    def generate_report(self, problem_set, results):
        row_format = '{:>2}' + '{:>9}' * len(problem_set[0]) + '{:>10}'

        header = ['#'] + self.get_problem_header() + ['time (us)']
        print row_format.format(*header)

        for problem_number, result in enumerate(results):
            row = [problem_number] + problem_set[problem_number] + [result]
            print row_format.format(*row)
