import config
import importlib
import sys
import pkg_resources
import benchmark
# from pkg_resources import iter_entry_points
# for entry_point in iter_entry_points(group='cms.plugin', name=None):
# print(entry_point)

def load_benchmarks():
    benchmarks = []

    bencmark_entry_points = \
        pkg_resources.iter_entry_points(group='dellve.benchmarks', name=None)

    for bencmark_entry_point in bencmark_entry_points:
        benchmark_class = bencmark_entry_point.load()

        if issubclass(benchmark_class, benchmark.Benchmark):
            benchmarks.append(benchmark_class)
        else:
            pass

            # TODO: add warning message saying that this 'benchmark' entry
            #       point is illformed and will be ignored!

    return benchmarks
