import dellve.benchmark
import multiprocessing as mp
import pytest
import random
import time


def test_benchmark():
    """Tests Benchmark class non-abstract methods"""
    benchmark_progress = random.randint(0, 100)

    # Create mock benchmark implementation that will
    # set it's progress to predefined value once it runs
    class MockBenchmark(dellve.benchmark.Benchmark):
        config = dellve.benchmark.BenchmarkConfig([])
        def routine(self):
            self.progress = benchmark_progress
            while True:
                time.sleep(1)

    # Create benchmark
    benchmark = MockBenchmark()

    # Note: benchmark isn't running when it's created,
    #       so the 'is_running' method should return False
    assert benchmark.is_running() == False

    # Start benchmark
    benchmark.start()

    time.sleep(1) # give the sub-process some time to start

    # Note: we started the benchmark, and 'is_running' method
    #       should return True
    assert benchmark.is_running() == True

    # Note: our mock benchmark set's progress value to pre-defined
    #       'benchmark_progress' value, so we expect them to be equal
    assert benchmark.progress == benchmark_progress

    # Stop benchmark
    benchmark.stop()

    time.sleep(1) # give the sub-process some time to stop

    # NOte: we stopped the benchmark, so the 'is_running' method
    #       should return False
    assert benchmark.is_running() == False

def test_benchmark_progress():
    """Tests Benchmark progress update logic"""

    # Create mock benchmark implementation
    #
    # Note: this benchmark doesn't do very much;
    #       however, we still need to provide default
    #       implementation of its abstract methods to
    #       be able to create an instance of it.
    class MockBenchmark(dellve.benchmark.Benchmark):
        config = dellve.benchmark.BenchmarkConfig([])
        def routine(self):
            pass

    # Create benchmark
    benchmark = MockBenchmark()

    # Benchmark should start with progress at 0
    assert benchmark.progress == 0

    # Progress should always be >= 0
    with pytest.raises(ValueError):
        benchmark.progress = -1.00

    # Progress should always be <= 100
    with pytest.raises(ValueError):
        benchmark.progress = 101.00

    # Progress should retain value
    benchmark_progress = random.randint(0, 100)
    benchmark.progress = benchmark_progress
    assert benchmark.progress == benchmark_progress

