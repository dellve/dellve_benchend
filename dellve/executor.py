
import stringcase

def execute(command_name, command_data):
    globals()[stringcase.snakecase(command_name)](command_data)

def start_benchmark(command_data):
    print 'Starting benchmark: ', str(command_data)

def stop_benchmark(command_data):
    print 'Stopping benchmark: ', str(command_data)

def start_metric_stream(command_data):
    print 'Starting metric stream: ', str(command_data)

def stop_metric_stream(command_data):
    print 'Stopping metric stream: ', str(command_data)
