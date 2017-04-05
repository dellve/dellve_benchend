import dellve_cudnn_benchmark as dcb
import time

bc = dcb.activation_forward(7000,1500,3,14)

numRuns = 500

bc.start(1,numRuns)

c = 0

while c != numRuns:
    c = bc.get_curr_run()

print numRuns
print bc.get_curr_time_micro()
print bc.get_avg_time_micro()
