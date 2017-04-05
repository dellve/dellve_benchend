
import dellve
import dellve_cudnn_benchmark as dcb
import time
from dellve_stress_helper import problem_size

class ForwardActivationStressTool(dellve.Benchmark): 
    name = 'ForwardActivationStressTool'

    def routine(self):
        # TODO: implement
        print self.memutil
        n,c,h,w = problem_size.calculate_nchw(1,0.5)
        print n
        print c
        print h
        print w
        controller = dcb.activation_forward(w,h,c,n)
        controller.start(1,100)

        p = controller.get_progress()

        while(p != 1.0):
            p = controller.get_progress()
            if p > 0:
                self.progress = p * 100
            time.sleep(0.5)
