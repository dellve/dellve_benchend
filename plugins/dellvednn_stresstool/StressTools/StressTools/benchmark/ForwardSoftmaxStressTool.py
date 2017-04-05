import dellve
import dellve_cudnn_benchmark as dcb
import time
from dellve_stress_helper import problem_size

class ForwardSoftmaxStressTool(dellve.Benchmark): 
    name = 'ForwardSoftmaxStressTool'

    def routine(self):
        # TODO: implement
        print self.memutil
        n,c,h,w = problem_size.calculate_nchw_forward(1,self.memutil)
        print n
        print c
        print h
        print w
        controller = dcb.softmax_forward(w,h,c,n,"fast")
        controller.start(1,100)

        p = controller.get_progress()

        while(p != 1.0):
            p = controller.get_progress()
            if p > 0:
                self.progress = p * 100
            time.sleep(0.5)
