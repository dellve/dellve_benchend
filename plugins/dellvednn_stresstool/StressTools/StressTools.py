
import dellve
import dellve_cudnn_benchmark as dcb
import time
from abc import abstractmethod
from helper import problem_size
from StressFactory import StressToolFactory

class ForwardActivationStressTool(StressToolFactory): 
    name = 'ForwardActivationStressTool'

    def get_controller(self):
        n,c,h,w = problem_size.calculate_nchw_forward(1,self.memutil)
        return dcb.activation_forward(w,h,c,n)

class BackwardActivationStressTool(StressToolFactory): 
    name = 'BackwardActivationStressTool'

    def get_controller(self):
        n,c,h,w = problem_size.calculate_nchw_activation_backward(1,self.memutil)
        return dcb.activation_backward(w,h,c,n)

class ForwardSoftmaxStressTool(StressToolFactory): 
    name = 'ForwardSoftmaxStressTool'

    def get_controller(self):
        n,c,h,w = problem_size.calculate_nchw_forward(1,self.memutil)
        return dcb.softmax_forward(w,h,c,n,"fast")

class BackwardSoftmaxStressTool(StressToolFactory): 
    name = 'BackwardSoftmaxStressTool'

    def get_controller(self):
        n,c,h,w = problem_size.calculate_nchw_softmax_backward(1,self.memutil)
        return dcb.softmax_backward(w,h,c,n,"fast")

    
class ForwardPoolingStressTool(StressToolFactory): 
    name = 'ForwardPoolingStressTool'

    def get_controller(self):
        # TODO: implement
        print self.memutil
        n,c,h,w = problem_size.calculate_nchw_pooling_forward(1,self.memutil)
        print n
        print c
        print h
        print w
        controller = dcb.pooling_forward(w,h,c,n,10,10,0,0,2,2,"max")

class BackwardPoolingStressTool(StressToolFactory): 
    name = 'BackwardPoolingStressTool'

    def get_controller(self):
        # TODO: implement
        print self.memutil
        n,c,h,w = problem_size.calculate_nchw_pooling_forward(1,self.memutil)
        print n
        print c
        print h
        print w
        controller = dcb.pooling_forward(w,h,c,n,10,10,0,0,2,2,"max")
