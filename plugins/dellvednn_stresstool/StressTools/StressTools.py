
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
        win = 3
        pad = 1
        stride = 1
        n,c,h,w = problem_size.calculate_nchw_pooling(1,self.memutil,win,pad,stride)

        controller = dcb.pooling_forward(w, h, c, n, win, win, pad, pad, stride, stride, "max")
        return controller

class BackwardPoolingStressTool(StressToolFactory): 
    name = 'BackwardPoolingStressTool'

    def get_controller(self):
        win = 3
        pad = 1
        stride = 1
        n,c,h,w = problem_size.calculate_nchw_pooling(1,self.memutil,win,pad,stride)

        controller = dcb.pooling_backward(w, h, c, n, win, win, pad, pad, stride, stride, "max")
        return controller

class ForwardConvolutionStressTool(StressToolFactory):
    name = 'ForwardConvolutionStressTool'

    def get_controller(self):
        win = 3
        pad = 1
        stride = 1
        k = 3
        n,c,h,w = problem_size.calculate_nchw_convolution(1,self.memutil,k,win,pad,stride)
        
        controller = dcb.convolution_forward(w, h, c, n, k, win, win, pad, pad, stride, stride)
        return controller

class BackwardConvolutionDataStressTool(StressToolFactory):
    name = 'BackwardConvolutionDataStressTool'

    def get_controller(self):
        win = 3
        pad = 1
        stride = 1
        k = 3
        n,c,h,w = problem_size.calculate_nchw_convolution(1,self.memutil,k,win,pad,stride)
        
        controller = dcb.convolution_backward_data(w, h, c, n, k, win, win, pad, pad, stride, stride)
        return controller

