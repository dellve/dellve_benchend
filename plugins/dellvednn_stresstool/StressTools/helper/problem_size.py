import gpu_info
import math

def calculate_nchw(mem):
    c = 1
    n = 75

    mem = mem/c
    mem = mem/n

    h = int(math.sqrt(mem))
    w = mem/h

    return n,c,h,w

def calculate_nchw_forward(device_id, mem_util):
    # TODO: Limit Mem_Util to .5 < mem_util < .95
    overhead = 71*1000*1000
    mem = int(gpu_info.get_total_mem(device_id) * mem_util)
    mem -= overhead
    mem /= 8

    return calculate_nchw(mem)

def calculate_nchw_activation_backward(device_id, mem_util):
    # TODO: Limit Mem_Util
    # Overhead counted twice so half it from forward
    overhead = (71*1000*1000)/2
    mem = int((gpu_info.get_total_mem(device_id)/2) * mem_util)
    mem -= overhead
    mem /= 8

    return calculate_nchw(mem)

def calculate_nchw_softmax_backward(device_id, mem_util):
    # TODO: Limit Mem_Util
    # Overhead counted roughly 2/3 (calc 2611/1763 =~ 3/2)
    overhead = ((71*1000*1000)*2)/3
    mem = int(((gpu_info.get_total_mem(device_id)*2)/3) * mem_util)
    mem -= overhead
    mem /= 8

    return calculate_nchw(mem)

# Stuff~
def add_pooling(mem):
    dep = 633*1024*1024
    dep_source = 1848705024
    dep_f = mem/float(dep_source)
    dep *= dep_f
    dep *= (1.5)

    return int(dep) 

# Stuff~
def calculate_nchw_pooling_forward(device_id, mem_util):
    overhead = (71*1000*1000)
    mem = int(gpu_info.get_total_mem(device_id) * mem_util)

    s = add_pooling(mem)
    mem += s

    #mem += s
    mem /= 8

    return calculate_nchw(mem)


     
