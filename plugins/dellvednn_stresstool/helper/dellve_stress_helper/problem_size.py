import gpu_info
import math

def calculate_nchw(device_id, mem_util):
    # TODO: Limit Mem_Util to .5 < mem_util < .95
    overhead = 71*1000*1000
    mem = int(gpu_info.get_total_mem(device_id) * mem_util)
    mem -= overhead
    mem /= 8

    c = 1
    n = 75

    mem = mem/c
    mem = mem/n

    h = int(math.sqrt(mem))
    w = mem/h

    return n,c,h,w
