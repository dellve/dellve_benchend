
from pynvml import *

def get_total_mem(device_id):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)
    mem = nvmlDeviceGetMemoryInfo(handle).total
    nvmlShutdown()

    return mem
