NVCC=nvcc
ARCH=sm_30

CUDA_PATH?=/usr/local/cuda
CUDA_INCLUDE=$(CUDA_PATH)/include
CUDA_LIB64=$(CUDA_PATH)/lib64

CUDNN_PATH?=/usr/local/cudnn
CUDNN_INCLUDE=$(CUDNN_PATH)/include
CUDNN_LIB64=$(CUDNN_PATH)/lib64

BIN_DIR?=bin
SRC_DIR?=src
INC_DIR=include

MKDIR=mkdir -p

all: version forward backward_data

version:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) $(SRC_DIR)/version.cu -o $(BIN_DIR)/version -I $(CUDA_PATH)/include -L $(CUDA_LIB64) -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -lcudnn -arch=$(ARCH) -std=c++11

forward:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) $(SRC_DIR)/forward_conv.cu -o $(BIN_DIR)/forward_conv -I $(INC_DIR)/ -I $(CUDA_PATH)/include -L $(CUDA_LIB64) -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -lcudnn -lcurand -arch=$(ARCH) -std=c++11


backward_data:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) $(SRC_DIR)/backward_data_conv.cu -o $(BIN_DIR)/backward_data_conv -I $(INC_DIR)/ -I $(CUDA_PATH)/include -L $(CUDA_LIB64) -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -lcudnn -lcurand -arch=$(ARCH) -std=c++11

clean:
	rm -rf $(BIN_DIR)

rebuild: clean all


