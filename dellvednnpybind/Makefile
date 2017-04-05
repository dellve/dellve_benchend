NVCC=nvcc
ARCH?=sm_30

BIN_DIR?=bin
SRC_DIR?=src
INC_DIR=include

CUDA_PATH?=/usr/local/cuda
CUDA_INCLUDE=$(CUDA_PATH)/include
CUDA_LIB64=$(CUDA_PATH)/lib64

CUDNN_PATH?=/usr/local/cudnn
CUDNN_INCLUDE=$(CUDNN_PATH)/include
CUDNN_LIB64=$(CUDNN_PATH)/lib64

CC=$(CUDA_PATH)/bin/$/$(NVCC)
IFLAGS=-I $(INC_DIR) -I $(CUDA_INCLUDE) -I $(CUDNN_INCLUDE)
LFLAGS=-L $(CUDA_LIB64) -L $(CUDNN_LIB64) -lcudnn -lcurand -lpthread
NVCC_FLAGS=$(IFLAGS) $(LFLAGS) -arch=$(ARCH) -std=c++11

MKDIR=mkdir -p
RM_RF=rm -rf

##############################
# Add programs here
##############################
UTIL=version
CONV=forward_conv backward_data_conv backward_filter_conv
POOL=forward_pool_max forward_pool_avgpad forward_pool_avgnopad backward_pool_max backward_pool_avgpad backward_pool_avgnopad
SOFTMAX=forward_softmax_fast forward_softmax_accurate forward_softmax_log backward_softmax_fast backward_softmax_accurate backward_softmax_log
ACTIVATION=forward_activation backward_activation
##############################

UTIL_BENCH=$(addprefix $(BIN_DIR)/, $(UTIL:=.bench))
CONV_BENCH=$(addprefix $(BIN_DIR)/, $(CONV:=.bench))
POOL_BENCH=$(addprefix $(BIN_DIR)/, $(POOL:=.bench))
SOFTMAX_BENCH=$(addprefix $(BIN_DIR)/, $(SOFTMAX:=.bench))
ACTIVATION_BENCH=$(addprefix $(BIN_DIR)/, $(ACTIVATION:=.bench))
HEADERS=$(wildcard $(INC_DIR)/*.hpp)

#TODO: Add dependencies

all: bin $(UTIL_BENCH) $(CONV_BENCH) $(POOL_BENCH) $(SOFTMAX_BENCH) $(ACTIVATION_BENCH)

# TEMP ALL for fast make
# all: bin $(POOL_BENCH)

bin:
	$(MKDIR) $(BIN_DIR)

$(BIN_DIR)/%.bench: $(SRC_DIR)/%.cu $(HEADERS)
	$(CC) $< -o $@ $(NVCC_FLAGS)

clean:
	$(RM_RF) $(BIN_DIR)

rebuild: clean all


