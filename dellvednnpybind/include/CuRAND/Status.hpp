#ifndef CURAND_STATUS_HPP
#define CURAND_STATUS_HPP

#include <curand.h>
#include <iostream>

namespace CuRAND {
    typedef curandStatus_t Status;

    class Exception {
        Status mStatus;

    public:
    	Exception(Status status) : mStatus(status) {}

        const char* what() const noexcept {
            switch (mStatus) {
                case CURAND_STATUS_SUCCESS:
                    return "CuRAND Exception: CURAND_STATUS_SUCCESS";
                case CURAND_STATUS_VERSION_MISMATCH:
                    return "CuRAND Exception: CURAND_STATUS_VERSION_MISMATCH";
                case CURAND_STATUS_NOT_INITIALIZED:
                    return "CuRAND Exception: CURAND_STATUS_NOT_INITIALIZED";
                case CURAND_STATUS_ALLOCATION_FAILED:
                    return "CuRAND Exception: CURAND_STATUS_ALLOCATION_FAILED";
                case CURAND_STATUS_TYPE_ERROR:
                    return "CuRAND Exception: CURAND_STATUS_TYPE_ERROR";
                case CURAND_STATUS_OUT_OF_RANGE:
                    return "CuRAND Exception: CURAND_STATUS_OUT_OF_RANGE";
                case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                    return "CuRAND Exception: CURAND_STATUS_LENGTH_NOT_MULTIPLE";
                case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                    return "CuRAND Exception: CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
                case CURAND_STATUS_LAUNCH_FAILURE:
                    return "CuRAND Exception: CURAND_STATUS_LAUNCH_FAILURE";
                case CURAND_STATUS_PREEXISTING_FAILURE:
                    return "CuRAND Exception: CURAND_STATUS_PREEXISTING_FAILURE";
                case CURAND_STATUS_INITIALIZATION_FAILED:
                    return "CuRAND Exception: CURAND_STATUS_INITIALIZATION_FAILED";
                case CURAND_STATUS_ARCH_MISMATCH:
                    return "CuRAND Exception: CURAND_STATUS_ARCH_MISMATCH";
                case CURAND_STATUS_INTERNAL_ERROR:
                    return "CuRAND Exception: CURAND_STATUS_INTERNAL_ERROR";
                default:
                    return "Unknown exception.";
            };
        }

        Status getStatus() const {
            return mStatus;
        }
    };

    void checkStatus(Status status) {
        if (status != CURAND_STATUS_SUCCESS) {
            Exception e = Exception(status);
            std::cout << e.what() << std::endl;
            throw e;
        }
    }
}

#endif // CURAND_STATUS_HPP
