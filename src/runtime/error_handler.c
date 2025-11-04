#include "qallow/error_codes.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

/* Get current timestamp in milliseconds */
static long qallow_get_timestamp_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000L + tv.tv_usec / 1000L;
}

const char* qallow_error_message(int code) {
    switch (code) {
        case QALLOW_SUCCESS:
            return "Success";

        /* Core System */
        case QALLOW_ERR_UNKNOWN:
            return "Unknown error";
        case QALLOW_ERR_NOT_IMPLEMENTED:
            return "Feature not implemented";
        case QALLOW_ERR_UNSUPPORTED:
            return "Operation not supported on this platform";

        /* Memory Management */
        case QALLOW_ERR_MEMORY_ALLOC:
            return "Memory allocation failed";
        case QALLOW_ERR_MEMORY_LIMIT:
            return "Memory limit exceeded";
        case QALLOW_ERR_NULL_POINTER:
            return "Null pointer dereference";

        /* Input Validation */
        case QALLOW_ERR_INVALID_PARAMETER:
            return "Invalid parameter";
        case QALLOW_ERR_INVALID_STATE:
            return "Invalid state for operation";
        case QALLOW_ERR_BOUNDS_CHECK:
            return "Bounds check failed";
        case QALLOW_ERR_TYPE_MISMATCH:
            return "Type mismatch";

        /* File I/O */
        case QALLOW_ERR_FILE_NOT_FOUND:
            return "File not found";
        case QALLOW_ERR_FILE_OPEN_FAILED:
            return "Failed to open file";
        case QALLOW_ERR_FILE_READ_FAILED:
            return "Failed to read file";
        case QALLOW_ERR_FILE_WRITE_FAILED:
            return "Failed to write file";
        case QALLOW_ERR_PATH_INVALID:
            return "Invalid file path";

        /* CUDA/Hardware */
        case QALLOW_ERR_CUDA_INIT:
            return "CUDA initialization failed";
        case QALLOW_ERR_CUDA_DEVICE:
            return "CUDA device error";
        case QALLOW_ERR_CUDA_KERNEL:
            return "CUDA kernel execution failed";
        case QALLOW_ERR_CUDA_MEMORY:
            return "CUDA memory error";
        case QALLOW_ERR_GPU_NOT_AVAILABLE:
            return "GPU not available";

        /* Quantum */
        case QALLOW_ERR_QUANTUM_CIRCUIT:
            return "Invalid quantum circuit";
        case QALLOW_ERR_QUANTUM_STATE:
            return "Invalid quantum state";
        case QALLOW_ERR_QUBIT_LIMIT:
            return "Qubit limit exceeded";
        case QALLOW_ERR_QUANTUM_SIM:
            return "Quantum simulator error";

        /* Networking */
        case QALLOW_ERR_SOCKET_CREATE:
            return "Failed to create socket";
        case QALLOW_ERR_SOCKET_BIND:
            return "Failed to bind socket";
        case QALLOW_ERR_SOCKET_LISTEN:
            return "Failed to listen on socket";
        case QALLOW_ERR_SOCKET_CONNECT:
            return "Connection failed";
        case QALLOW_ERR_SOCKET_SEND:
            return "Send failed";
        case QALLOW_ERR_SOCKET_RECV:
            return "Receive failed";

        /* Database */
        case QALLOW_ERR_DB_OPEN:
            return "Database open failed";
        case QALLOW_ERR_DB_QUERY:
            return "Database query failed";
        case QALLOW_ERR_DB_COMMIT:
            return "Database commit failed";
        case QALLOW_ERR_DB_CONSTRAINT:
            return "Constraint violation";

        /* Timeout/Performance */
        case QALLOW_ERR_TIMEOUT:
            return "Operation timeout";
        case QALLOW_ERR_DEADLOCK:
            return "Deadlock detected";
        case QALLOW_ERR_RESOURCE_EXHAUSTED:
            return "Resource exhausted";

        /* Logic */
        case QALLOW_ERR_ASSERTION_FAILED:
            return "Assertion failed";
        case QALLOW_ERR_INVARIANT_VIOLATED:
            return "Invariant violated";
        case QALLOW_ERR_LOGIC_ERROR:
            return "Logic error";

        default:
            return "Unrecognized error code";
    }
}

qallow_error_severity_t qallow_error_severity(int code) {
    if (code >= 900) {
        return QALLOW_SEVERITY_CRITICAL;
    } else if (code >= 800) {
        return QALLOW_SEVERITY_ERROR;
    } else if (code >= 400) {
        return QALLOW_SEVERITY_WARNING;
    } else if (code >= 100) {
        return QALLOW_SEVERITY_ERROR;
    } else if (code > 0) {
        return QALLOW_SEVERITY_WARNING;
    } else {
        return QALLOW_SEVERITY_INFO;
    }
}

int qallow_error_is_recoverable(int code) {
    switch (code) {
        /* Recoverable errors */
        case QALLOW_ERR_TIMEOUT:
        case QALLOW_ERR_RESOURCE_EXHAUSTED:
        case QALLOW_ERR_GPU_NOT_AVAILABLE:
        case QALLOW_ERR_FILE_NOT_FOUND:
            return 1;

        /* Non-recoverable errors */
        case QALLOW_ERR_INVALID_PARAMETER:
        case QALLOW_ERR_NULL_POINTER:
        case QALLOW_ERR_INVARIANT_VIOLATED:
        case QALLOW_ERR_ASSERTION_FAILED:
        case QALLOW_ERR_MEMORY_LIMIT:
            return 0;

        default:
            /* Most errors are potentially recoverable */
            return 1;
    }
}

const char* qallow_error_recovery_hint(int code) {
    switch (code) {
        case QALLOW_ERR_MEMORY_ALLOC:
            return "Try freeing some resources or reducing allocation size";
        case QALLOW_ERR_MEMORY_LIMIT:
            return "Reduce input size or increase available memory";
        case QALLOW_ERR_GPU_NOT_AVAILABLE:
            return "Fall back to CPU execution";
        case QALLOW_ERR_TIMEOUT:
            return "Increase timeout value or optimize operation";
        case QALLOW_ERR_FILE_NOT_FOUND:
            return "Check file path and permissions";
        case QALLOW_ERR_FILE_OPEN_FAILED:
            return "Verify file permissions and availability";
        case QALLOW_ERR_INVALID_PARAMETER:
            return "Check parameter values and types";
        case QALLOW_ERR_INVALID_STATE:
            return "Verify preconditions before retrying";
        case QALLOW_ERR_NULL_POINTER:
            return "Check pointer initialization";
        case QALLOW_ERR_BOUNDS_CHECK:
            return "Verify array/buffer sizes";
        case QALLOW_ERR_ASSERTION_FAILED:
            return "Internal assertion failure - contact support";
        case QALLOW_ERR_INVARIANT_VIOLATED:
            return "Internal invariant violation - contact support";
        default:
            return "No recovery hint available";
    }
}

void qallow_error_log(int code,
                      const char* file,
                      int line,
                      const char* function,
                      const char* context) {
    const char* msg = qallow_error_message(code);
    qallow_error_severity_t severity = qallow_error_severity(code);
    long timestamp = qallow_get_timestamp_ms();

    /* Determine severity prefix */
    const char* severity_str = "INFO";
    if (severity == QALLOW_SEVERITY_DEBUG) {
        severity_str = "DEBUG";
    } else if (severity == QALLOW_SEVERITY_WARNING) {
        severity_str = "WARN";
    } else if (severity == QALLOW_SEVERITY_ERROR) {
        severity_str = "ERROR";
    } else if (severity == QALLOW_SEVERITY_CRITICAL) {
        severity_str = "CRITICAL";
    }

    /* Log to stderr */
    fprintf(stderr, "[%s] %s (code=%d) at %s:%d in %s()\n",
            severity_str, msg, code, file, line, function);

    if (context) {
        fprintf(stderr, "       Context: %s\n", context);
    }

    fprintf(stderr, "       Recovery: %s\n", qallow_error_recovery_hint(code));
    fprintf(stderr, "       Timestamp: %ld ms\n", timestamp);
}
