#ifndef QALLOW_ERROR_CODES_H
#define QALLOW_ERROR_CODES_H

/**
 * @file error_codes.h
 * @brief Standardized error codes for Qallow system
 * 
 * All error conditions across modules should use these codes for:
 * - Consistent error handling
 * - Better debugging
 * - Automated error recovery
 * - Cross-module error communication
 * 
 * Error Propagation Pattern:
 *   1. Check return code
 *   2. Get error message with qallow_error_message()
 *   3. Decide recovery: qallow_error_is_recoverable()
 *   4. Log context: qallow_error_log_context()
 */

/* ============================================================================
 * Error Code Ranges (reserved by subsystem)
 * ============================================================================ */

#define QALLOW_SUCCESS                  0     /* No error */

/* Core System: 1-99 */
#define QALLOW_ERR_UNKNOWN              1     /* Unknown/unclassified error */
#define QALLOW_ERR_NOT_IMPLEMENTED      2     /* Feature not yet implemented */
#define QALLOW_ERR_UNSUPPORTED          3     /* Operation not supported on this platform */

/* Memory Management: 100-199 */

#define QALLOW_ERR_MEMORY_LIMIT         101   /* Exceeded memory limit */
#define QALLOW_ERR_NULL_POINTER         102   /* Null pointer dereference attempt */

/* Input Validation: 200-299 */
#define QALLOW_ERR_INVALID_PARAMETER    200   /* Invalid parameter passed */
#define QALLOW_ERR_INVALID_STATE        201   /* Invalid state for operation */
#define QALLOW_ERR_BOUNDS_CHECK         202   /* Array/buffer bounds exceeded */
#define QALLOW_ERR_TYPE_MISMATCH        203   /* Type mismatch in conversion */

/* File I/O: 300-399 */
#define QALLOW_ERR_FILE_NOT_FOUND       300   /* File does not exist */
#define QALLOW_ERR_FILE_OPEN_FAILED     301   /* Failed to open file */
#define QALLOW_ERR_FILE_READ_FAILED     302   /* Failed to read from file */
#define QALLOW_ERR_FILE_WRITE_FAILED    303   /* Failed to write to file */
#define QALLOW_ERR_PATH_INVALID         304   /* Invalid file path */

/* Hardware/CUDA: 400-499 */
#define QALLOW_ERR_CUDA_INIT            400   /* CUDA initialization failed */
#define QALLOW_ERR_CUDA_DEVICE          401   /* CUDA device error */
#define QALLOW_ERR_CUDA_KERNEL          402   /* CUDA kernel execution failed */
#define QALLOW_ERR_CUDA_MEMORY          403   /* CUDA memory error */
#define QALLOW_ERR_GPU_NOT_AVAILABLE    404   /* GPU not available/not found */

/* Quantum: 500-599 */
#define QALLOW_ERR_QUANTUM_CIRCUIT      500   /* Invalid quantum circuit */
#define QALLOW_ERR_QUANTUM_STATE        501   /* Invalid quantum state */
#define QALLOW_ERR_QUBIT_LIMIT          502   /* Exceeded qubit limit */
#define QALLOW_ERR_QUANTUM_SIM          503   /* Quantum simulator error */

/* Networking: 600-699 */
#define QALLOW_ERR_SOCKET_CREATE        600   /* Failed to create socket */
#define QALLOW_ERR_SOCKET_BIND          601   /* Failed to bind socket */
#define QALLOW_ERR_SOCKET_LISTEN        602   /* Failed to listen on socket */
#define QALLOW_ERR_SOCKET_CONNECT       603   /* Connection failed */
#define QALLOW_ERR_SOCKET_SEND          604   /* Send failed */
#define QALLOW_ERR_SOCKET_RECV          605   /* Receive failed */

/* Database: 700-799 */
#define QALLOW_ERR_DB_OPEN              700   /* Database open failed */
#define QALLOW_ERR_DB_QUERY             701   /* Database query failed */
#define QALLOW_ERR_DB_COMMIT            702   /* Database commit failed */
#define QALLOW_ERR_DB_CONSTRAINT        703   /* Constraint violation */

/* Timeout/Performance: 800-899 */
#define QALLOW_ERR_TIMEOUT              800   /* Operation timeout */
#define QALLOW_ERR_DEADLOCK             801   /* Deadlock detected */
#define QALLOW_ERR_RESOURCE_EXHAUSTED   802   /* Resource exhausted */

/* Logic/Assertion: 900-999 */
#define QALLOW_ERR_ASSERTION_FAILED     900   /* Assertion failed */
#define QALLOW_ERR_INVARIANT_VIOLATED   901   /* Invariant violated */
#define QALLOW_ERR_LOGIC_ERROR          902   /* Logic error detected */

/* ============================================================================
 * Error Severity Levels
 * ============================================================================ */

typedef enum {
    QALLOW_SEVERITY_DEBUG = 0,      /* Debug information only */
    QALLOW_SEVERITY_INFO = 1,       /* Informational message */
    QALLOW_SEVERITY_WARNING = 2,    /* Warning - operation may fail */
    QALLOW_SEVERITY_ERROR = 3,      /* Error - operation failed */
    QALLOW_SEVERITY_CRITICAL = 4    /* Critical - system may be unstable */
} qallow_error_severity_t;

/* ============================================================================
 * Error Context Structure
 * ============================================================================ */

typedef struct {
    int code;                       /* Error code from above */
    const char* message;            /* Error message string */
    const char* file;               /* Source file name */
    int line;                       /* Source line number */
    const char* function;           /* Function name */
    qallow_error_severity_t severity;
    long timestamp_ms;              /* When error occurred */
} qallow_error_context_t;

/* ============================================================================
 * Error Handling Functions (in error_handler.c)
 * ============================================================================ */

/**
 * Get human-readable error message
 * @param code - error code
 * @return Pointer to error message string
 */
const char* qallow_error_message(int code);

/**
 * Get error severity level
 * @param code - error code
 * @return Severity level
 */
qallow_error_severity_t qallow_error_severity(int code);

/**
 * Check if error is recoverable
 * @param code - error code
 * @return Non-zero if error is recoverable
 */
int qallow_error_is_recoverable(int code);

/**
 * Log error with full context
 * @param code - error code
 * @param file - source file
 * @param line - source line
 * @param function - function name
 * @param context - additional context message
 */
void qallow_error_log(int code, 
                      const char* file, 
                      int line, 
                      const char* function,
                      const char* context);

/**
 * Get suggested recovery action for error
 * @param code - error code
 * @return Recovery action description
 */
const char* qallow_error_recovery_hint(int code);

/* ============================================================================
 * Convenience Macros
 * ============================================================================ */

/* Log error with automatic file/line/function */
#define QALLOW_ERROR_LOG(code, context) \
    qallow_error_log(code, __FILE__, __LINE__, __FUNCTION__, context)

/* Check pointer, log error, and return code */
#define QALLOW_CHECK_NULL(ptr, error_code) \
    do { \
        if (!(ptr)) { \
            QALLOW_ERROR_LOG(error_code, #ptr); \
            return error_code; \
        } \
    } while (0)

/* Check bounds, log error, and return code */
#define QALLOW_CHECK_BOUNDS(index, limit, error_code) \
    do { \
        if ((index) < 0 || (index) >= (limit)) { \
            QALLOW_ERROR_LOG(error_code, #index " out of bounds"); \
            return error_code; \
        } \
    } while (0)

/* Check allocation, log error, and return code */
#define QALLOW_CHECK_ALLOC(ptr, size, error_code) \
    do { \
        if (!(ptr)) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "allocation failed for %zu bytes", (size_t)(size)); \
            QALLOW_ERROR_LOG(error_code, buf); \
            return error_code; \
        } \
    } while (0)

#endif  /* QALLOW_ERROR_CODES_H */
