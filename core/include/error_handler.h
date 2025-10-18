/**
 * @file error_handler.h
 * @brief Error handling and recovery system for Qallow
 * 
 * Provides comprehensive error handling, logging, and recovery mechanisms
 */

#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <stdio.h>
#include <time.h>

// Error severity levels
typedef enum {
    ERROR_LEVEL_DEBUG = 0,
    ERROR_LEVEL_INFO = 1,
    ERROR_LEVEL_WARNING = 2,
    ERROR_LEVEL_ERROR = 3,
    ERROR_LEVEL_CRITICAL = 4
} error_level_t;

// Error codes
typedef enum {
    ERROR_OK = 0,
    ERROR_MEMORY_ALLOC = 1,
    ERROR_NULL_POINTER = 2,
    ERROR_INVALID_STATE = 3,
    ERROR_FILE_IO = 4,
    ERROR_CUDA_INIT = 5,
    ERROR_KERNEL_EXEC = 6,
    ERROR_BOUNDS_CHECK = 7,
    ERROR_TIMEOUT = 8,
    ERROR_UNKNOWN = 99
} error_code_t;

// Error context structure
typedef struct {
    error_code_t code;
    error_level_t level;
    const char* message;
    const char* file;
    int line;
    const char* function;
    time_t timestamp;
} error_context_t;

// ============================================================================
// ERROR LOGGING
// ============================================================================

/**
 * Log an error with full context
 */
void error_log(error_code_t code, error_level_t level, 
               const char* message, const char* file, 
               int line, const char* function);

/**
 * Log error with printf-style formatting
 */
void error_logf(error_code_t code, error_level_t level,
                const char* file, int line, const char* function,
                const char* format, ...);

/**
 * Get human-readable error message
 */
const char* error_get_message(error_code_t code);

/**
 * Get human-readable level name
 */
const char* error_get_level_name(error_level_t level);

// ============================================================================
// ERROR RECOVERY
// ============================================================================

/**
 * Attempt to recover from error
 */
int error_recover(error_code_t code, void* context);

/**
 * Check if error is recoverable
 */
int error_is_recoverable(error_code_t code);

/**
 * Get last error
 */
error_context_t* error_get_last(void);

/**
 * Clear error state
 */
void error_clear(void);

// ============================================================================
// CONVENIENCE MACROS
// ============================================================================

#define ERROR_LOG(code, level, msg) \
    error_log(code, level, msg, __FILE__, __LINE__, __func__)

#define ERROR_LOGF(code, level, fmt, ...) \
    error_logf(code, level, __FILE__, __LINE__, __func__, fmt, __VA_ARGS__)

#define ERROR_CHECK_NULL(ptr, code) \
    do { \
        if ((ptr) == NULL) { \
            ERROR_LOG(code, ERROR_LEVEL_ERROR, "NULL pointer check failed"); \
            return code; \
        } \
    } while(0)

#define ERROR_CHECK_BOUNDS(val, min, max, code) \
    do { \
        if ((val) < (min) || (val) > (max)) { \
            ERROR_LOGF(code, ERROR_LEVEL_ERROR, \
                      "Bounds check failed: %f not in [%f, %f]", \
                      (double)(val), (double)(min), (double)(max)); \
            return code; \
        } \
    } while(0)

#define ERROR_CHECK_ALLOC(ptr, code) \
    do { \
        if ((ptr) == NULL) { \
            ERROR_LOG(code, ERROR_LEVEL_CRITICAL, "Memory allocation failed"); \
            return code; \
        } \
    } while(0)

#endif // ERROR_HANDLER_H

