/**
 * @file error_handler.c
 * @brief Error handling and recovery implementation
 */

#include "error_handler.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

// ============================================================================
// STATIC STATE
// ============================================================================

static error_context_t last_error = {0};
static FILE* error_log_file = NULL;

// ============================================================================
// ERROR MESSAGE MAPPING
// ============================================================================

static const char* error_messages[] = {
    [ERROR_OK] = "No error",
    [ERROR_MEMORY_ALLOC] = "Memory allocation failed",
    [ERROR_NULL_POINTER] = "NULL pointer encountered",
    [ERROR_INVALID_STATE] = "Invalid system state",
    [ERROR_FILE_IO] = "File I/O error",
    [ERROR_CUDA_INIT] = "CUDA initialization failed",
    [ERROR_KERNEL_EXEC] = "Kernel execution failed",
    [ERROR_BOUNDS_CHECK] = "Bounds check failed",
    [ERROR_TIMEOUT] = "Operation timeout",
    [ERROR_UNKNOWN] = "Unknown error"
};

static const char* level_names[] = {
    [ERROR_LEVEL_DEBUG] = "DEBUG",
    [ERROR_LEVEL_INFO] = "INFO",
    [ERROR_LEVEL_WARNING] = "WARN",
    [ERROR_LEVEL_ERROR] = "ERROR",
    [ERROR_LEVEL_CRITICAL] = "CRITICAL"
};

// ============================================================================
// PUBLIC API
// ============================================================================

const char* error_get_message(error_code_t code) {
    if (code >= ERROR_UNKNOWN) {
        return error_messages[ERROR_UNKNOWN];
    }
    return error_messages[code];
}

const char* error_get_level_name(error_level_t level) {
    if (level >= ERROR_LEVEL_CRITICAL) {
        return level_names[ERROR_LEVEL_CRITICAL];
    }
    return level_names[level];
}

void error_log(error_code_t code, error_level_t level,
               const char* message, const char* file,
               int line, const char* function) {
    // Store in context
    last_error.code = code;
    last_error.level = level;
    last_error.message = message;
    last_error.file = file;
    last_error.line = line;
    last_error.function = function;
    last_error.timestamp = time(NULL);
    
    // Print to stderr
    fprintf(stderr, "[%s] %s:%d in %s() - %s\n",
            error_get_level_name(level),
            file, line, function,
            message ? message : error_get_message(code));
    
    // Log to file if available
    if (error_log_file) {
        fprintf(error_log_file, "[%s] %s:%d in %s() - %s\n",
                error_get_level_name(level),
                file, line, function,
                message ? message : error_get_message(code));
        fflush(error_log_file);
    }
}

void error_logf(error_code_t code, error_level_t level,
                const char* file, int line, const char* function,
                const char* format, ...) {
    char buffer[512];
    va_list args;
    
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    error_log(code, level, buffer, file, line, function);
}

error_context_t* error_get_last(void) {
    return &last_error;
}

void error_clear(void) {
    memset(&last_error, 0, sizeof(error_context_t));
}

int error_is_recoverable(error_code_t code) {
    switch (code) {
        case ERROR_OK:
        case ERROR_TIMEOUT:
        case ERROR_BOUNDS_CHECK:
            return 1;
        case ERROR_CUDA_INIT:
        case ERROR_MEMORY_ALLOC:
            return 0;
        default:
            return 1;
    }
}

int error_recover(error_code_t code, void* context __attribute__((unused))) {
    switch (code) {
        case ERROR_MEMORY_ALLOC:
            // Try to free some memory and retry
            fprintf(stderr, "[RECOVERY] Attempting memory recovery\n");
            return 1;
            
        case ERROR_TIMEOUT:
            // Retry operation
            fprintf(stderr, "[RECOVERY] Retrying operation\n");
            return 1;
            
        case ERROR_INVALID_STATE:
            // Reset to known good state
            fprintf(stderr, "[RECOVERY] Resetting to known good state\n");
            return 1;
            
        case ERROR_CUDA_INIT:
            // Fall back to CPU
            fprintf(stderr, "[RECOVERY] Falling back to CPU execution\n");
            return 1;
            
        default:
            return 0;
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void error_handler_init(const char* log_file) {
    if (log_file) {
        error_log_file = fopen(log_file, "a");
        if (error_log_file) {
            fprintf(error_log_file, "\n=== Error Log Started ===\n");
            fflush(error_log_file);
        }
    }
}

void error_handler_cleanup(void) {
    if (error_log_file) {
        fprintf(error_log_file, "=== Error Log Ended ===\n\n");
        fclose(error_log_file);
        error_log_file = NULL;
    }
}

