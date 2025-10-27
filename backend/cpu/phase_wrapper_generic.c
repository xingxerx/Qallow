/**
 * Generic Phase Wrapper for Phases 16-20
 * Provides runner interface for standalone phase implementations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

static int execute_phase_binary(int phase_num, int argc, char** argv) {
    // Map phase number to binary path
    const char* phase_paths[] = {
        NULL,  // 0
        NULL,  // 1-15 handled by other runners
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        "phases/phase_16_rebellion",      // 16
        "phases/phase_17_memory",         // 17
        "phases/phase_18_multiplayer",    // 18
        "phases/phase_19_audit",          // 19
        "phases/phase_20_loreweave"       // 20
    };
    
    if (phase_num < 16 || phase_num > 20) {
        fprintf(stderr, "[PHASE%d] ERROR: Invalid phase number\n", phase_num);
        return 1;
    }
    
    const char* phase_path = phase_paths[phase_num];
    
    // Check if the phase binary exists
    if (access(phase_path, X_OK) != 0) {
        fprintf(stderr, "[PHASE%d] ERROR: Phase binary not found at %s\n", phase_num, phase_path);
        return 1;
    }
    
    // Fork and execute phase
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[PHASE%d] ERROR: Failed to fork process\n", phase_num);
        return 1;
    }
    
    if (pid == 0) {
        // Child process - execute phase
        char** phase_argv = malloc((argc + 1) * sizeof(char*));
        phase_argv[0] = (char*)phase_path;
        for (int i = 1; i < argc; i++) {
            phase_argv[i] = argv[i];
        }
        phase_argv[argc] = NULL;
        
        execv(phase_path, phase_argv);
        
        // If execv returns, there was an error
        fprintf(stderr, "[PHASE%d] ERROR: Failed to execute phase\n", phase_num);
        exit(1);
    } else {
        // Parent process - wait for child
        int status;
        waitpid(pid, &status, 0);
        
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            fprintf(stderr, "[PHASE%d] ERROR: Phase process terminated abnormally\n", phase_num);
            return 1;
        }
    }
}

int qallow_phase16_runner(int argc, char** argv) {
    return execute_phase_binary(16, argc, argv);
}

int qallow_phase17_runner(int argc, char** argv) {
    return execute_phase_binary(17, argc, argv);
}

int qallow_phase18_runner(int argc, char** argv) {
    return execute_phase_binary(18, argc, argv);
}

int qallow_phase19_runner(int argc, char** argv) {
    return execute_phase_binary(19, argc, argv);
}

int qallow_phase20_runner(int argc, char** argv) {
    return execute_phase_binary(20, argc, argv);
}

