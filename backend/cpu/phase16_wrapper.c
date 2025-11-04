/**
 * Phase 16 Wrapper - Rebellion Simulation
 * Provides runner interface for phase 16
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int qallow_phase16_runner(int argc, char** argv) {




    const char* phase16_path = "phases/phase_16_constraint_validation";


    if (access(phase16_path, X_OK) != 0) {
        fprintf(stderr, "[PHASE16] ERROR: Phase 16 binary not found at %s\n", phase16_path);
        return 1;
    }


    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[PHASE16] ERROR: Failed to fork process\n");
        return 1;
    }

    if (pid == 0) {



        phase_argv[0] = (char*)phase16_path;
        for (int i = 1; i < argc; i++) {
            phase_argv[i] = argv[i];
        }
        phase_argv[argc] = NULL;

        execv(phase16_path, phase_argv);


        fprintf(stderr, "[PHASE16] ERROR: Failed to execute phase 16\n");
        exit(1);
    } else {

        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            fprintf(stderr, "[PHASE16] ERROR: Phase 16 process terminated abnormally\n");
            return 1;
        }
    }
}
