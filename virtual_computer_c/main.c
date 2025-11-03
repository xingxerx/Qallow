#include "virtual_computer.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    VirtualComputer vc;
    if (vc_init(&vc) != 0) {
        fprintf(stderr, "Failed to initialise virtual computer\n");
        return EXIT_FAILURE;
    }

    Workload numeric = {.type = TASK_NUMERIC, .payload = NULL, .payload_size = 0};
    Workload spiking = {.type = TASK_SPIKING, .payload = NULL, .payload_size = 0};
    Workload photonic = {.type = TASK_PHOTONIC, .payload = NULL, .payload_size = 0};

    vc_run_task(&vc, &numeric);
    vc_run_task(&vc, &spiking);
    vc_run_task(&vc, &photonic);

    vc_free(&vc);
    return EXIT_SUCCESS;
}
