#include "qallow_phase12.h"
#include "qallow_phase13.h"
#include "qallow_integration.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

static void ensure_log_dirs(void) {
    mkdir("data", 0755);
    mkdir("data/logs", 0755);
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    ensure_log_dirs();

    char phase12_prog[] = "qallow_tests";
    char phase12_cmd[] = "phase12";
    char phase12_ticks[] = "--ticks=20";
    char phase12_eps[] = "--eps=0.001";
    char* phase12_args[] = {phase12_prog, phase12_cmd, phase12_ticks, phase12_eps};
    if (qallow_phase12_runner(4, phase12_args) != 0) {
        fprintf(stderr, "[integration] phase12 runner failed\n");
        return EXIT_FAILURE;
    }

    char phase13_prog[] = "qallow_tests";
    char phase13_cmd[] = "phase13";
    char phase13_nodes[] = "--nodes=8";
    char phase13_ticks[] = "--ticks=50";
    char phase13_k[] = "--k=0.001";
    char* phase13_args[] = {phase13_prog, phase13_cmd, phase13_nodes, phase13_ticks, phase13_k};
    if (qallow_phase13_runner(5, phase13_args) != 0) {
        fprintf(stderr, "[integration] phase13 runner failed\n");
        return EXIT_FAILURE;
    }

    qallow_lattice_config_t lattice_cfg;
    qallow_lattice_config_init(&lattice_cfg);
    lattice_cfg.ticks = 32;
    lattice_cfg.no_split = true;
    lattice_cfg.print_summary = false;
    if (qallow_lattice_integrate(&lattice_cfg) != 0) {
        fprintf(stderr, "[integration] unified lattice integration failed\n");
        return EXIT_FAILURE;
    }

    puts("integration smoke test passed");
    return EXIT_SUCCESS;
}
