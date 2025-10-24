#ifndef QUANTUM_SUITE_H
#define QUANTUM_SUITE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct quantum_suite_metrics_s {
    int valid;
    int algorithms_total;
    int algorithms_passed;
    double grover_probability;
    double vqe_best_energy;
    char timestamp[64];
} quantum_suite_metrics_t;

int quantum_run_all(quantum_suite_metrics_t* out_metrics);
const quantum_suite_metrics_t* quantum_suite_get_metrics(void);

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_SUITE_H */
