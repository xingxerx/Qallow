#ifndef QALLOW_PHASE12_RUNNER_H
#define QALLOW_PHASE12_RUNNER_H

int qallow_phase12_runner(int argc, char** argv);
int run_phase12_elasticity(const char* audit_tag,
						   const char* requested_log_path,
						   int ticks,
						   float eps);

#endif /* QALLOW_PHASE12_RUNNER_H */
