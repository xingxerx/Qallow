#ifndef QALLOW_PHASE1_RUNNER_H
#define QALLOW_PHASE1_RUNNER_H

int qallow_phase1_runner(int argc, char** argv);
int run_phase1_elasticity(const char* audit_tag,
						   const char* requested_log_path,
						   int ticks,
						   float eps);

#endif /* QALLOW_PHASE1_RUNNER_H */
