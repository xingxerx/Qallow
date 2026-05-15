#ifndef QALLOW_PHASE2_RUNNER_H
#define QALLOW_PHASE2_RUNNER_H

int qallow_phase2_runner(int argc, char** argv);
int run_phase2_harmonic(const char* audit_tag,
						 const char* requested_log_path,
						 int pockets,
						 int ticks,
						 float coupling);

#endif /* QALLOW_PHASE2_RUNNER_H */
