#ifndef PHASE1_H
#define PHASE1_H




/**
 * Run Phase 12 elasticity simulation
 * @param audit_tag Audit tag to stamp into telemetry rows
 * @param log_path Path to CSV log file (NULL uses default data/logs/phase1.csv)
 * @param ticks Number of simulation ticks to run
 * @param eps Epsilon parameter for elasticity calculations
 * @return 0 on success, non-zero on error
 */
int run_phase1_elasticity(const char* audit_tag,
						   const char* log_path,
						   int ticks,
						   float eps);

#endif // PHASE1_H