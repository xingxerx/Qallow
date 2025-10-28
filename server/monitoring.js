/**
 * Advanced Monitoring System for Qallow
 * Real-time health checks, performance analysis, and adaptive optimization
 */

class QallowMonitor {
  constructor() {
    this.healthChecks = [];
    this.performanceThresholds = {
      minCoherence: 0.7,
      minFidelity: 0.8,
      maxPhaseTime: 120000, // 2 minutes
      maxCycleTime: 2400000 // 40 minutes
    };
    this.alerts = [];
    this.optimizations = [];
  }

  /**
   * Perform health check on system metrics
   */
  performHealthCheck(metrics, phaseNumber) {
    const check = {
      timestamp: Date.now(),
      phase: phaseNumber,
      status: 'HEALTHY',
      issues: []
    };

    // Check coherence
    if (metrics.coherence && metrics.coherence < this.performanceThresholds.minCoherence) {
      check.status = 'WARNING';
      check.issues.push(`Low coherence: ${metrics.coherence.toFixed(4)}`);
    }

    // Check fidelity
    if (metrics.fidelity && metrics.fidelity < this.performanceThresholds.minFidelity) {
      check.status = 'WARNING';
      check.issues.push(`Low fidelity: ${metrics.fidelity.toFixed(4)}`);
    }

    // Check energy
    if (metrics.energy && metrics.energy > 0.9) {
      check.status = 'WARNING';
      check.issues.push(`High energy consumption: ${metrics.energy.toFixed(4)}`);
    }

    this.healthChecks.push(check);
    if (this.healthChecks.length > 1000) {
      this.healthChecks.shift();
    }

    return check;
  }

  /**
   * Analyze performance trends
   */
  analyzePerformanceTrends(phaseTimings, cycleTimings) {
    const analysis = {
      timestamp: Date.now(),
      phaseAnalysis: {},
      cycleAnalysis: {}
    };

    // Analyze phase timings
    for (const [phase, timings] of Object.entries(phaseTimings)) {
      if (Array.isArray(timings) && timings.length > 0) {
        const avg = timings.reduce((a, b) => a + b, 0) / timings.length;
        const max = Math.max(...timings);
        const min = Math.min(...timings);
        const trend = timings.length > 1 ? timings[timings.length - 1] - timings[0] : 0;

        analysis.phaseAnalysis[phase] = {
          average: avg,
          max,
          min,
          trend,
          count: timings.length
        };
      }
    }

    // Analyze cycle timings
    if (cycleTimings.length > 0) {
      const avg = cycleTimings.reduce((a, b) => a + b, 0) / cycleTimings.length;
      const max = Math.max(...cycleTimings);
      const min = Math.min(...cycleTimings);
      const trend = cycleTimings.length > 1 ? cycleTimings[cycleTimings.length - 1] - cycleTimings[0] : 0;

      analysis.cycleAnalysis = {
        average: avg,
        max,
        min,
        trend,
        count: cycleTimings.length
      };
    }

    return analysis;
  }

  /**
   * Generate optimization recommendations
   */
  generateOptimizations(metrics, phaseTimings, cycleTimings) {
    const recommendations = [];

    // Check for slow phases
    for (const [phase, timings] of Object.entries(phaseTimings)) {
      if (Array.isArray(timings) && timings.length > 0) {
        const avg = timings.reduce((a, b) => a + b, 0) / timings.length;
        if (avg > this.performanceThresholds.maxPhaseTime) {
          recommendations.push({
            type: 'PERFORMANCE',
            phase: parseInt(phase),
            suggestion: `Phase ${phase} is slow (avg: ${avg.toFixed(0)}ms). Consider increasing ticks or optimizing algorithm.`,
            severity: 'HIGH'
          });
        }
      }
    }

    // Check for low coherence trend
    if (metrics.coherence && metrics.coherence < 0.75) {
      recommendations.push({
        type: 'STABILITY',
        suggestion: 'System coherence is degrading. Consider reducing phase complexity or increasing error correction.',
        severity: 'MEDIUM'
      });
    }

    // Check for high energy
    if (metrics.energy && metrics.energy > 0.85) {
      recommendations.push({
        type: 'EFFICIENCY',
        suggestion: 'High energy consumption detected. Consider switching to CPU build or reducing workload.',
        severity: 'MEDIUM'
      });
    }

    this.optimizations = recommendations;
    return recommendations;
  }

  /**
   * Get health summary
   */
  getHealthSummary() {
    const recentChecks = this.healthChecks.slice(-100);
    const healthyCount = recentChecks.filter(c => c.status === 'HEALTHY').length;
    const warningCount = recentChecks.filter(c => c.status === 'WARNING').length;
    const errorCount = recentChecks.filter(c => c.status === 'ERROR').length;

    return {
      timestamp: Date.now(),
      totalChecks: recentChecks.length,
      healthy: healthyCount,
      warnings: warningCount,
      errors: errorCount,
      healthPercentage: (healthyCount / recentChecks.length * 100).toFixed(2),
      recentIssues: recentChecks.filter(c => c.issues.length > 0).slice(-10)
    };
  }

  /**
   * Export monitoring data
   */
  exportData() {
    return {
      timestamp: Date.now(),
      healthChecks: this.healthChecks,
      optimizations: this.optimizations,
      healthSummary: this.getHealthSummary()
    };
  }
}

module.exports = QallowMonitor;

