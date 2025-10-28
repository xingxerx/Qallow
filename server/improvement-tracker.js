/**
 * Real-time Improvement Tracker
 * Logs all code improvements and optimizations made during runtime
 */

const fs = require('fs');
const path = require('path');

class ImprovementTracker {
  constructor(logDir = '/root/Qallow/data/improvements') {
    this.logDir = logDir;
    this.improvements = [];
    this.startTime = Date.now();
    
    // Create log directory if it doesn't exist
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }
  }

  /**
   * Log an improvement
   */
  logImprovement(category, title, description, impact = 'MEDIUM', files = []) {
    const improvement = {
      id: `imp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      category,
      title,
      description,
      impact,
      files,
      status: 'IMPLEMENTED'
    };

    this.improvements.push(improvement);
    this.saveImprovement(improvement);
    
    return improvement;
  }

  /**
   * Save individual improvement to file
   */
  saveImprovement(improvement) {
    const filename = `${improvement.id}.json`;
    const filepath = path.join(this.logDir, filename);
    
    try {
      fs.writeFileSync(filepath, JSON.stringify(improvement, null, 2));
    } catch (err) {
      console.error(`Failed to save improvement: ${err.message}`);
    }
  }

  /**
   * Generate improvement report
   */
  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      uptime: Date.now() - this.startTime,
      totalImprovements: this.improvements.length,
      byCategory: {},
      byImpact: {},
      improvements: this.improvements
    };

    // Group by category
    this.improvements.forEach(imp => {
      if (!report.byCategory[imp.category]) {
        report.byCategory[imp.category] = [];
      }
      report.byCategory[imp.category].push(imp);
    });

    // Group by impact
    this.improvements.forEach(imp => {
      if (!report.byImpact[imp.impact]) {
        report.byImpact[imp.impact] = [];
      }
      report.byImpact[imp.impact].push(imp);
    });

    return report;
  }

  /**
   * Export report to file
   */
  exportReport() {
    const report = this.generateReport();
    const filename = `improvement_report_${Date.now()}.json`;
    const filepath = path.join(this.logDir, filename);

    try {
      fs.writeFileSync(filepath, JSON.stringify(report, null, 2));
      return { success: true, filepath, filename };
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  /**
   * Get summary
   */
  getSummary() {
    const report = this.generateReport();
    return {
      timestamp: report.timestamp,
      uptime: report.uptime,
      totalImprovements: report.totalImprovements,
      categories: Object.keys(report.byCategory),
      impacts: Object.keys(report.byImpact),
      recentImprovements: this.improvements.slice(-5)
    };
  }
}

module.exports = ImprovementTracker;

