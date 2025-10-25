/**
 * COMPREHENSIVE ERROR HANDLER
 * Handles all internal errors and provides detailed logging
 */

const fs = require('fs');
const path = require('path');

class ErrorHandler {
  constructor() {
    this.logDir = path.join(__dirname, '../logs');
    this.ensureLogDir();
    this.errorLog = [];
    this.maxErrors = 1000;
  }

  ensureLogDir() {
    if (!fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }
  }

  log(level, message, error = null, context = {}) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      message,
      error: error ? {
        name: error.name,
        message: error.message,
        stack: error.stack
      } : null,
      context
    };

    // Add to in-memory log
    this.errorLog.push(logEntry);
    if (this.errorLog.length > this.maxErrors) {
      this.errorLog.shift();
    }

    // Write to file
    this.writeToFile(logEntry);

    // Console output
    this.consoleOutput(logEntry);

    return logEntry;
  }

  writeToFile(logEntry) {
    try {
      const logFile = path.join(this.logDir, `qallow-${new Date().toISOString().split('T')[0]}.log`);
      const logLine = JSON.stringify(logEntry) + '\n';
      fs.appendFileSync(logFile, logLine);
    } catch (err) {
      console.error('Failed to write to log file:', err);
    }
  }

  consoleOutput(logEntry) {
    const { timestamp, level, message, error } = logEntry;
    const prefix = `[${level}] ${timestamp}`;

    switch (level) {
      case 'ERROR':
        console.error(`${prefix} - ${message}`, error || '');
        break;
      case 'WARN':
        console.warn(`${prefix} - ${message}`);
        break;
      case 'INFO':
        console.log(`${prefix} - ${message}`);
        break;
      case 'DEBUG':
        console.debug(`${prefix} - ${message}`);
        break;
      default:
        console.log(`${prefix} - ${message}`);
    }
  }

  info(message, context = {}) {
    return this.log('INFO', message, null, context);
  }

  warn(message, context = {}) {
    return this.log('WARN', message, null, context);
  }

  error(message, error = null, context = {}) {
    return this.log('ERROR', message, error, context);
  }

  debug(message, context = {}) {
    return this.log('DEBUG', message, null, context);
  }

  // Get error statistics
  getStats() {
    const stats = {
      total: this.errorLog.length,
      byLevel: {
        ERROR: 0,
        WARN: 0,
        INFO: 0,
        DEBUG: 0
      },
      recent: this.errorLog.slice(-10)
    };

    this.errorLog.forEach(entry => {
      stats.byLevel[entry.level]++;
    });

    return stats;
  }

  // Clear old logs
  clearOldLogs(daysOld = 7) {
    try {
      const files = fs.readdirSync(this.logDir);
      const now = Date.now();
      const maxAge = daysOld * 24 * 60 * 60 * 1000;

      files.forEach(file => {
        const filePath = path.join(this.logDir, file);
        const stats = fs.statSync(filePath);
        if (now - stats.mtime.getTime() > maxAge) {
          fs.unlinkSync(filePath);
          this.info(`Deleted old log file: ${file}`);
        }
      });
    } catch (err) {
      this.error('Failed to clear old logs:', err);
    }
  }

  // Export logs
  exportLogs(format = 'json') {
    try {
      if (format === 'json') {
        return JSON.stringify(this.errorLog, null, 2);
      } else if (format === 'csv') {
        let csv = 'timestamp,level,message,error\n';
        this.errorLog.forEach(entry => {
          csv += `"${entry.timestamp}","${entry.level}","${entry.message}","${entry.error ? entry.error.message : ''}"\n`;
        });
        return csv;
      }
    } catch (err) {
      this.error('Failed to export logs:', err);
      return null;
    }
  }
}

// Health check system
class HealthCheck {
  constructor() {
    this.checks = {};
    this.lastCheck = null;
  }

  register(name, checkFn) {
    this.checks[name] = checkFn;
  }

  async runAll() {
    const results = {};
    const timestamp = new Date().toISOString();

    for (const [name, checkFn] of Object.entries(this.checks)) {
      try {
        results[name] = {
          status: 'healthy',
          result: await checkFn(),
          timestamp
        };
      } catch (err) {
        results[name] = {
          status: 'unhealthy',
          error: err.message,
          timestamp
        };
      }
    }

    this.lastCheck = results;
    return results;
  }

  getStatus() {
    if (!this.lastCheck) return { status: 'unknown' };

    const allHealthy = Object.values(this.lastCheck).every(
      check => check.status === 'healthy'
    );

    return {
      status: allHealthy ? 'healthy' : 'unhealthy',
      checks: this.lastCheck,
      timestamp: new Date().toISOString()
    };
  }
}

// Circuit breaker pattern for fault tolerance
class CircuitBreaker {
  constructor(name, threshold = 5, timeout = 60000) {
    this.name = name;
    this.threshold = threshold;
    this.timeout = timeout;
    this.failures = 0;
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.lastFailureTime = null;
  }

  async execute(fn) {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.timeout) {
        this.state = 'HALF_OPEN';
      } else {
        throw new Error(`Circuit breaker ${this.name} is OPEN`);
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (err) {
      this.onFailure();
      throw err;
    }
  }

  onSuccess() {
    this.failures = 0;
    this.state = 'CLOSED';
  }

  onFailure() {
    this.failures++;
    this.lastFailureTime = Date.now();
    if (this.failures >= this.threshold) {
      this.state = 'OPEN';
    }
  }

  getStatus() {
    return {
      name: this.name,
      state: this.state,
      failures: this.failures,
      lastFailureTime: this.lastFailureTime
    };
  }
}

module.exports = {
  ErrorHandler,
  HealthCheck,
  CircuitBreaker
};

