const fs = require('fs');
const path = require('path');
const { ErrorHandler, HealthCheck, CircuitBreaker } = require('../errorHandler');

describe('ErrorHandler Tests', () => {
  let errorHandler;
  const testLogDir = path.join(__dirname, '../logs');

  beforeEach(() => {
    errorHandler = new ErrorHandler();
    // Ensure logs directory exists
    if (!fs.existsSync(testLogDir)) {
      fs.mkdirSync(testLogDir, { recursive: true });
    }
  });

  afterEach(() => {
    // Clean up test logs
    if (fs.existsSync(testLogDir)) {
      const files = fs.readdirSync(testLogDir);
      files.forEach((file) => {
        if (file.startsWith('test-')) {
          fs.unlinkSync(path.join(testLogDir, file));
        }
      });
    }
  });

  describe('Logging Methods', () => {
    test('info() should log info messages', () => {
      const result = errorHandler.info('Test info message');
      expect(result).toBeDefined();
      expect(result.level).toBe('INFO');
      expect(result.message).toBe('Test info message');
    });

    test('error() should log error messages', () => {
      const testError = new Error('Test error');
      const result = errorHandler.error('Error occurred', testError);
      expect(result).toBeDefined();
      expect(result.level).toBe('ERROR');
      expect(result.error).toBeDefined();
    });

    test('warn() should log warning messages', () => {
      const result = errorHandler.warn('Test warning');
      expect(result).toBeDefined();
      expect(result.level).toBe('WARN');
    });

    test('debug() should log debug messages', () => {
      const result = errorHandler.debug('Test debug', { key: 'value' });
      expect(result).toBeDefined();
      expect(result.level).toBe('DEBUG');
    });
  });

  describe('Error Buffer', () => {
    test('should maintain error buffer', () => {
      errorHandler.info('Message 1');
      errorHandler.info('Message 2');
      errorHandler.info('Message 3');
      expect(errorHandler.errorLog.length).toBeGreaterThanOrEqual(3);
    });

    test('should not exceed max errors', () => {
      for (let i = 0; i < 1100; i++) {
        errorHandler.info(`Message ${i}`);
      }
      expect(errorHandler.errorLog.length).toBeLessThanOrEqual(1000);
    });
  });

  describe('Log Entry Format', () => {
    test('log entry should have required fields', () => {
      const result = errorHandler.info('Test message', { context: 'test' });
      expect(result).toHaveProperty('timestamp');
      expect(result).toHaveProperty('level');
      expect(result).toHaveProperty('message');
      expect(result).toHaveProperty('context');
    });

    test('timestamp should be ISO format', () => {
      const result = errorHandler.info('Test');
      const timestamp = new Date(result.timestamp);
      expect(timestamp).toBeInstanceOf(Date);
      expect(timestamp.getTime()).not.toBeNaN();
    });
  });
});

describe('HealthCheck Tests', () => {
  let healthCheck;

  beforeEach(() => {
    healthCheck = new HealthCheck();
  });

  describe('Health Check Registration', () => {
    test('should register health checks', () => {
      const checkFn = jest.fn().mockResolvedValue(true);
      healthCheck.register('test-check', checkFn);
      expect(healthCheck.checks['test-check']).toBeDefined();
    });

    test('should run all registered checks', async () => {
      const check1 = jest.fn().mockResolvedValue(true);
      const check2 = jest.fn().mockResolvedValue(true);
      healthCheck.register('check1', check1);
      healthCheck.register('check2', check2);

      const results = await healthCheck.runAll();
      expect(Object.keys(results).length).toBe(2);
    });
  });

  describe('Health Check Results', () => {
    test('should return healthy status for passing checks', async () => {
      healthCheck.register('memory', async () => ({ used: 100, total: 1000 }));
      const results = await healthCheck.runAll();
      expect(results.memory.status).toBe('healthy');
    });

    test('should return unhealthy status for failing checks', async () => {
      healthCheck.register('cpu', async () => {
        throw new Error('CPU check failed');
      });
      const results = await healthCheck.runAll();
      expect(results.cpu.status).toBe('unhealthy');
      expect(results.cpu.error).toBeDefined();
    });
  });
});

describe('CircuitBreaker Tests', () => {
  let circuitBreaker;

  beforeEach(() => {
    circuitBreaker = new CircuitBreaker('test-breaker', 3, 1000);
  });

  describe('Circuit Breaker States', () => {
    test('should start in CLOSED state', () => {
      expect(circuitBreaker.state).toBe('CLOSED');
    });

    test('should execute function when CLOSED', async () => {
      const fn = jest.fn().mockResolvedValue('success');
      const result = await circuitBreaker.execute(fn);
      expect(result).toBe('success');
      expect(fn).toHaveBeenCalled();
    });

    test('should track failures', async () => {
      const fn = jest.fn().mockRejectedValue(new Error('fail'));
      for (let i = 0; i < 3; i++) {
        try {
          await circuitBreaker.execute(fn);
        } catch (e) {
          // Expected
        }
      }
      expect(circuitBreaker.failures).toBe(3);
    });

    test('should open after threshold failures', async () => {
      const fn = jest.fn().mockRejectedValue(new Error('fail'));
      for (let i = 0; i < 3; i++) {
        try {
          await circuitBreaker.execute(fn);
        } catch (e) {
          // Expected
        }
      }
      expect(circuitBreaker.state).toBe('OPEN');
    });

    test('should reject calls when OPEN', async () => {
      const fn = jest.fn().mockRejectedValue(new Error('fail'));
      for (let i = 0; i < 3; i++) {
        try {
          await circuitBreaker.execute(fn);
        } catch (e) {
          // Expected
        }
      }

      const testFn = jest.fn().mockResolvedValue('success');
      await expect(circuitBreaker.execute(testFn)).rejects.toThrow();
    });
  });

  describe('Circuit Breaker Recovery', () => {
    test('should transition to HALF_OPEN after timeout', async () => {
      const fn = jest.fn().mockRejectedValue(new Error('fail'));
      for (let i = 0; i < 3; i++) {
        try {
          await circuitBreaker.execute(fn);
        } catch (e) {
          // Expected
        }
      }

      // Wait for timeout
      await new Promise((resolve) => setTimeout(resolve, 1100));

      // Try to execute - should transition to HALF_OPEN
      const testFn = jest.fn().mockResolvedValue('success');
      try {
        await circuitBreaker.execute(testFn);
      } catch (e) {
        // May fail, but state should be HALF_OPEN
      }
      expect(circuitBreaker.state).toBe('HALF_OPEN');
    });
  });
});

