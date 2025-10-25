// Jest setup file
// Configure test environment before running tests

// Suppress console output during tests (optional)
// global.console = {
//   log: jest.fn(),
//   error: jest.fn(),
//   warn: jest.fn(),
//   info: jest.fn(),
//   debug: jest.fn(),
// };

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error';

// Increase timeout for slow tests
jest.setTimeout(10000);

// Mock timers if needed
// jest.useFakeTimers();

// Global test utilities
global.testUtils = {
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  createMockRequest: (method = 'GET', path = '/') => ({
    method,
    path,
    headers: {},
    body: {},
  }),
  createMockResponse: () => ({
    status: jest.fn().mockReturnThis(),
    json: jest.fn().mockReturnThis(),
    send: jest.fn().mockReturnThis(),
  }),
};

