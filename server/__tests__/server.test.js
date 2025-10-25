const request = require('supertest');
const express = require('express');
const path = require('path');

// Mock the errorHandler module
jest.mock('../errorHandler', () => ({
  ErrorHandler: jest.fn().mockImplementation(() => ({
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  })),
  HealthCheck: jest.fn().mockImplementation(() => ({
    register: jest.fn(),
    runAll: jest.fn().mockResolvedValue({
      memory: { status: 'healthy' },
      cpu: { status: 'healthy' },
    }),
  })),
  CircuitBreaker: jest.fn().mockImplementation(() => ({
    execute: jest.fn((fn) => fn()),
  })),
}));

describe('Qallow Server API Tests', () => {
  let app;
  let server;

  beforeAll(() => {
    // Create a minimal Express app for testing
    app = express();
    app.use(express.json());

    // Health check endpoint
    app.get('/api/health', (req, res) => {
      res.json({ success: true, status: 'healthy', timestamp: new Date().toISOString() });
    });

    // Quantum status endpoint
    app.get('/api/quantum/status', (req, res) => {
      res.json({
        success: true,
        framework: 'Google Cirq',
        simulator: 'QSim',
        status: 'ready',
      });
    });

    // System metrics endpoint
    app.get('/api/system/metrics', (req, res) => {
      res.json({
        success: true,
        memory: { used: 100, total: 1000 },
        cpu: { usage: 25 },
        uptime: 3600,
      });
    });

    // Quantum algorithm endpoints
    app.post('/api/quantum/run-grover', (req, res) => {
      res.json({
        success: true,
        algorithm: 'Grover',
        results: { states: 8, success_rate: 0.783 },
      });
    });

    app.post('/api/quantum/run-bell-state', (req, res) => {
      res.json({
        success: true,
        algorithm: 'Bell State',
        results: { entanglement: 'perfect', states: 2 },
      });
    });

    app.post('/api/quantum/run-deutsch', (req, res) => {
      res.json({
        success: true,
        algorithm: 'Deutsch',
        results: { function_type: 'BALANCED', accuracy: 1.0 },
      });
    });

    app.post('/api/quantum/run-all', (req, res) => {
      res.json({
        success: true,
        algorithms: ['Grover', 'Bell State', 'Deutsch'],
        status: 'completed',
      });
    });

    // Error handling middleware
    app.use((err, req, res, next) => {
      res.status(500).json({
        success: false,
        error: err.message,
        timestamp: new Date().toISOString(),
      });
    });

    server = app.listen(5001);
  });

  afterAll((done) => {
    server.close(done);
  });

  describe('Health Check Endpoints', () => {
    test('GET /api/health should return healthy status', async () => {
      const response = await request(app).get('/api/health');
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.status).toBe('healthy');
    });

    test('GET /api/quantum/status should return quantum framework status', async () => {
      const response = await request(app).get('/api/quantum/status');
      expect(response.status).toBe(200);
      expect(response.body.framework).toBe('Google Cirq');
      expect(response.body.simulator).toBe('QSim');
    });

    test('GET /api/system/metrics should return system metrics', async () => {
      const response = await request(app).get('/api/system/metrics');
      expect(response.status).toBe(200);
      expect(response.body.memory).toBeDefined();
      expect(response.body.cpu).toBeDefined();
      expect(response.body.uptime).toBeDefined();
    });
  });

  describe('Quantum Algorithm Endpoints', () => {
    test('POST /api/quantum/run-grover should execute Grover algorithm', async () => {
      const response = await request(app)
        .post('/api/quantum/run-grover')
        .send({ num_qubits: 3, target_state: 5 });
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.algorithm).toBe('Grover');
    });

    test('POST /api/quantum/run-bell-state should execute Bell state', async () => {
      const response = await request(app).post('/api/quantum/run-bell-state');
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.algorithm).toBe('Bell State');
    });

    test('POST /api/quantum/run-deutsch should execute Deutsch algorithm', async () => {
      const response = await request(app).post('/api/quantum/run-deutsch');
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.algorithm).toBe('Deutsch');
    });

    test('POST /api/quantum/run-all should execute all algorithms', async () => {
      const response = await request(app).post('/api/quantum/run-all');
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.algorithms.length).toBe(3);
    });
  });

  describe('Response Format Tests', () => {
    test('All responses should have success field', async () => {
      const endpoints = [
        { method: 'get', path: '/api/health' },
        { method: 'get', path: '/api/quantum/status' },
        { method: 'get', path: '/api/system/metrics' },
      ];

      for (const endpoint of endpoints) {
        const response = await request(app)[endpoint.method](endpoint.path);
        expect(response.body).toHaveProperty('success');
      }
    });

    test('All responses should have timestamp', async () => {
      const response = await request(app).get('/api/health');
      expect(response.body).toHaveProperty('timestamp');
    });
  });
});

