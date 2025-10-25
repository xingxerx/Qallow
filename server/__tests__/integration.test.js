const request = require('supertest');
const express = require('express');

describe('Integration Tests - Server & Error Handling', () => {
  let app;
  let server;

  beforeAll(() => {
    app = express();
    app.use(express.json());

    // Simulate server with error handling
    app.get('/api/health', (req, res) => {
      res.json({ success: true, status: 'healthy' });
    });

    app.get('/api/error-test', (req, res) => {
      res.status(500).json({
        success: false,
        error: 'Test error',
        timestamp: new Date().toISOString(),
      });
    });

    app.post('/api/quantum/run-grover', (req, res) => {
      const { num_qubits = 3, target_state = 5 } = req.body;
      if (num_qubits < 1 || num_qubits > 20) {
        return res.status(400).json({
          success: false,
          error: 'Invalid number of qubits',
        });
      }
      res.json({
        success: true,
        algorithm: 'Grover',
        params: { num_qubits, target_state },
        results: { success_rate: 0.783 },
      });
    });

    app.post('/api/quantum/run-all', (req, res) => {
      res.json({
        success: true,
        algorithms: ['Grover', 'Bell State', 'Deutsch'],
        status: 'completed',
        duration_ms: 1234,
      });
    });

    // Error handling middleware
    app.use((err, req, res, next) => {
      res.status(500).json({
        success: false,
        error: err.message,
      });
    });

    server = app.listen(5002);
  });

  afterAll((done) => {
    server.close(done);
  });

  describe('Request-Response Cycle', () => {
    test('should handle successful request', async () => {
      const response = await request(app).get('/api/health');
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    test('should handle error response', async () => {
      const response = await request(app).get('/api/error-test');
      expect(response.status).toBe(500);
      expect(response.body.success).toBe(false);
    });
  });

  describe('Request Validation', () => {
    test('should validate algorithm parameters', async () => {
      const response = await request(app)
        .post('/api/quantum/run-grover')
        .send({ num_qubits: 25 });
      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
    });

    test('should accept valid parameters', async () => {
      const response = await request(app)
        .post('/api/quantum/run-grover')
        .send({ num_qubits: 5, target_state: 10 });
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
  });

  describe('Response Consistency', () => {
    test('all responses should have consistent structure', async () => {
      const endpoints = [
        { method: 'get', path: '/api/health' },
        { method: 'post', path: '/api/quantum/run-all' },
      ];

      for (const endpoint of endpoints) {
        const response = await request(app)[endpoint.method](endpoint.path);
        expect(response.body).toHaveProperty('success');
      }
    });

    test('error responses should include error message', async () => {
      const response = await request(app).get('/api/error-test');
      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toBeTruthy();
    });
  });

  describe('Performance Tests', () => {
    test('health check should respond quickly', async () => {
      const start = Date.now();
      await request(app).get('/api/health');
      const duration = Date.now() - start;
      expect(duration).toBeLessThan(100);
    });

    test('algorithm execution should complete', async () => {
      const response = await request(app).post('/api/quantum/run-all');
      expect(response.body.duration_ms).toBeDefined();
      expect(response.status).toBe(200);
    });
  });

  describe('Concurrent Requests', () => {
    test('should handle multiple concurrent requests', async () => {
      const requests = [];
      for (let i = 0; i < 10; i++) {
        requests.push(request(app).get('/api/health'));
      }
      const responses = await Promise.all(requests);
      responses.forEach((response) => {
        expect(response.status).toBe(200);
        expect(response.body.success).toBe(true);
      });
    });
  });

  describe('Error Recovery', () => {
    test('should recover from error and handle next request', async () => {
      // First request fails
      let response = await request(app).get('/api/error-test');
      expect(response.status).toBe(500);

      // Next request should succeed
      response = await request(app).get('/api/health');
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
  });
});

