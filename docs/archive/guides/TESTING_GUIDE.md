# 🧪 Qallow Server - Complete Testing Guide

## Overview

The Qallow server includes comprehensive testing infrastructure with:
- ✅ Unit tests (Jest)
- ✅ Integration tests
- ✅ API endpoint tests (Supertest)
- ✅ Error handling tests
- ✅ Manual testing commands

---

## 📦 Test Files

```
/root/Qallow/server/__tests__/
├── server.test.js          - API endpoint tests
├── errorHandler.test.js    - Error handling & health checks
├── integration.test.js     - Integration tests
├── setup.js                - Jest configuration
└── jest.config.js          - Jest settings
```

---

## 🚀 Quick Start Testing

### 1. Install Dependencies
```bash
cd /root/Qallow/server
npm install
```

### 2. Run All Tests
```bash
npm test
```

### 3. Run Tests with Coverage
```bash
npm test -- --coverage
```

### 4. Run Specific Test File
```bash
npm test -- server.test.js
npm test -- errorHandler.test.js
npm test -- integration.test.js
```

### 5. Run Tests in Watch Mode
```bash
npm test -- --watch
```

---

## 📊 Test Coverage

### Server Tests (`server.test.js`)
Tests all API endpoints:
- ✅ `GET /api/health` - Server health status
- ✅ `GET /api/quantum/status` - Quantum framework status
- ✅ `GET /api/system/metrics` - System metrics
- ✅ `POST /api/quantum/run-grover` - Grover's algorithm
- ✅ `POST /api/quantum/run-bell-state` - Bell state
- ✅ `POST /api/quantum/run-deutsch` - Deutsch algorithm
- ✅ `POST /api/quantum/run-all` - All algorithms

**Test Cases:**
- Health check endpoints return correct status
- Quantum algorithms execute successfully
- Response format consistency
- Timestamp validation

### Error Handler Tests (`errorHandler.test.js`)
Tests error handling system:
- ✅ Logging methods (info, error, warn, debug)
- ✅ Error buffer management
- ✅ Log entry format validation
- ✅ Health check registration
- ✅ Health check execution
- ✅ Circuit breaker states
- ✅ Circuit breaker failure tracking
- ✅ Circuit breaker recovery

**Test Cases:**
- Structured logging
- Error buffer limits
- Health check results
- Circuit breaker state transitions
- Timeout handling

### Integration Tests (`integration.test.js`)
Tests system integration:
- ✅ Request-response cycles
- ✅ Parameter validation
- ✅ Response consistency
- ✅ Performance metrics
- ✅ Concurrent requests
- ✅ Error recovery

**Test Cases:**
- Successful requests
- Error handling
- Parameter validation
- Response structure
- Concurrent request handling
- Recovery from errors

---

## 🧪 Manual Testing

### 1. Start the Server
```bash
bash /root/Qallow/QUICK_START_SERVER.sh
```

### 2. Test Health Endpoint
```bash
curl http://localhost:5000/api/health
```

**Expected Response:**
```json
{
  "success": true,
  "status": "healthy",
  "timestamp": "2025-10-25T12:00:00.000Z"
}
```

### 3. Test Quantum Status
```bash
curl http://localhost:5000/api/quantum/status
```

**Expected Response:**
```json
{
  "success": true,
  "framework": "Google Cirq",
  "simulator": "QSim",
  "status": "ready"
}
```

### 4. Test System Metrics
```bash
curl http://localhost:5000/api/system/metrics
```

**Expected Response:**
```json
{
  "success": true,
  "memory": { "used": 100, "total": 1000 },
  "cpu": { "usage": 25 },
  "uptime": 3600
}
```

### 5. Run Grover's Algorithm
```bash
curl -X POST http://localhost:5000/api/quantum/run-grover \
  -H "Content-Type: application/json" \
  -d '{"num_qubits": 3, "target_state": 5}'
```

### 6. Run Bell State
```bash
curl -X POST http://localhost:5000/api/quantum/run-bell-state
```

### 7. Run Deutsch Algorithm
```bash
curl -X POST http://localhost:5000/api/quantum/run-deutsch
```

### 8. Run All Algorithms
```bash
curl -X POST http://localhost:5000/api/quantum/run-all
```

---

## 📈 Performance Testing

### Load Testing with Apache Bench
```bash
# Install Apache Bench (if not installed)
sudo apt-get install apache2-utils

# Test health endpoint with 1000 requests, 10 concurrent
ab -n 1000 -c 10 http://localhost:5000/api/health

# Test with POST request
ab -n 100 -c 5 -p data.json -T application/json \
  http://localhost:5000/api/quantum/run-grover
```

### Load Testing with wrk
```bash
# Install wrk
sudo apt-get install wrk

# Test with 4 threads, 100 connections, 30 seconds
wrk -t4 -c100 -d30s http://localhost:5000/api/health
```

---

## 🔍 Debugging Tests

### Run Single Test
```bash
npm test -- --testNamePattern="should return healthy status"
```

### Run with Verbose Output
```bash
npm test -- --verbose
```

### Debug Mode
```bash
node --inspect-brk node_modules/.bin/jest --runInBand
```

Then open `chrome://inspect` in Chrome DevTools.

---

## 📋 Test Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code coverage > 70%
- [ ] No console errors
- [ ] API endpoints respond correctly
- [ ] Error handling works
- [ ] Health checks pass
- [ ] Circuit breaker functions
- [ ] Concurrent requests handled
- [ ] Performance acceptable

---

## 🐛 Troubleshooting

### Tests Fail with "Cannot find module"
```bash
cd /root/Qallow/server
npm install
```

### Port Already in Use
```bash
# Kill process on port 5000
lsof -i :5000
kill -9 <PID>
```

### Tests Timeout
```bash
# Increase timeout in jest.config.js
testTimeout: 20000
```

### Memory Issues
```bash
# Run tests with more memory
NODE_OPTIONS=--max-old-space-size=4096 npm test
```

---

## 📊 Coverage Report

After running tests with coverage:
```bash
npm test -- --coverage
```

View HTML report:
```bash
open coverage/lcov-report/index.html
```

---

## 🚀 CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '16'
      - run: npm install
      - run: npm test -- --coverage
      - uses: codecov/codecov-action@v2
```

---

## 📚 Additional Resources

- [Jest Documentation](https://jestjs.io/)
- [Supertest Documentation](https://github.com/visionmedia/supertest)
- [Express Testing Guide](https://expressjs.com/en/guide/testing.html)

---

## ✅ Status

- **Test Framework**: Jest 29.7.0
- **HTTP Testing**: Supertest 6.3.3
- **Coverage Threshold**: 70%
- **Test Timeout**: 10 seconds
- **Total Test Cases**: 40+

**Status**: ✅ READY FOR TESTING


