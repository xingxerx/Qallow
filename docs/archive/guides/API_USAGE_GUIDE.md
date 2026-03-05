# Real-Time Monitoring API Usage Guide

**System:** Qallow v2.0 with Live Update Capability  
**Date:** 2025-10-28  

---

## 🚀 Quick Start

### 1. Start the API Server
```bash
cd /root/Qallow/server
npm start
```

The API will be available at `http://localhost:5050`

### 2. Check System Status
```bash
curl http://localhost:5050/api/status
```

### 3. Monitor Health
```bash
curl http://localhost:5050/api/health
```

---

## 📊 API Endpoints Reference

### System Status
```bash
GET /api/status
```
**Returns:** Current system status, phase info, metrics, logs

**Example Response:**
```json
{
  "vm_running": true,
  "continuous_mode": true,
  "current_phase": 5,
  "cycle_count": 17,
  "metrics": {
    "coherence": 0.85,
    "fidelity": 0.92,
    "stability": 0.88
  }
}
```

---

### Health Monitoring
```bash
GET /api/health
```
**Returns:** Current health status and health summary

**Example Response:**
```json
{
  "current_health": {
    "status": "HEALTHY",
    "phase": 5,
    "issues": []
  },
  "health_summary": {
    "totalChecks": 100,
    "healthy": 95,
    "warnings": 5,
    "errors": 0,
    "healthPercentage": "95.00"
  }
}
```

---

### Performance Analytics
```bash
GET /api/performance
```
**Returns:** Performance metrics and trend analysis

**Example Response:**
```json
{
  "performance_metrics": {
    "phaseTimings": {
      "1": [100, 105, 102],
      "2": [150, 155, 152]
    },
    "cycleTimings": [450, 470, 459]
  },
  "averages": {
    "phase_time_ms": 127.5,
    "cycle_time_ms": 459.7
  }
}
```

---

### Optimization Recommendations
```bash
GET /api/optimizations
```
**Returns:** Optimization suggestions based on current metrics

**Example Response:**
```json
{
  "recommendations": [
    {
      "type": "PERFORMANCE",
      "phase": 2,
      "suggestion": "Phase 2 is slow. Consider increasing ticks.",
      "severity": "HIGH"
    },
    {
      "type": "EFFICIENCY",
      "suggestion": "High energy consumption. Consider CPU build.",
      "severity": "MEDIUM"
    }
  ],
  "count": 2
}
```

---

### Log Improvement
```bash
POST /api/improvements/log
```
**Body:**
```json
{
  "category": "Performance",
  "title": "Optimized Phase 5",
  "description": "Reduced phase 5 execution time by 15%",
  "impact": "HIGH",
  "files": ["phases/phase_5.c", "backend/cpu/phase5_core.c"]
}
```

**Returns:** Logged improvement with ID

**Example Response:**
```json
{
  "success": true,
  "improvement": {
    "id": "imp_1761625351759_abc123def",
    "timestamp": "2025-10-28T15:30:00.000Z",
    "category": "Performance",
    "title": "Optimized Phase 5",
    "status": "IMPLEMENTED"
  }
}
```

---

### Get Improvement Report
```bash
GET /api/improvements/report
```
**Returns:** Full improvement report with categorization

**Example Response:**
```json
{
  "report": {
    "timestamp": "2025-10-28T15:30:00.000Z",
    "totalImprovements": 10,
    "byCategory": {
      "Performance": [/* improvements */],
      "Monitoring": [/* improvements */],
      "Analytics": [/* improvements */]
    },
    "byImpact": {
      "HIGH": [/* improvements */],
      "MEDIUM": [/* improvements */]
    }
  }
}
```

---

### Get Improvement Summary
```bash
GET /api/improvements/summary
```
**Returns:** Quick summary of improvements

**Example Response:**
```json
{
  "summary": {
    "timestamp": "2025-10-28T15:30:00.000Z",
    "uptime": 3600000,
    "totalImprovements": 10,
    "categories": ["Performance", "Monitoring", "Analytics"],
    "impacts": ["HIGH", "MEDIUM"],
    "recentImprovements": [
      {
        "id": "imp_1761625351759_abc123def",
        "title": "Optimized Phase 5",
        "category": "Performance"
      }
    ]
  }
}
```

---

## 🔍 Usage Examples

### Monitor System Health Every 5 Seconds
```bash
watch -n 5 'curl -s http://localhost:5050/api/health | jq .health_summary'
```

### Get Performance Trends
```bash
curl -s http://localhost:5050/api/performance | jq '.averages'
```

### Log a Performance Improvement
```bash
curl -X POST http://localhost:5050/api/improvements/log \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Performance",
    "title": "Optimized Phase 5",
    "description": "Reduced execution time by 15%",
    "impact": "HIGH",
    "files": ["phases/phase_5.c"]
  }'
```

### Get All Recommendations
```bash
curl -s http://localhost:5050/api/optimizations | jq '.recommendations'
```

### Export Metrics
```bash
curl -s http://localhost:5050/api/metrics/export | jq '.'
```

---

## 📈 Monitoring Dashboard

Create a simple monitoring dashboard:

```bash
#!/bin/bash
while true; do
  clear
  echo "╔════════════════════════════════════════════════════════════════╗"
  echo "║  QALLOW REAL-TIME MONITORING DASHBOARD                        ║"
  echo "╚════════════════════════════════════════════════════════════════╝"
  echo ""
  
  echo "📊 HEALTH STATUS:"
  curl -s http://localhost:5050/api/health | jq '.health_summary | {healthy, warnings, errors, healthPercentage}'
  echo ""
  
  echo "⚡ PERFORMANCE:"
  curl -s http://localhost:5050/api/performance | jq '.averages'
  echo ""
  
  echo "💡 RECOMMENDATIONS:"
  curl -s http://localhost:5050/api/optimizations | jq '.count'
  echo ""
  
  echo "📝 IMPROVEMENTS:"
  curl -s http://localhost:5050/api/improvements/summary | jq '.summary | {totalImprovements, uptime}'
  echo ""
  
  sleep 5
done
```

---

## 🔧 Advanced Usage

### Filter Recommendations by Severity
```bash
curl -s http://localhost:5050/api/optimizations | \
  jq '.recommendations[] | select(.severity == "HIGH")'
```

### Get Recent Improvements
```bash
curl -s http://localhost:5050/api/improvements/report | \
  jq '.report.improvements[-5:]'
```

### Monitor Phase Performance
```bash
curl -s http://localhost:5050/api/performance | \
  jq '.performance_metrics.phaseTimings'
```

### Check System Uptime
```bash
curl -s http://localhost:5050/api/improvements/summary | \
  jq '.summary.uptime / 1000 / 60 | "Uptime: \(.) minutes"'
```

---

## 📊 Metrics Explained

### Coherence
- **Range:** 0.0 - 1.0
- **Meaning:** Quantum coherence level
- **Healthy:** > 0.7
- **Critical:** < 0.5

### Fidelity
- **Range:** 0.0 - 1.0
- **Meaning:** Quantum gate fidelity
- **Healthy:** > 0.8
- **Critical:** < 0.6

### Stability
- **Range:** 0.0 - 1.0
- **Meaning:** System stability
- **Healthy:** > 0.8
- **Critical:** < 0.5

### Energy
- **Range:** 0.0 - 1.0
- **Meaning:** Energy consumption
- **Healthy:** < 0.7
- **Critical:** > 0.9

---

## 🚨 Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Coherence | < 0.75 | < 0.5 |
| Fidelity | < 0.85 | < 0.6 |
| Energy | > 0.85 | > 0.95 |
| Phase Time | > 100s | > 120s |
| Cycle Time | > 2000s | > 2400s |

---

## 💾 Data Export

### Export Metrics
```bash
curl -s http://localhost:5050/api/metrics/export > metrics_backup.json
```

### Export Improvements
```bash
curl -s http://localhost:5050/api/improvements/report > improvements_report.json
```

### Export Configuration
```bash
curl -s http://localhost:5050/api/config/save > config_backup.json
```

---

## 🔐 Security Notes

- All endpoints are currently open (no authentication)
- For production, add authentication middleware
- Sensitive data should be encrypted
- Consider rate limiting for public deployments

---

## 📞 Troubleshooting

### API Not Responding
```bash
# Check if server is running
ps aux | grep "node.*api-web"

# Check port 5050
netstat -tlnp | grep 5050

# Restart server
cd /root/Qallow/server && npm start
```

### Metrics Not Updating
```bash
# Check if VM is running
curl http://localhost:5050/api/status | jq '.vm_running'

# Check terminal output
curl http://localhost:5050/api/status | jq '.terminal_output[-5:]'
```

### High Latency
```bash
# Check system load
curl http://localhost:5050/api/performance | jq '.averages'

# Check recommendations
curl http://localhost:5050/api/optimizations
```

---

## 📚 Related Documentation

- `REALTIME_IMPROVEMENTS.md` - Detailed improvement log
- `LIVE_UPDATE_SUMMARY.md` - Integration summary
- `server/monitoring.js` - Monitoring system source
- `server/improvement-tracker.js` - Improvement tracker source
- `server/api-web.js` - API implementation

---

**Last Updated:** 2025-10-28  
**Version:** 1.0  
**Status:** ✅ PRODUCTION READY
