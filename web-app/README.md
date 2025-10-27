# Qallow Web Application

Modern React web interface for the Qallow Unified Quantum-AGI System.

## Features

- 🎨 **Beautiful Dark Theme** - Cyberpunk-inspired UI with cyan/green accents
- 📊 **Dashboard** - Real-time system metrics and status
- 💻 **Terminal** - Live output streaming from Qallow VM
- 📈 **Metrics** - Detailed performance analytics
- 🔍 **Audit Log** - Event tracking and filtering
- ⚙️ **Control Panel** - VM management and configuration

## Quick Start

### Prerequisites

- Node.js 16+ and npm 8+
- Qallow backend server running on port 3001

### Installation

```bash
cd /root/Qallow/web-app
npm install
```

### Development

```bash
npm start
```

Opens http://localhost:3000 in your browser.

### Production Build

```bash
npm run build
```

Creates optimized production build in `build/` directory.

## Architecture

```
web-app/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Dashboard.js
│   │   ├── Terminal.js
│   │   ├── Metrics.js
│   │   ├── AuditLog.js
│   │   └── ControlPanel.js
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── index.css
└── package.json
```

## API Integration

The app connects to the backend API at `http://localhost:3001/api`:

- `GET /api/status` - Get current system status
- `POST /api/vm/start` - Start the VM
- `POST /api/vm/stop` - Stop the VM
- `GET /api/metrics` - Get system metrics
- `GET /api/logs` - Get audit logs

## Environment Variables

Create `.env` file in the project root:

```
REACT_APP_API_URL=http://localhost:3001/api
```

## Styling

The app uses a custom CSS theme with:
- Primary color: `#00d4ff` (Cyan)
- Accent color: `#00ff64` (Green)
- Background: `#0a0e27` (Dark Blue)
- Secondary: `#1a1f3a` (Navy)

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## License

MIT
