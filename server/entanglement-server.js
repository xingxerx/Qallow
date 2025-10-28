/**
 * Quantum Entanglement Server
 * 
 * Manages real-time synchronization between Web App and Native App
 * using WebSocket connections and quantum entanglement principles
 */

const WebSocket = require('ws');
const http = require('http');

class EntanglementServer {
  constructor(port = 3002) {
    this.port = port;
    this.server = null;
    this.wss = null;
    this.clients = new Map(); // Map of app -> WebSocket
    this.sharedState = this.initializeState();
    this.messageLog = [];
    this.maxLogSize = 1000;
  }

  /**
   * Initialize shared quantum state
   */
  initializeState() {
    return {
      phase: 1,
      buildType: 'CPU',
      vmRunning: false,
      selectedPhase: 13,
      metrics: {
        fidelity: 0.981,
        energy: 0,
        risk: 0,
        reward: 0,
      },
      timestamp: Date.now(),
      lastUpdatedBy: null,
    };
  }

  /**
   * Start entanglement server
   */
  start() {
    return new Promise((resolve, reject) => {
      try {
        this.server = http.createServer();
        this.wss = new WebSocket.Server({ server: this.server, path: '/entanglement' });

        this.wss.on('connection', (ws, req) => {
          this.handleConnection(ws, req);
        });

        this.server.listen(this.port, () => {
          console.log(`[Entanglement Server] Started on port ${this.port}`);
          resolve();
        });
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Stop entanglement server
   */
  stop() {
    return new Promise((resolve) => {
      if (this.wss) {
        this.wss.close();
      }
      if (this.server) {
        this.server.close(() => {
          console.log('[Entanglement Server] Stopped');
          resolve();
        });
      } else {
        resolve();
      }
    });
  }

  /**
   * Handle new connection
   */
  handleConnection(ws, req) {
    const clientId = this.generateClientId();
    console.log(`[Entanglement] New connection: ${clientId}`);

    this.clients.set(clientId, { ws, appId: null, connected: true });

    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        this.handleMessage(clientId, message);
      } catch (error) {
        console.error('[Entanglement] Message parse error:', error);
      }
    });

    ws.on('close', () => {
      console.log(`[Entanglement] Connection closed: ${clientId}`);
      this.clients.delete(clientId);
    });

    ws.on('error', (error) => {
      console.error(`[Entanglement] Connection error (${clientId}):`, error);
    });

    // Send initial state
    this.sendToClient(clientId, {
      type: 'SYNC',
      payload: this.sharedState,
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
    });
  }

  /**
   * Handle incoming message
   */
  handleMessage(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Log message
    this.logMessage(message);

    switch (message.type) {
      case 'STATE_UPDATE':
        this.handleStateUpdate(clientId, message);
        break;

      case 'ACTION':
        this.handleAction(clientId, message);
        break;

      case 'SYNC':
        this.handleSync(clientId, message);
        break;

      case 'HEARTBEAT':
        this.handleHeartbeat(clientId, message);
        break;

      case 'ACK':
        // Acknowledgment received
        break;
    }
  }

  /**
   * Handle state update (entangle to other app)
   */
  handleStateUpdate(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Update shared state
    this.sharedState = {
      ...this.sharedState,
      ...message.payload,
      timestamp: Date.now(),
      lastUpdatedBy: client.appId,
    };

    // Broadcast to other app
    this.broadcastToOtherApp(clientId, {
      type: 'STATE_UPDATE',
      payload: message.payload,
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
      sourceApp: client.appId,
    });

    // Send acknowledgment
    this.sendToClient(clientId, {
      type: 'ACK',
      payload: { messageId: message.messageId },
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
    });

    console.log(`[Entanglement] State entangled from ${client.appId}:`, message.payload);
  }

  /**
   * Handle action
   */
  handleAction(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Broadcast action to other app
    this.broadcastToOtherApp(clientId, {
      type: 'ACTION',
      payload: message.payload,
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
      sourceApp: client.appId,
    });

    // Send acknowledgment
    this.sendToClient(clientId, {
      type: 'ACK',
      payload: { messageId: message.messageId },
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
    });
  }

  /**
   * Handle sync request
   */
  handleSync(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Send current shared state
    this.sendToClient(clientId, {
      type: 'SYNC',
      payload: this.sharedState,
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
    });
  }

  /**
   * Handle heartbeat
   */
  handleHeartbeat(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Update app ID if not set
    if (!client.appId && message.payload.appId) {
      client.appId = message.payload.appId;
      console.log(`[Entanglement] App identified: ${client.appId}`);
    }

    // Send heartbeat response
    this.sendToClient(clientId, {
      type: 'HEARTBEAT',
      payload: { timestamp: Date.now() },
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
    });
  }

  /**
   * Send message to specific client
   */
  sendToClient(clientId, message) {
    const client = this.clients.get(clientId);
    if (client && client.ws.readyState === WebSocket.OPEN) {
      client.ws.send(JSON.stringify(message));
    }
  }

  /**
   * Broadcast to other app
   */
  broadcastToOtherApp(sourceClientId, message) {
    const sourceClient = this.clients.get(sourceClientId);
    if (!sourceClient) return;

    for (const [clientId, client] of this.clients) {
      if (clientId !== sourceClientId && client.appId !== sourceClient.appId) {
        this.sendToClient(clientId, message);
      }
    }
  }

  /**
   * Log message
   */
  logMessage(message) {
    this.messageLog.push({
      ...message,
      receivedAt: Date.now(),
    });

    if (this.messageLog.length > this.maxLogSize) {
      this.messageLog.shift();
    }
  }

  /**
   * Generate client ID
   */
  generateClientId() {
    return `client-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate message ID
   */
  generateMessageId() {
    return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get statistics
   */
  getStatistics() {
    return {
      connectedClients: this.clients.size,
      sharedState: this.sharedState,
      messageLogSize: this.messageLog.length,
      clients: Array.from(this.clients.values()).map(c => ({
        appId: c.appId,
        connected: c.connected,
      })),
    };
  }

  /**
   * Get message log
   */
  getMessageLog() {
    return this.messageLog;
  }
}

module.exports = EntanglementServer;

