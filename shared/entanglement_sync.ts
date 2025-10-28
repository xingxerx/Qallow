/**
 * Quantum Entanglement Synchronization Protocol
 * 
 * WebSocket-based real-time synchronization between Web App and Native App
 * Implements quantum entanglement principles for instant state propagation
 */

import { EventEmitter } from 'events';
import { QuantumState, EntanglementEvent } from './quantum_entanglement';

/**
 * Sync Message Format
 */
export interface SyncMessage {
  type: 'SYNC' | 'ACK' | 'HEARTBEAT' | 'STATE_UPDATE' | 'ACTION';
  payload: any;
  timestamp: number;
  messageId: string;
  sourceApp: 'web' | 'native';
}

/**
 * Quantum Entanglement Sync Manager
 * Handles WebSocket communication and state synchronization
 */
export class EntanglementSyncManager extends EventEmitter {
  private ws: WebSocket | null = null;
  private appId: 'web' | 'native';
  private serverUrl: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private messageQueue: SyncMessage[] = [];
  private pendingAcks = new Map<string, NodeJS.Timeout>();
  private ackTimeout = 5000;

  constructor(appId: 'web' | 'native', serverUrl: string = 'ws://localhost:3001') {
    super();
    this.appId = appId;
    this.serverUrl = serverUrl;
  }

  /**
   * Connect to sync server
   */
  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(`${this.serverUrl}/entanglement`);

        this.ws.onopen = () => {
          console.log(`[${this.appId}] Connected to entanglement sync server`);
          this.reconnectAttempts = 0;
          this.flushMessageQueue();
          this.emit('connected');
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(JSON.parse(event.data));
        };

        this.ws.onerror = (error) => {
          console.error(`[${this.appId}] WebSocket error:`, error);
          this.emit('error', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log(`[${this.appId}] Disconnected from entanglement sync server`);
          this.emit('disconnected');
          this.attemptReconnect();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from sync server
   */
  public disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Send state update (entangle)
   */
  public sendStateUpdate(state: Partial<QuantumState>): void {
    const message: SyncMessage = {
      type: 'STATE_UPDATE',
      payload: state,
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
      sourceApp: this.appId,
    };

    this.sendMessage(message);
  }

  /**
   * Send action
   */
  public sendAction(action: string, params: any): void {
    const message: SyncMessage = {
      type: 'ACTION',
      payload: { action, params },
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
      sourceApp: this.appId,
    };

    this.sendMessage(message);
  }

  /**
   * Request full sync
   */
  public requestFullSync(): Promise<QuantumState> {
    return new Promise((resolve) => {
      const message: SyncMessage = {
        type: 'SYNC',
        payload: {},
        timestamp: Date.now(),
        messageId: this.generateMessageId(),
        sourceApp: this.appId,
      };

      const timeout = setTimeout(() => {
        this.pendingAcks.delete(message.messageId);
        resolve({} as QuantumState);
      }, this.ackTimeout);

      this.pendingAcks.set(message.messageId, timeout);
      this.sendMessage(message);
    });
  }

  /**
   * Send message
   */
  private sendMessage(message: SyncMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      this.messageQueue.push(message);
    }
  }

  /**
   * Handle incoming message
   */
  private handleMessage(message: SyncMessage): void {
    switch (message.type) {
      case 'STATE_UPDATE':
        this.emit('state-update', message.payload);
        this.sendAck(message.messageId);
        break;

      case 'ACTION':
        this.emit('action', message.payload);
        this.sendAck(message.messageId);
        break;

      case 'SYNC':
        this.emit('sync-request', message.payload);
        this.sendAck(message.messageId);
        break;

      case 'ACK':
        this.handleAck(message.messageId);
        break;

      case 'HEARTBEAT':
        this.sendHeartbeat();
        break;
    }
  }

  /**
   * Send acknowledgment
   */
  private sendAck(messageId: string): void {
    const ackMessage: SyncMessage = {
      type: 'ACK',
      payload: { messageId },
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
      sourceApp: this.appId,
    };

    this.sendMessage(ackMessage);
  }

  /**
   * Handle acknowledgment
   */
  private handleAck(messageId: string): void {
    const timeout = this.pendingAcks.get(messageId);
    if (timeout) {
      clearTimeout(timeout);
      this.pendingAcks.delete(messageId);
    }
  }

  /**
   * Send heartbeat
   */
  private sendHeartbeat(): void {
    const message: SyncMessage = {
      type: 'HEARTBEAT',
      payload: { appId: this.appId },
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
      sourceApp: this.appId,
    };

    this.sendMessage(message);
  }

  /**
   * Flush message queue
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.sendMessage(message);
      }
    }
  }

  /**
   * Attempt reconnect
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      console.log(`[${this.appId}] Reconnecting in ${delay}ms...`);

      setTimeout(() => {
        this.connect().catch((error) => {
          console.error(`[${this.appId}] Reconnect failed:`, error);
        });
      }, delay);
    }
  }

  /**
   * Generate unique message ID
   */
  private generateMessageId(): string {
    return `${this.appId}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get connection status
   */
  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get statistics
   */
  public getStatistics() {
    return {
      connected: this.isConnected(),
      appId: this.appId,
      messageQueueSize: this.messageQueue.length,
      pendingAcks: this.pendingAcks.size,
      reconnectAttempts: this.reconnectAttempts,
    };
  }
}

/**
 * Create sync manager instance
 */
export function createSyncManager(appId: 'web' | 'native', serverUrl?: string): EntanglementSyncManager {
  return new EntanglementSyncManager(appId, serverUrl);
}

