#!/usr/bin/env python3
"""
Qallow Real-time Monitoring Dashboard
Live telemetry, ethics visualization, and phase progression tracking
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
import os
import threading
import time
from datetime import datetime
from collections import deque
import subprocess

app = Flask(__name__)
CORS(app)

# Telemetry buffer
telemetry_buffer = deque(maxlen=1000)
ethics_history = deque(maxlen=100)
phase_log = []

# Global state
current_state = {
    'reward': 0.0,
    'energy': 0.5,
    'risk': 0.5,
    'modules': 0,
    'step': 0,
    'running': False,
    'ethics_s': 0.99,
    'ethics_c': 1.0,
    'ethics_h': 1.0,
}

def parse_mind_output(line):
    """Parse [MIND] output lines"""
    try:
        if '[MIND]' not in line:
            return None
        
        parts = line.split()
        if 'steps=' in line:
            # Header line: [MIND] steps=50 modules=18
            for part in parts:
                if part.startswith('steps='):
                    current_state['total_steps'] = int(part.split('=')[1])
                elif part.startswith('modules='):
                    current_state['modules'] = int(part.split('=')[1])
            return None
        
        # Data line: [MIND][000] reward=0.127 energy=0.376 risk=0.339
        if 'reward=' in line:
            step = int(parts[1].strip('[]'))
            reward = float(parts[2].split('=')[1])
            energy = float(parts[3].split('=')[1])
            risk = float(parts[4].split('=')[1])
            
            current_state['step'] = step
            current_state['reward'] = reward
            current_state['energy'] = energy
            current_state['risk'] = risk
            
            telemetry_buffer.append({
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'reward': reward,
                'energy': energy,
                'risk': risk,
            })
            
            return True
    except Exception as e:
        print(f"Parse error: {e}")
    
    return None

def run_mind_process():
    """Run mind command and stream output"""
    try:
        current_state['running'] = True
        proc = subprocess.Popen(
            ['./build/qallow_unified', 'mind'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/root/Qallow'
        )
        
        for line in proc.stdout:
            parse_mind_output(line.strip())
            time.sleep(0.01)
        
        proc.wait()
        current_state['running'] = False
    except Exception as e:
        print(f"Process error: {e}")
        current_state['running'] = False

@app.route('/')
def index():
    """Serve dashboard HTML"""
    return render_template('dashboard.html')

@app.route('/api/state')
def get_state():
    """Get current system state"""
    return jsonify(current_state)

@app.route('/api/telemetry')
def get_telemetry():
    """Get telemetry history"""
    return jsonify(list(telemetry_buffer))

@app.route('/api/ethics')
def get_ethics():
    """Get ethics scores"""
    return jsonify({
        'safety': current_state.get('ethics_s', 0.99),
        'clarity': current_state.get('ethics_c', 1.0),
        'human': current_state.get('ethics_h', 1.0),
        'total': current_state.get('ethics_s', 0.99) + 
                 current_state.get('ethics_c', 1.0) + 
                 current_state.get('ethics_h', 1.0),
    })

@app.route('/api/start')
def start_mind():
    """Start mind process"""
    if not current_state['running']:
        thread = threading.Thread(target=run_mind_process, daemon=True)
        thread.start()
    return jsonify({'status': 'started'})

@app.route('/api/stop')
def stop_mind():
    """Stop mind process"""
    current_state['running'] = False
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

