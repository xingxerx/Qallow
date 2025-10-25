#!/usr/bin/env python3
"""
Qallow Real-time Monitoring Dashboard
Live telemetry, ethics visualization, and phase progression tracking
Enhanced with phase metrics, CSV telemetry integration, and audit logs
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
import csv
import glob
import shlex

app = Flask(__name__)
CORS(app)

# Telemetry buffer
telemetry_buffer = deque(maxlen=1000)
ethics_history = deque(maxlen=100)
phase_log = []
audit_log = []

# Process coordination
process_lock = threading.Lock()
mind_process = None
mind_thread = None
stop_event = threading.Event()


def _default_state():
    return {
        'reward': 0.0,
        'energy': 0.5,
        'risk': 0.5,
        'modules': 0,
        'step': 0,
        'running': False,
        'ethics_s': 0.99,
        'ethics_c': 1.0,
        'ethics_h': 1.0,
        'current_phase': 'idle',
        'phase_progress': 0.0,
        'fidelity': 0.0,
        'coherence': 0.0,
    }


# Global state
current_state = _default_state()


def reset_state():
    current_state.clear()
    current_state.update(_default_state())


def get_mind_command():
    raw = os.environ.get('QALLOW_MIND_COMMAND', './build/qallow_unified mind')
    if isinstance(raw, str):
        raw = raw.strip()
    if not raw:
        return []
    return shlex.split(raw)


def _terminate_process(timeout=5):
    global mind_process
    with process_lock:
        if mind_process and mind_process.poll() is None:
            mind_process.terminate()
            try:
                mind_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                mind_process.kill()
                mind_process.wait(timeout=timeout)

def load_csv_telemetry(filepath):
    """Load telemetry from CSV file (phase logs)"""
    try:
        if not os.path.exists(filepath):
            return []

        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    data.append({
                        'timestamp': datetime.now().isoformat(),
                        'tick': int(row.get('tick', 0)),
                        'coherence': float(row.get('coherence', 0.0)),
                        'fidelity': float(row.get('fidelity', 0.0)),
                        'phase_drift': float(row.get('phase_drift', 0.0)),
                        'energy': float(row.get('energy', 0.0)),
                    })
                except (ValueError, KeyError):
                    continue
        return data
    except Exception as e:
        print(f"CSV load error: {e}")
        return []

def load_json_metrics(filepath):
    """Load metrics from JSON file"""
    try:
        if not os.path.exists(filepath):
            return {}
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON load error: {e}")
        return {}

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
    global mind_process
    global mind_thread
    try:
        command = get_mind_command()
        if not command:
            raise RuntimeError('QALLOW_MIND_COMMAND is empty')

        with process_lock:
            mind_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd='/root/Qallow'
            )

        current_state['running'] = True

        if mind_process.stdout is None:
            raise RuntimeError('Mind process stdout unavailable')

        for line in iter(mind_process.stdout.readline, ''):
            if stop_event.is_set():
                break
            parse_mind_output(line.strip())
            time.sleep(0.01)

        if stop_event.is_set():
            _terminate_process()
        else:
            mind_process.wait()
        current_state['running'] = False
    except Exception as e:
        print(f"Process error: {e}")
        current_state['running'] = False
    finally:
        stop_event.clear()
        with process_lock:
            if mind_process and mind_process.poll() is None:
                try:
                    mind_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    mind_process.kill()
                    mind_process.wait(timeout=1)
            mind_process = None
        mind_thread = None

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

@app.route('/api/phases')
def get_phases():
    """Get phase metrics from CSV logs"""
    phases = {}
    log_dir = '/root/Qallow/data/logs'

    # Load phase CSV files
    for phase_file in glob.glob(f'{log_dir}/phase*.csv'):
        phase_name = os.path.basename(phase_file).replace('.csv', '')
        data = load_csv_telemetry(phase_file)
        if data:
            phases[phase_name] = {
                'data': data[-10:],  # Last 10 entries
                'latest': data[-1] if data else {},
                'count': len(data),
            }

    # Load phase JSON metrics
    for metric_file in glob.glob(f'{log_dir}/phase*.json'):
        phase_name = os.path.basename(metric_file).replace('.json', '')
        metrics = load_json_metrics(metric_file)
        if phase_name in phases:
            phases[phase_name]['metrics'] = metrics
        else:
            phases[phase_name] = {'metrics': metrics}

    return jsonify(phases)

@app.route('/api/audit')
def get_audit():
    """Get ethics audit logs"""
    audit_file = '/root/Qallow/data/ethics_audit.log'
    audit_entries = []

    try:
        if os.path.exists(audit_file):
            with open(audit_file, 'r') as f:
                for line in f.readlines()[-50:]:  # Last 50 lines
                    audit_entries.append(line.strip())
    except Exception as e:
        print(f"Audit load error: {e}")

    return jsonify({'entries': audit_entries})

@app.route('/api/start', methods=['GET', 'POST'])
def start_mind():
    """Start mind process"""
    global mind_thread
    if current_state.get('running') and mind_thread and mind_thread.is_alive():
        return jsonify({'status': 'already running'})

    stop_event.clear()
    mind_thread = threading.Thread(target=run_mind_process, daemon=True)
    mind_thread.start()
    current_state['running'] = True
    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['GET', 'POST'])
def stop_mind():
    """Stop mind process"""
    global mind_thread
    stop_event.set()
    _terminate_process()

    if mind_thread and mind_thread.is_alive():
        mind_thread.join(timeout=2)
    mind_thread = None
    current_state['running'] = False
    return jsonify({'status': 'stopped'})


@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear telemetry and reset state"""
    if current_state.get('running'):
        return jsonify({'status': 'running'}), 409

    telemetry_buffer.clear()
    ethics_history.clear()
    phase_log.clear()
    audit_log.clear()
    reset_state()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

