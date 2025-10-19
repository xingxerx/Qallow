#!/usr/bin/env python3
"""
Minimal Tkinter UI for Qallow workflows.

Provides buttons to trigger the CUDA build, launch the CUDA binary,
and start the accelerator. Output from the invoked commands is streamed
into the UI without blocking the main thread so that performance of the
underlying processes is unaffected.
"""

import json
import os
import pathlib
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _expand_command(cmd):
    """Return a printable command string."""
    return " ".join(cmd)


class CommandRunner:
    """Run shell commands in a background thread and stream output."""

    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue
        self.process = None
        self.thread = None

    def run(self, cmd, cwd):
        if self.process and self.process.poll() is None:
            self.output_queue.put("[WARN] A command is already running.\n")
            return

        def worker():
            try:
                self.output_queue.put(f"[CMD] {_expand_command(cmd)}\n\n")
                self.process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.output_queue.put(line)
                self.process.wait()
                code = self.process.returncode
                self.output_queue.put(f"\n[CMD] Finished with exit code {code}\n\n")
            except Exception as exc:  # pylint: disable=broad-except
                self.output_queue.put(f"[ERROR] {exc}\n")
            finally:
                self.process = None

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.output_queue.put("[CMD] Termination signal sent.\n")


class QallowMonitorApp(tk.Tk):
    """Main window."""

    def __init__(self):
        super().__init__()
        self.title("Qallow CUDA Monitor")
        self.geometry("960x600")

        self.output_queue: queue.Queue[str] = queue.Queue()
        self.runner = CommandRunner(self.output_queue)

        self._build_ui()
        self.after(100, self._poll_output_queue)
        self.after(1000, self._refresh_metrics)

    def _build_ui(self):
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            btn_frame,
            text="Qallow Tasks",
            font=("Helvetica", 14, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 20))

        tk.Button(
            btn_frame,
            text="Build CUDA",
            command=self._build_cuda,
            width=15,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Run CUDA Binary",
            command=self._run_cuda_binary,
            width=15,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Run Accelerator",
            command=self._run_accelerator,
            width=15,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Stop",
            command=self.runner.stop,
            width=10,
        ).pack(side=tk.RIGHT, padx=5)

        info_frame = tk.Frame(self)
        info_frame.pack(fill=tk.X, padx=10)

        tk.Label(
            info_frame,
            text="Log Output",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor=tk.W)

        self.output_text = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            font=("Courier New", 10),
            state=tk.DISABLED,
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        metrics_frame = tk.LabelFrame(self, text="Pocket Dimension Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.metric_vars = {
            "tick": tk.StringVar(value="—"),
            "average_score": tk.StringVar(value="—"),
            "memory_usage": tk.StringVar(value="—"),
            "memory_peak": tk.StringVar(value="—"),
        }

        tk.Label(metrics_frame, text="Tick:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        tk.Label(metrics_frame, textvariable=self.metric_vars["tick"]).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Label(metrics_frame, text="Avg Score:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        tk.Label(metrics_frame, textvariable=self.metric_vars["average_score"]).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        tk.Label(metrics_frame, text="Memory Usage (MB):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        tk.Label(metrics_frame, textvariable=self.metric_vars["memory_usage"]).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Label(metrics_frame, text="Peak (MB):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        tk.Label(metrics_frame, textvariable=self.metric_vars["memory_peak"]).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        hint = (
            "Hints:\n"
            "  • Build CUDA: runs scripts/build_unified_cuda.sh\n"
            "  • Run CUDA Binary: executes build/qallow_unified_cuda if present\n"
            "  • Run Accelerator: executes ./scripts/run_auto.sh --watch=$(pwd)\n"
        )
        tk.Label(
            self,
            text=hint,
            justify=tk.LEFT,
            font=("Helvetica", 9),
        ).pack(fill=tk.X, padx=10, pady=(0, 10))

    def _poll_output_queue(self):
        try:
            while True:
                line = self.output_queue.get_nowait()
                self._append_output(line)
        except queue.Empty:
            pass
        self.after(100, self._poll_output_queue)

    def _append_output(self, text: str):
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def _refresh_metrics(self):
        metrics_path = REPO_ROOT / "data" / "telemetry" / "pocket_metrics.json"
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            self.output_queue.put("[WARN] Could not parse pocket_metrics.json\n")
        else:
            self.metric_vars["tick"].set(str(data.get("tick", "—")))
            self.metric_vars["average_score"].set(f"{data.get('average_score', 0.0):.4f}")
            self.metric_vars["memory_usage"].set(f"{data.get('memory_usage_mb', 0.0):.2f}")
            self.metric_vars["memory_peak"].set(f"{data.get('memory_peak_mb', 0.0):.2f}")

        self.after(1000, self._refresh_metrics)

    def _build_cuda(self):
        script = REPO_ROOT / "scripts" / "build_unified_cuda.sh"
        if not script.exists():
            messagebox.showerror("Error", f"Script not found: {script}")
            return
        self.runner.run(["bash", str(script)], cwd=REPO_ROOT)

    def _run_cuda_binary(self):
        binary = REPO_ROOT / "build" / "qallow_unified_cuda"
        if not binary.exists():
            messagebox.showwarning(
                "Missing Binary",
                f"{binary} not found.\nRun the CUDA build first.",
            )
            return
        self.runner.run([str(binary)], cwd=REPO_ROOT)

    def _run_accelerator(self):
        script = REPO_ROOT / "scripts" / "run_auto.sh"
        if not script.exists():
            messagebox.showerror("Error", f"Script not found: {script}")
            return
        cmd = ["bash", str(script), "--watch", str(REPO_ROOT)]
        self.runner.run(cmd, cwd=REPO_ROOT)


def main():
    os.chdir(REPO_ROOT)
    app = QallowMonitorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
