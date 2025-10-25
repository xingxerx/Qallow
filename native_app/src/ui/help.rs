use crate::models::AppState;
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

pub fn create_help_panel(parent: &mut group::Tabs, _state: Arc<Mutex<AppState>>) {
    let help_group = group::Group::default().with_label("❓ Help");

    let mut flex = group::Flex::default().with_size(1450, 950).column();
    flex.set_color(Color::from_hex(0x0a0e27));

    // Title
    let mut title = text::TextDisplay::default().with_size(1450, 40);
    title.set_buffer(text::TextBuffer::default());
    title
        .buffer()
        .unwrap()
        .set_text("Qallow Help & Documentation");
    title.set_text_color(Color::from_hex(0x00d4ff));

    // Help content
    let mut help_text = text::TextDisplay::default().with_size(1450, 850);
    help_text.set_buffer(text::TextBuffer::default());

    let help_content = r#"🚀 QALLOW UNIFIED VM - HELP & DOCUMENTATION

📋 QUICK START
==============
1. Select a build type (CPU or CUDA) from the Control Panel
2. Choose a phase (13, 14, or 15)
3. Configure parameters (ticks, fidelity, epsilon)
4. Click "Start VM" to begin execution
5. Monitor progress in the Terminal and Metrics tabs

🎮 MAIN FEATURES
================
• Dashboard: Overview of system status and metrics
• Metrics: Real-time performance and stability metrics
• Terminal: Live output from running processes
• Audit Log: Historical record of all operations
• Control Panel: VM configuration and execution controls
• Settings: Application preferences and configuration
• Help: This documentation

⌨️ KEYBOARD SHORTCUTS
====================
Ctrl+Q       - Quit application
Ctrl+Shift+S - Start VM
Ctrl+Shift+X - Stop VM
Ctrl+1-5     - Switch between tabs
F1           - Show this help
Ctrl+L       - Clear terminal
Ctrl+S       - Save settings

⚙️ PHASES EXPLAINED
===================
Phase 13: Accelerated Quantum Simulation
  - Coherence lattice integration
  - Quantum state evolution
  - Default: 1000 ticks

Phase 14: Coherence Lattice Integration
  - QAOA tuner integration
  - Fidelity optimization
  - Default: 0.981 target fidelity

Phase 15: Convergence & Lock-in
  - Final convergence analysis
  - Stability assessment
  - Default: 5e-6 epsilon

📊 METRICS EXPLAINED
====================
Overlay Stability:
  - Orbital: Orbital mechanics stability
  - River: River dynamics stability
  - Mycelial: Mycelial network stability
  - Global: Overall system stability

Ethics Score:
  - Safety: Safety assessment score
  - Clarity: Clarity and transparency score
  - Human: Human alignment score

Coherence: Quantum coherence level (0-1)
GPU Memory: GPU memory usage in GB
CPU Memory: CPU memory usage in GB

🔧 TROUBLESHOOTING
==================
Q: VM won't start
A: Check that the binary path is correct and has execute permissions.
   Verify the selected phase is available.

Q: Process crashes
A: Check the Audit Log for error messages.
   Try reducing the number of ticks or target fidelity.

Q: High memory usage
A: Reduce the problem size or number of ticks.
   Monitor metrics to identify bottlenecks.

Q: Slow performance
A: Try using CUDA build if available.
   Reduce the number of ticks or problem complexity.

📚 ADDITIONAL RESOURCES
=======================
• GitHub: https://github.com/xingxerx/Qallow
• Documentation: See docs/ directory
• Issues: Report bugs on GitHub
• Contributing: See CONTRIBUTING.md

💡 TIPS & TRICKS
================
1. Use Ctrl+L to clear terminal for better readability
2. Monitor metrics in real-time while running
3. Save settings regularly with Ctrl+S
4. Check audit log for detailed operation history
5. Use keyboard shortcuts for faster navigation

🆘 GETTING HELP
================
If you encounter issues:
1. Check the Audit Log for error messages
2. Review the Terminal output for details
3. Check the Help documentation
4. Report issues on GitHub with:
   - Error message
   - Steps to reproduce
   - System information
   - Relevant logs

Version: 1.0.0
Last Updated: 2025-10-25
"#;

    help_text.buffer().unwrap().set_text(help_content);
    help_text.set_text_color(Color::White);

    flex.end();
    help_group.end();
    parent.add(&help_group);
}
