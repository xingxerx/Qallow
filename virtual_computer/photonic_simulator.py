# [REVIEWED] # [REVIEWED] # [REVIEWED] #!/usr/bin/env python3
# [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] Photonic Processor Simulator
# [REVIEWED] # [REVIEWED] # [REVIEWED] Simulates optical computing with photonic gates, light propagation, and optical circuits
# [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] class PhotonicGateType(Enum):
# [REVIEWED] # [REVIEWED] # [REVIEWED]     MACH_ZEHNDER = "mz"
# [REVIEWED] # [REVIEWED] # [REVIEWED]     BEAM_SPLITTER = "bs"
# [REVIEWED] # [REVIEWED] # [REVIEWED]     PHASE_MODULATOR = "pm"
# [REVIEWED] # [REVIEWED] # [REVIEWED]     OPTICAL_SWITCH = "os"
# [REVIEWED] # [REVIEWED] # [REVIEWED]     DIRECTIONAL_COUPLER = "dc"
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] @dataclass
# [REVIEWED] # [REVIEWED] # [REVIEWED] class Photon:
    """Represents a photon in the optical system"""
    photon_id: int
    wavelength_nm: float  # wavelength in nanometers
    power_dbm: float  # power in dBm
    phase: float  # phase in radians
    position: int  # position in circuit
    timestamp: float
    
    def __hash__(self):
        return hash(self.photon_id)


@dataclass
class OpticalWaveguide:
    """Represents an optical waveguide channel"""
    waveguide_id: int
    length_mm: float
    propagation_loss_db_per_km: float = 0.2
    chromatic_dispersion: float = 0.0  # ps/nm/km
    
    def get_loss(self) -> float:
        """Calculate total loss in this waveguide"""
        loss_km = self.propagation_loss_db_per_km
        loss_db = (self.length_mm / 1_000_000) * loss_km
        return loss_db


@dataclass
class PhotonicGate:
    """Photonic logic gate or component"""
    gate_id: int
    gate_type: PhotonicGateType
    input_ports: int
    output_ports: int
    
    # Optical parameters
    insertion_loss_db: float = 0.5
    switching_time_ns: float = 10.0
    modulation_efficiency: float = 0.95
    
    # State tracking
    is_active: bool = True
    switching_count: int = 0
    last_switch_time: float = 0.0
    output_power_dbm: float = 0.0


class PhotonicProcessor:
    """Simulates an integrated photonic processor"""
    
    def __init__(self, num_waveguides: int = 64, num_gates: int = 256):
        self.num_waveguides = num_waveguides
        self.num_gates = num_gates
        
        # Optical components
        self.waveguides: Dict[int, OpticalWaveguide] = {}
        self.gates: Dict[int, PhotonicGate] = {}
        self.photons: Dict[int, Photon] = {}
        
        # Initialize waveguides
        for i in range(num_waveguides):
            self.waveguides[i] = OpticalWaveguide(
                waveguide_id=i,
                length_mm=random.uniform(1.0, 10.0),
                propagation_loss_db_per_km=0.2 + random.uniform(-0.05, 0.05)
            )
        
        # Initialize gates
        gate_types = list(PhotonicGateType)
        for i in range(num_gates):
            gate_type = random.choice(gate_types)
            
            if gate_type == PhotonicGateType.BEAM_SPLITTER:
                inputs, outputs = 2, 2
            elif gate_type == PhotonicGateType.MACH_ZEHNDER:
                inputs, outputs = 2, 2
            elif gate_type == PhotonicGateType.OPTICAL_SWITCH:
                inputs, outputs = 2, 2
            else:
                inputs, outputs = 1, 1
            
            self.gates[i] = PhotonicGate(
                gate_id=i,
                gate_type=gate_type,
                input_ports=inputs,
                output_ports=outputs,
                insertion_loss_db=random.uniform(0.3, 1.0)
            )
        
        # Photon tracking
        self.photon_counter = 0
        self.photon_log: List[Photon] = []
        
        # Statistics
        self.total_photons_injected = 0
        self.total_photons_detected = 0
        self.total_switching_operations = 0
        self.total_propagation_distance_mm = 0.0
        self.power_budget_dbm = 0.0
        self.current_time = 0.0
    
    def inject_photons(self, count: int, power_dbm: float = -20.0, 
                      wavelength_nm: float = 1550.0) -> List[int]:
        """Inject photons into the optical system"""
        photon_ids = []
        
        for _ in range(count):
            self.photon_counter += 1
            photon = Photon(
                photon_id=self.photon_counter,
                wavelength_nm=wavelength_nm + random.uniform(-10, 10),
                power_dbm=power_dbm + random.uniform(-2, 2),
                phase=random.uniform(0, 2 * math.pi),
                position=0,
                timestamp=self.current_time
            )
            
            self.photons[self.photon_counter] = photon
            photon_ids.append(self.photon_counter)
            self.total_photons_injected += 1
        
        return photon_ids
    
    def propagate_through_waveguide(self, photon_id: int, 
                                   waveguide_id: int) -> Tuple[bool, float]:
        """Propagate a photon through a waveguide"""
        if photon_id not in self.photons or waveguide_id not in self.waveguides:
            return False, 0.0
        
        photon = self.photons[photon_id]
        waveguide = self.waveguides[waveguide_id]
        
        # Calculate loss
        loss_db = waveguide.get_loss()
        photon.power_dbm -= loss_db
        
        # Calculate propagation time
        # Speed of light in fiber: ~200,000 km/s
        speed_mm_per_ns = 200  # mm per nanosecond
        propagation_time_ns = waveguide.length_mm / speed_mm_per_ns
        
        # Chromatic dispersion effect
        dispersion_penalty = waveguide.chromatic_dispersion * (photon.wavelength_nm - 1550.0)
        propagation_time_ns += abs(dispersion_penalty) / 1000
        
        # Track statistics
        self.total_propagation_distance_mm += waveguide.length_mm
        
        return True, propagation_time_ns
    
    def apply_gate_operation(self, photon_ids: List[int], gate_id: int) -> Dict:
        """Apply a photonic gate operation to photons"""
        if gate_id not in self.gates:
            return {"success": False, "error": "Gate not found"}
        
        gate = self.gates[gate_id]
        
        if not gate.is_active:
            return {"success": False, "error": "Gate is inactive"}
        
        # Check inputs
        valid_photons = [pid for pid in photon_ids if pid in self.photons]
        if len(valid_photons) > gate.input_ports:
            valid_photons = valid_photons[:gate.input_ports]
        
        # Apply gate operation
        gate.switching_count += 1
        gate.last_switch_time = self.current_time
        self.total_switching_operations += 1
        
        output_photons = []
        
        if gate.gate_type == PhotonicGateType.BEAM_SPLITTER:
            # 50/50 beamsplitter - split photons between two outputs
            for photon_id in valid_photons:
                photon = self.photons[photon_id]
                photon.power_dbm -= gate.insertion_loss_db
                photon.power_dbm -= 3.0  # 3dB split
                output_photons.append(photon_id)
        
        elif gate.gate_type == PhotonicGateType.MACH_ZEHNDER:
            # Mach-Zehnder interferometer
            for photon_id in valid_photons:
                photon = self.photons[photon_id]
                # Simulate phase modulation
                photon.phase += random.uniform(0, math.pi)
                photon.power_dbm -= gate.insertion_loss_db
                output_photons.append(photon_id)
        
        elif gate.gate_type == PhotonicGateType.PHASE_MODULATOR:
            # Phase modulation
            for photon_id in valid_photons:
                photon = self.photons[photon_id]
                phase_shift = random.uniform(0, 2 * math.pi)
                photon.phase = (photon.phase + phase_shift) % (2 * math.pi)
                photon.power_dbm -= gate.insertion_loss_db * gate.modulation_efficiency
                output_photons.append(photon_id)
        
        elif gate.gate_type == PhotonicGateType.OPTICAL_SWITCH:
            # Optical switch - can route to different output
            for photon_id in valid_photons:
                photon = self.photons[photon_id]
                photon.position = random.randint(0, gate.output_ports - 1)
                photon.power_dbm -= gate.insertion_loss_db
                output_photons.append(photon_id)
        
        return {
            "success": True,
            "gate_type": gate.gate_type.value,
            "photons_processed": len(valid_photons),
            "output_photons": output_photons,
            "switching_time_ns": gate.switching_time_ns,
        }
    
    def detect_photons(self, photon_ids: List[int]) -> Dict:
        """Detect photons at output (with quantum efficiency)"""
        quantum_efficiency = 0.95  # ~95% typical
        detected = []
        not_detected = []
        
        for photon_id in photon_ids:
            if photon_id not in self.photons:
                continue
            
            photon = self.photons[photon_id]
            
            # Quantum efficiency based on power
            min_detectable_power = -40  # dBm
            if photon.power_dbm < min_detectable_power:
                not_detected.append(photon_id)
                continue
            
            # Random detection based on QE
            if random.random() < quantum_efficiency:
                detected.append(photon_id)
                self.total_photons_detected += 1
            else:
                not_detected.append(photon_id)
        
        return {
            "detected": detected,
            "not_detected": not_detected,
            "quantum_efficiency": quantum_efficiency,
        }
    
    def get_processor_stats(self) -> Dict:
        """Get processor statistics"""
        active_photons = len(self.photons)
        
        return {
            "num_waveguides": self.num_waveguides,
            "num_gates": self.num_gates,
            "total_photons_injected": self.total_photons_injected,
            "total_photons_detected": self.total_photons_detected,
            "active_photons": active_photons,
            "total_switching_operations": self.total_switching_operations,
            "total_propagation_distance_mm": self.total_propagation_distance_mm,
            "detection_efficiency": (self.total_photons_detected / max(1, self.total_photons_injected)),
        }
    
    def get_gate_stats(self) -> Dict:
        """Get statistics about gate operations"""
        total_switching = sum(g.switching_count for g in self.gates.values())
        avg_loss = sum(g.insertion_loss_db for g in self.gates.values()) / len(self.gates)
        
        return {
            "total_gates": len(self.gates),
            "active_gates": sum(1 for g in self.gates.values() if g.is_active),
            "total_switching_operations": total_switching,
            "avg_insertion_loss_db": avg_loss,
        }
    
    def get_waveguide_stats(self) -> Dict:
        """Get waveguide statistics"""
        avg_loss = sum(w.get_loss() for w in self.waveguides.values()) / len(self.waveguides)
        total_length = sum(w.length_mm for w in self.waveguides.values())
        
        return {
            "total_waveguides": len(self.waveguides),
            "total_length_mm": total_length,
            "avg_loss_per_waveguide_db": avg_loss,
        }
    
    def print_status(self):
        """Print processor status summary"""
        stats = self.get_processor_stats()
        gates = self.get_gate_stats()
        waveguides = self.get_waveguide_stats()
        
        print(f"\n{'='*70}")
        print(f"  Photonic Processor Status")
        print(f"{'='*70}")
        print(f"  Waveguides: {waveguides['total_waveguides']} (total: {waveguides['total_length_mm']:.1f} mm)")
        print(f"  Gates: {gates['total_gates']} active (avg loss: {gates['avg_insertion_loss_db']:.2f} dB)")
        print(f"  Photons: {stats['total_photons_injected']} injected, {stats['total_photons_detected']} detected")
        print(f"  Detection Efficiency: {stats['detection_efficiency']:.2%}")
        print(f"{'='*70}\n")
