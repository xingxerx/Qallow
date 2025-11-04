# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #!/usr/bin/env python3
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] Neuromorphic Processor Simulator
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] Simulates spiking neural networks (SNNs) with neuron dynamics, synaptic plasticity,
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] and event-based processing
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] class NeuronType(Enum):
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     LEAKY_INTEGRATE_AND_FIRE = "lif"
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     HODGKIN_HUXLEY = "hh"
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED]     IZHIKEVICH = "iz"
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] @dataclass
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] class Spike:
    """Represents a spike event"""
    neuron_id: int
    timestamp: float
    source_layer: int
    
    def __hash__(self):
        return hash((self.neuron_id, self.timestamp))


@dataclass
class Neuron:
    """Neuromorphic neuron model"""
    neuron_id: int
    neuron_type: NeuronType
    layer: int
    
    # LIF parameters
    membrane_potential: float = 0.0
    threshold: float = 1.0
    resting_potential: float = -70.0  # mV
    tau_membrane: float = 20.0  # ms (time constant)
    tau_synaptic: float = 5.0  # ms
    
    # Tracking
    spike_count: int = 0
    last_spike_time: float = 0.0
    synaptic_input: float = 0.0
    refractory_period: float = 2.0  # ms
    in_refractory: bool = False
    
    def reset(self):
        """Reset neuron to resting state"""
        self.membrane_potential = self.resting_potential
        self.synaptic_input = 0.0
        self.in_refractory = False


@dataclass
class Synapse:
    """Neural synapse with plasticity"""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float = 0.5
    delay: float = 1.0  # ms
    learning_rate: float = 0.01
    
    # Plasticity tracking
    last_update: float = 0.0
    pre_spike_trace: float = 0.0
    post_spike_trace: float = 0.0
    plasticity_enabled: bool = True


class NeuromorphicProcessor:
    """Simulates a neuromorphic processor with spiking neural networks"""
    
    def __init__(self, num_neurons: int = 1000, num_layers: int = 4):
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: Dict[Tuple[int, int], Synapse] = {}
        
        # Initialize neurons across layers
        neurons_per_layer = num_neurons // num_layers
        for layer in range(num_layers):
            for n in range(neurons_per_layer):
                neuron_id = layer * neurons_per_layer + n
                self.neurons[neuron_id] = Neuron(
                    neuron_id=neuron_id,
                    neuron_type=NeuronType.LEAKY_INTEGRATE_AND_FIRE,
                    layer=layer,
                    threshold=1.0 + random.uniform(-0.2, 0.2)
                )
        
        # Create random connectivity
        self.create_connectivity()
        
        # Spike tracking
        self.spike_log: List[Spike] = []
        self.current_time = 0.0
        self.time_step = 1.0  # ms
        
        # Statistics
        self.total_spikes = 0
        self.total_simulation_steps = 0
        self.energy_consumed_uj = 0.0  # microjoules
        self.latency_ms = 0.0
    
    def create_connectivity(self, connectivity_ratio: float = 0.1):
        """Create random synaptic connections"""
        neurons_list = list(self.neurons.keys())
        
        for _ in range(int(len(neurons_list) * connectivity_ratio)):
            pre_neuron = random.choice(neurons_list)
            post_neuron = random.choice(neurons_list)
            
            if pre_neuron != post_neuron:
                key = (pre_neuron, post_neuron)
                if key not in self.synapses:
                    self.synapses[key] = Synapse(
                        pre_neuron_id=pre_neuron,
                        post_neuron_id=post_neuron,
                        weight=random.uniform(0.1, 1.0)
                    )
    
    def inject_spikes(self, neuron_ids: List[int], current_time: float):
        """Inject spikes into neurons (input layer)"""
        for nid in neuron_ids:
            if nid in self.neurons:
                neuron = self.neurons[nid]
                neuron.synaptic_input += 2.0  # Input current
                
                # Record spike
                spike = Spike(neuron_id=nid, timestamp=current_time, source_layer=0)
                self.spike_log.append(spike)
    
    def update_neuron(self, neuron_id: int, current_time: float):
        """Update neuron state using LIF model"""
        neuron = self.neurons[neuron_id]
        
        # Skip if in refractory period
        if neuron.in_refractory:
            if current_time - neuron.last_spike_time > neuron.refractory_period:
                neuron.in_refractory = False
            else:
                return False
        
        # LIF dynamics: dV/dt = (V_rest - V + I*R) / tau_m
        decay = math.exp(-self.time_step / neuron.tau_membrane)
        neuron.membrane_potential = (
            neuron.resting_potential +
            (neuron.membrane_potential - neuron.resting_potential) * decay +
            neuron.synaptic_input * 10.0 * (1.0 - decay)
        )
        
        # Decay synaptic input
        neuron.synaptic_input *= math.exp(-self.time_step / neuron.tau_synaptic)
        
        # Check threshold
        did_spike = False
        if neuron.membrane_potential >= neuron.threshold:
            neuron.spike_count += 1
            neuron.last_spike_time = current_time
            neuron.in_refractory = True
            neuron.membrane_potential = neuron.resting_potential
            
            # Record spike
            spike = Spike(
                neuron_id=neuron_id,
                timestamp=current_time,
                source_layer=neuron.layer
            )
            self.spike_log.append(spike)
            self.total_spikes += 1
            did_spike = True
        
        return did_spike
    
    def propagate_spikes(self, current_time: float):
        """Propagate spikes through synapses"""
        # Find neurons that spiked in recent time window
        recent_spikes = [s for s in self.spike_log 
                        if s.timestamp > current_time - 5.0]
        
        for spike in recent_spikes:
            # Find outgoing synapses
            for (pre, post), synapse in self.synapses.items():
                if pre == spike.neuron_id:
                    # Apply synaptic delay
                    if current_time - spike.timestamp >= synapse.delay:
                        post_neuron = self.neurons[post]
                        post_neuron.synaptic_input += synapse.weight
                        
                        # Spike-timing-dependent plasticity (STDP)
                        if synapse.plasticity_enabled:
                            self.update_synapse_weight(synapse, current_time)
    
    def update_synapse_weight(self, synapse: Synapse, current_time: float):
        """Update synapse weight using STDP"""
        # Simplified STDP: Hebbian learning
        # If pre and post fire together, increase weight
        pre_neuron = self.neurons[synapse.pre_neuron_id]
        post_neuron = self.neurons[synapse.post_neuron_id]
        
        # Check if neurons recently spiked
        pre_recent = current_time - pre_neuron.last_spike_time < 10.0
        post_recent = current_time - post_neuron.last_spike_time < 10.0
        
        if pre_recent and post_recent:
            delta_w = synapse.learning_rate * 0.1
            synapse.weight = min(1.0, synapse.weight + delta_w)
        else:
            # Slight decay
            synapse.weight *= 0.99
    
    def simulate_step(self, current_time: float, inject_input: bool = False) -> Dict:
        """Simulate one time step"""
        self.current_time = current_time
        
        # Inject random input spikes (simulating sensory input)
        if inject_input:
            input_neurons = random.sample(
                [n for n in self.neurons.keys() if self.neurons[n].layer == 0],
                k=max(1, self.num_neurons // 100)
            )
            self.inject_spikes(input_neurons, current_time)
        
        # Update all neurons
        spikes_this_step = 0
        for neuron_id in self.neurons.keys():
            if self.update_neuron(neuron_id, current_time):
                spikes_this_step += 1
        
        # Propagate spikes
        self.propagate_spikes(current_time)
        
        # Energy calculation: roughly proportional to spike activity
        self.energy_consumed_uj += spikes_this_step * 0.001  # microjoules per spike
        
        self.total_simulation_steps += 1
        
        return {
            "time": current_time,
            "spikes_this_step": spikes_this_step,
            "total_spikes": self.total_spikes,
            "energy_uj": self.energy_consumed_uj,
        }
    
    def get_layer_spike_rate(self, layer: int) -> float:
        """Get average spike rate for a layer"""
        neurons_in_layer = [n for n in self.neurons.values() if n.layer == layer]
        if not neurons_in_layer:
            return 0.0
        
        avg_spikes = sum(n.spike_count for n in neurons_in_layer) / len(neurons_in_layer)
        return avg_spikes
    
    def get_stats(self) -> Dict:
        """Get processor statistics"""
        return {
            "total_neurons": self.num_neurons,
            "total_synapses": len(self.synapses),
            "total_spikes": self.total_spikes,
            "total_simulation_steps": self.total_simulation_steps,
            "energy_consumed_uj": self.energy_consumed_uj,
            "avg_spike_rate": self.total_spikes / max(1, self.total_simulation_steps),
            "spike_log_size": len(self.spike_log),
        }
    
    def get_connectivity_stats(self) -> Dict:
        """Get network connectivity statistics"""
        connectivity_ratio = len(self.synapses) / (self.num_neurons ** 2)
        avg_weight = sum(s.weight for s in self.synapses.values()) / max(1, len(self.synapses))
        
        return {
            "total_synapses": len(self.synapses),
            "connectivity_ratio": connectivity_ratio,
            "avg_synapse_weight": avg_weight,
            "max_synapse_weight": max((s.weight for s in self.synapses.values()), default=0),
            "min_synapse_weight": min((s.weight for s in self.synapses.values()), default=0),
        }
    
    def print_status(self):
        """Print processor status summary"""
        stats = self.get_stats()
        conn = self.get_connectivity_stats()
        
        print(f"\n{'='*70}")
        print(f"  Neuromorphic Processor Status")
        print(f"{'='*70}")
        print(f"  Neurons: {stats['total_neurons']}")
        print(f"  Synapses: {conn['total_synapses']} (connectivity: {conn['connectivity_ratio']:.2%})")
        print(f"  Total Spikes: {stats['total_spikes']}")
        print(f"  Avg Spike Rate: {stats['avg_spike_rate']:.2f} Hz")
        print(f"  Energy Consumed: {stats['energy_consumed_uj']:.2f} µJ")
        print(f"{'='*70}\n")
