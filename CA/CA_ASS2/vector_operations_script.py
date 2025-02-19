import m5
from m5.objects import *

# Create a system
system = System()

# Create a CPU model (e.g., TimingSimpleCPU)
system.cpu = TimingSimpleCPU()

# Define memory ranges and buses
system.mem_ranges = [AddrRange('512MB')]
system.membus = SystemXBar()
system.cpu.icache_port = system.membus.slave
system.cpu.dcache_port = system.membus.slave
system.system_port = system.membus.master

# Reference the existing VectorOperations SimObject
vector_ops = VectorOperations

# Define initial vectors, DEBUG flags, and event cycles
vector_ops_vector1 = [1.0, 2.0, 3.0]
vector_ops_vector2 = [4.0, 5.0, 6.0]
debug_flags = {
    "Vector": True,
    "ResultCross": True,
    "Normalize": True,
    "ResultSub": True
}

# Add the SimObject and parameters to the system
system.workload = vector_ops
system.vector_ops_vector1 = vector_ops_vector1
system.vector_ops_vector2 = vector_ops_vector2
system.debug_flags = debug_flags

# Schedule the events
system.cpu.add_vector_event(150, vector_ops.event)
system.cpu.add_vector_event(1500, vector_ops.event)
system.cpu.add_vector_event(15000, vector_ops.event)

# Create a root and instantiate the simulation
root = Root(full_system=False, system=system)
m5.instantiate()

# Run the simulation
print("Running simulation...")
exit_event = m5.simulate()

# Print simulation result
print("Exiting @ tick %i because %s" % (m5.curTick(), exit_event.getCause()))
