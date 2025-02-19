# my_config.py

import m5
from m5.objects import *

# Create a system with no CPUs or caches (as per the assignment requirements)
system = System()

# Create an instance of your VectorOperations SimObject
# vector_ops = VectorOperations()

# # Attach the SimObject to the system
# system.system = vector_ops

# # Run the simulation
# root = Root(full_system=False, system=system)
# # m5.instantiate()
# # m5.instantiate()

# print("Beginning simulation!")
# exit_event = m5.simulate()
# print('Exiting @ tick {} because {}'
#       .format(m5.curTick(), exit_event.getCause()))
# m5.simulate()



# Specify the path to the binary executable
binary = "/mnt/c/Users/moham/OneDrive/Desktop/GEM5/gem5/configs/CATutorial/test"

# Create a process for the application
process = Process()
# Set the command (executable path)
process.cmd = [binary]

# Set the CPU workload to the process and create thread contexts
system.cpu.workload = process
system.cpu.createThreads()

# Set up the root SimObject and start the simulation
root = Root(full_system=False, system=system)
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()
print('Exiting @ tick %i because %s' % (m5.curTick(), exit_event.getCause()))
m5.simulate()