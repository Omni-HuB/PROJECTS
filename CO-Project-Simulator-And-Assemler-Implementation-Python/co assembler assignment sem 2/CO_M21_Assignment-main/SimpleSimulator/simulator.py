#!/usr/bin/python
from memory import memory
from programCounter import programCounter
from registerFile import registerFile
from executeEngine import executeEngine
from plot import plot
import random

MEM = memory()
PC = programCounter()
RF = registerFile()
EE = executeEngine(RF, PC, MEM)
GRAPH = plot()

def initialize(MEM):
	MEM.initialize()
	return


def main():
	initialize(MEM)
	PC.update(0)
	halted = False;
	cycle = 1;

	while(not halted):
		GRAPH.add_data(cycle, PC.getValue()) # for Graph X:cycle Y:PC(As this is the address of memory to execute)  
		instruction = MEM.getData(PC.getValue())
		halted, new_PC = EE.execute(instruction)
		PC.dump()
		RF.dump()
		PC.update(new_PC)
		cycle += 1

	MEM.dump()
	# GRAPH.display() #to display Graph
	GRAPH.save("graph{}.png".format(random.random())) # to save Graph

if __name__ == "__main__":
	main()
