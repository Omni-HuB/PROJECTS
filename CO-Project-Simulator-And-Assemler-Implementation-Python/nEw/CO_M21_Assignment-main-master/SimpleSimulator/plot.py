#!/usr/bin/python
import matplotlib.pyplot as plt

class plot:
	
	cycle = []
	mem_address = []

	def add_data(self, x, y):
		self.cycle.append(x)
		self.mem_address.append(y)

	def display(self):
		plt.xlabel('Cycle')
		plt.ylabel('Memory Address')
		plt.scatter(self.cycle, self.mem_address)
		plt.grid()
		plt.show()

	def save(self, filename):
		plt.xlabel('Cycle')
		plt.ylabel('Memory Address')
		plt.scatter(self.cycle, self.mem_address)
		plt.savefig(filename)
