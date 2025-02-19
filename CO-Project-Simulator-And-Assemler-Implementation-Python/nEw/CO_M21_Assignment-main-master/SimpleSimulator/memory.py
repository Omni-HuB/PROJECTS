#!/usr/bin/python
import sys

class memory:
	mem = []
	
	def initialize(self):
		fileinput = sys.stdin.read()
		asmLines = fileinput.split('\n')
		for line in asmLines:
			self.mem.append(line)
		totalLines = len(self.mem)
		for x in range(totalLines, 256):
			self.mem.append('0000000000000000')

	def setData(self, line, value):
		self.mem[line] = value;

	def getData(self, line):
		return self.mem[line];

	def dump(self):
		for line in self.mem:
			print(line)
		return	
	
