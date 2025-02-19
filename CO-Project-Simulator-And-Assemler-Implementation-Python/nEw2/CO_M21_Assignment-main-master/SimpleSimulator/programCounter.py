#!/usr/bin/python

#Function to convert decimal to 8bit binary
def dtob(decimal):
	bnr = bin(decimal).replace('0b','')
	x = bnr[::-1] #this reverses an array
	while len(x) < 8:
		x += '0'
	return x[::-1]

class programCounter:
	pc = 0;
	def __init__(self):
		self.pc = 0
	
	def getValue(self):
		return self.pc		

	def dump(self):
		print(dtob(self.pc), end=" ")

	def update(self, index):
		self.pc = index
