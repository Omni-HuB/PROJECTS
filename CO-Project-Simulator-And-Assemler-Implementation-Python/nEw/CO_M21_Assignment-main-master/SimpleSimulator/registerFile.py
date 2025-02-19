#!/usr/bin/python

#Function to convert decimal to 8bit binary
def dto16b(decimal):
        bnr = bin(decimal).replace('0b','')
        x = bnr[::-1] #this reverses an array
        while len(x) < 16:
                x += '0'
        return x[::-1]

class registerFile:
	registers = [0, 0, 0, 0, 0, 0, 0, 0]

	def dump(self):
		for reg in self.registers:
			print(dto16b(reg), end=" ")
		print()
		return

	def getValue(self, reg_no):
		return self.registers[int(reg_no, 2)]

	def getFlag(self):
		return self.registers[7];

	def resetFlag(self):
		self.registers[7] = 0;
	
	def setVFlag(self):
		self.registers[7] = 8;

	def setLFlag(self):
		self.registers[7] = 4;

	def setGFlag(self):
		self.registers[7] = 2;

	def setEFlag(self):
		self.registers[7] = 1;

	def setR0(self, value):
		self.registers[0] = value;

	def setR1(self, value):
		self.registers[1] = value;

	def setValue(self, reg_no, value):
		self.registers[int(reg_no,2)] = value;
		return;
