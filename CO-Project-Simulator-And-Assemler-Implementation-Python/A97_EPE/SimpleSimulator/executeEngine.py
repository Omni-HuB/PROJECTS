#!/usr/bin/python

def dtob(decimal):
        bnr = bin(decimal).replace('0b','')
        x = bnr[::-1] #this reverses an array
        while len(x) < 8:
                x += '0'
        return x[::-1]
        
def dto16b(decimal):
        bnr = bin(decimal).replace('0b','')
        x = bnr[::-1] #this reverses an array
        while len(x) < 16:
                x += '0'
        return x[::-1]
        
#instruction table syntax instruction: opcode: [name, function] 
inst_table = {
        '10000': "add",
        '10001': "sub",
        '10010': "mov1",
        '10011': "mov2",
        '10100': "ld" ,
        '10101': "st" ,
        '10110': "mul",
        '10111': "div",
        '11000': "rs" ,
        '11001': "ls" ,
        '11010': "xor",
        '11011': "or" ,
        '11100': "and",
        '11101': "not",
        '11110': "cmp",
        '11111': "jmp",
        '01100': "jlt",
        '01101': "jgt",
        '01111': "je" ,
        '01010': "hlt",
        }

class executeEngine:
	
	
	def __init__(self, rf, pc, mem):
		self.rf = rf;
		self.pc = pc
		self.mem = mem

	def execute(self, line):
		function = 'func_' + inst_table[line[:5]]
		method= getattr(self, function, "Invalid Instruction")
		return method(line)

	def func_add(self, line):
		self.rf.resetFlag()
		r2 = self.rf.getValue(line[10:13])		
		r3 = self.rf.getValue(line[13:16])
		result = r2+r3
		if ((result/(2**16)) > 0):
			self.rf.setVFlag()
			result = result%(2**16)
		self.rf.setValue(line[7:10], result)
		return False, self.pc.getValue() + 1

	def func_sub(self, line):
		self.rf.resetFlag()
		r2 = self.rf.getValue(line[10:13])
		r3 = self.rf.getValue(line[13:16])
		result = r2 - r3
		if (r2 < r3) :
			self.rf.setVFlag()
			result = 0;
		self.rf.setValue(line[7:10], result)
		return False, self.pc.getValue() + 1
			 
	def func_mov1(self, line):
		imm = int(line[8:16],2)
		self.rf.setValue(line[5:8], imm)
		self.rf.resetFlag()
		return False, self.pc.getValue() + 1

	def func_mov2(self, line):
		r2 = self.rf.getValue(line[13:16]) #TODO 
		self.rf.setValue(line[10:13], r2)
		self.rf.resetFlag()
		return False, self.pc.getValue() + 1

	def func_ld(self, line):
		self.rf.resetFlag()
		result = int(self.mem.getData(int(line[8:16],2)),2)
		self.rf.setValue(line[5:8], result)
		return False, self.pc.getValue() + 1

	def func_st(self, line):
		self.rf.resetFlag()
		r1 = self.rf.getValue(line[5:8])
		self.mem.setData(int(line[8:16],2), dto16b(r1))		
		return False, self.pc.getValue() + 1

	def func_mul(self, line):
		self.rf.resetFlag()
		r2 = self.rf.getValue(line[10:13])		
		r3 = self.rf.getValue(line[13:16])
		result = r2*r3
		if ((result/(2**16)) > 0):
			self.rf.setVFlag()
			result = result%(2**16)
		self.rf.setValue(line[7:10], result)
		return False, self.pc.getValue() + 1

	def func_div(self, line):
		self.rf.resetFlag()
		r2 = self.rf.getValue(line[10:13])		
		r3 = self.rf.getValue(line[13:16])
		if (r3 == 0):
			return True, self.pc.getValue() + 1
		result = r2/r3
		result_remainder = r2%r3
		self.rf.setR0(result)
		self.rf.setR1(result_remainder)
		return False, self.pc.getValue() + 1

	def func_rs(self, line):
		self.rf.resetFlag()
		imm = int(line[8:16],2)
		r1 = self.rf.getValue(line[5:8])
		result = r1 >> imm
		result = result % (2**16)
		self.rf.setValue(line[5:8], result)
		return False, self.pc.getValue() + 1

	def func_ls(self, line):
		self.rf.resetFlag()
		imm = int(line[8:16],2)
		r1 = self.rf.getValue(line[5:8])
		result = r1 << imm
		result = result % (2**16)
		self.rf.setValue(line[5:8], result)
		return False, self.pc.getValue() + 1

	def func_xor(self, line):
		self.rf.resetFlag()
		r2 = self.rf.getValue(line[10:13])
		r3 = self.rf.getValue(line[13:16])
		result = r2 ^ r3
		self.rf.setValue(line[7:10], result)
		return False, self.pc.getValue() + 1

	def func_or(self, line):
		self.rf.resetFlag()
		r2 = self.rf.getValue(line[10:13])
		r3 = self.rf.getValue(line[13:16])
		result = r2 | r3
		self.rf.setValue(line[7:10], result)
		return False, self.pc.getValue() + 1

	def func_and(self, line):
		self.rf.resetFlag()
		r2 = self.rf.getValue(line[10:13])
		r3 = self.rf.getValue(line[13:16])
		result = r2 & r3
		self.rf.setValue(line[7:10], result)
		return False, self.pc.getValue() + 1

	def func_not(self, line):
		self.rf.resetFlag()
		r2 = self.rf.getValue(line[13:16])
		result = ~r2
		result = result%(2**16)
		self.rf.setValue(line[10:13], result)
		return False, self.pc.getValue() + 1

	def func_cmp(self, line):
		self.rf.resetFlag()
		r2 = self.rf.getValue(line[10:13])
		r3 = self.rf.getValue(line[13:16])
		if (r2 == r3) :
			self.rf.setEFlag()
		elif (r2 < r3) :
			self.rf.setLFlag()
		else:
			self.rf.setGFlag()
		return False, self.pc.getValue() + 1

	def func_jmp(self, line):
		pc_counter = int(line[8:16],2)
		self.rf.resetFlag()
		return False, pc_counter

	def func_jlt(self, line):
		flag = self.rf.getFlag()
		pc_value = int(line[8:16],2)
		pc_counter = self.pc.getValue() + 1
		if (flag & 0x04):
			pc_counter = pc_value
		self.rf.resetFlag()
		return False, pc_counter

	def func_jgt(self, line):
		flag = self.rf.getFlag()
		pc_value = int(line[8:16],2)
		pc_counter = self.pc.getValue() + 1
		if (flag & 0x02):
			pc_counter = pc_value
		self.rf.resetFlag()
		return False, pc_counter

	def func_je(self, line):
		flag = self.rf.getFlag()
		pc_value = int(line[8:16],2)
		pc_counter = self.pc.getValue() + 1
		if (flag & 0x01):
			pc_counter = pc_value
		self.rf.resetFlag()
		return False, pc_counter

	def func_hlt(self, line):
		return True, self.pc.getValue() + 1

