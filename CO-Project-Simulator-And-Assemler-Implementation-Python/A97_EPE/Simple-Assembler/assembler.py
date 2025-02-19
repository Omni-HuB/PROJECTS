#!/usr/biin/python

# sys import for fetching the input file name from parameter
import sys
# re import for regex 
import re

# For Regex matching
label = "^[A-Za-z0-9_]+:$"
variable = "^[A-Za-z0-9_]+$"
imm = "^\$[0-9]+$"

#global variables
inst_lines = []
variable_table = {}
label_table = {}
already_halt = False

#Register Table Syntax: name:code
register_table = {
	"R0":'000',
	"R1":'001',
	"R2":'010',
	"R3":'011',
	"R4":'100',
	"R5":'101',
	"R6":'110',
	}

#Function to convert decimal to 8bit binary
def dtob(decimal):
	bnr = bin(decimal).replace('0b','')
	x = bnr[::-1] #this reverses an array
	while len(x) < 8:
		x += '0'
	return x[::-1]

#Function to parse add,sub,mul,xor,or,and
def parse_A(token, count, lineno):
	global already_halt
	if bool(already_halt):
		return "Invalid instructions {} at line {} after hlt".format(token[count], lineno)
	total_count = len(token)
	if total_count != (count + 1 + 3):
		return "Invalid Number of argument, \'{}\' expected 3 arguments at line {}".format(token[count], lineno)
	opcode = inst_table[token[count]][0]
	count += 1
	if not token[count] in register_table.keys():
		return "Invalid Register {} value at line {}".format(token[count], lineno)
	r1 = register_table[token[count]]
	count += 1
	if not token[count] in register_table.keys():
		return "Invalid Register {} value at line {}".format(token[count], lineno)
	r2 = register_table[token[count]]
	count += 1
	if not token[count] in register_table.keys():
		return "Invalid Register {} value at line {}".format(token[count], lineno)
	r3 = register_table[token[count]]
	inst_lines.append( ["{}00{}{}{}".format(opcode,r1,r2,r3), lineno, 0, ""]) 
	return ""	

#Function to parse rs and ls
def parse_B(token, count, lineno):
	global already_halt
	if bool(already_halt):
		return "Invalid instructions {} at line {} after hlt".format(token[count], lineno)
	total_count = len(token)
	if total_count != (count + 1 + 2):
		return "Invalid Number of argument, \'{}\' expected 2 arguments at line {}".format(token[count], lineno)
	opcode = inst_table[token[count]][0]
	count += 1
	if not token[count] in register_table.keys():
		return "Invalid Register {} value at line {}".format(token[count], lineno)
	r1 = register_table[token[count]]
	count += 1
	#check if it is immediate value
	if not bool(re.match(imm, token[count])):
		return "Invalid immediate {} value at line {}".format(token[count], lineno)
	imm_value = token[count]
	if int(imm_value[1:]) > 255:
		return "Invalid immediate {} value, value should be less than 255 at line {}".format(token[count], lineno)
	inst_lines.append( ["{}{}{}".format(opcode,r1,dtob(int(imm_value[1:]))), lineno, 0, ""]) 
	return ""	

#Function to parse div, not, cmp
def parse_C(token, count, lineno):
	global already_halt
	if bool(already_halt):
		return "Invalid instructions {} at line {} after hlt".format(token[count], lineno)
	total_count = len(token)
	if total_count != (count + 1 + 2):
		return "Invalid Number of argument, \'{}\' expected 2 arguments at line {}".format(token[count], lineno)
	opcode = inst_table[token[count]][0]
	count += 1
	if not token[count] in register_table.keys():
		return "Invalid Register {} value at line {}".format(token[count], lineno)
	r1 = register_table[token[count]]
	count += 1
	if not token[count] in register_table.keys():
		return "Invalid Register {} value at line {}".format(token[count], lineno)
	r2 = register_table[token[count]]
	inst_lines.append( ["{}00000{}{}".format(opcode,r1,r2), lineno, 0, ""]) 
	return ""	

#Function to parse ld and st
def parse_D(token, count, lineno):
	global already_halt
	if bool(already_halt):
		return "Invalid instructions {} at line {} after hlt".format(token[count], lineno)
	total_count = len(token)
	if total_count != (count + 1 + 2):
		return "Invalid Number of argument, \'{}\' expected 2 arguments at line {}".format(token[count], lineno)
	opcode = inst_table[token[count]][0]
	count += 1
	if not token[count] in register_table.keys():
		return "Invalid Register {} value at line {}".format(token[count], lineno)
	r1 = register_table[token[count]]

	count += 1
	if not token[count] in variable_table.keys():
		return "Variable {} is undeclared at line {}".format(token[count], lineno)
	inst_lines.append( ["{}{}".format(opcode,r1), lineno, 1, token[count]]) 
	return ""	

#Function to parse jmp, jlt, jgt, je
def parse_E(token, count, lineno):
	global already_halt
	if bool(already_halt):
		return "Invalid instructions {} at line {} after hlt".format(token[count], lineno)
	total_count = len(token)
	if total_count != (count + 1 + 1):
		return "Invalid Number of argument, \'{}\' expected 1 arguments at line {}".format(token[count], lineno)
	opcode = inst_table[token[count]][0]
	# depending on opcode we have to check if previous command was valid or not
	last_opcode = inst_lines[len(inst_lines)-1][0][:5]
	if opcode != '01111' and last_opcode != '01110':
		return "Invalid {} instruction at line {}".format(token[count] , lineno)
	count += 1
	inst_lines.append( ["{}000".format(opcode), lineno, 2, token[count]]) 
	return ""	

#Function to parse hlt
def parse_F(token, count, lineno):
	global already_halt
	total_count = len(token)
	if total_count != (count + 1):
		return "Invalid Number of argument, \'{}\' expected 0 arguments at line {}".format(token[count], lineno)
	opcode = inst_table[token[count]][0]
	count += 1
	if bool(already_halt):
		return "Multiple halt instruction detected at line {}".format(lineno)
	already_halt = True
	inst_lines.append( ["{}00000000000".format(opcode), lineno, 0, ""]) 
	return ""	

#Function to parse mov
def parse_G(token, count, lineno):
	global already_halt
	if bool(already_halt):
		return "Invalid instructions {} at line {} after hlt".format(token[count], lineno)
	total_count = len(token)
	if total_count != (count + 1 + 2):
		return "Invalid Number of argument, \'{}\' expected 2 arguments at line {}".format(token[count], lineno)
	count += 1
	if not token[count] in register_table.keys():
		return "Invalid Register {} value at line {}".format(token[count], lineno)
	r1 = register_table[token[count]]
	count += 1
	if token[count] in register_table.keys():
		inst_lines.append(["1001100000{}{}".format(r1, register_table[token[count]]), lineno, 0, ""])
		return ""
	if token[count] == "FLAGS":
		inst_lines.append(["1001100000{}111".format(r1), lineno, 0, ""])
		return ""
	if bool(re.match(imm, token[count])):
		imm_value = token[count]
		if int(imm_value[1:]) > 255:
			return "Invalid immediate {} value, value should be less than 255 at line {}".format(token[count], lineno)
		inst_lines.append( ["10010{}{}".format(r1,dtob(int(imm_value[1:]))), lineno, 0, ""])
		return ""
	return "Invalid immediate {} value format at line {}".format(token[count], lineno)

#Function to insert variable list 
def handle_variable(token, count, lineno):
	global already_halt
	if bool(already_halt):
		return "Invalid instructions {} at line {} after hlt".format(token[count], lineno)
	total_count = len(token)
	if total_count != (count + 1 + 1):
		return "Invalid Number of argument, \'{}\' expected 1 arguments at line {}".format(token[count], lineno)
	count += 1
	if not bool(re.match(variable, token[count])):
		return "Invalid name for Variable {} at line {}".format(token[count], lineno)
	if token[count] in variable_table.keys():
		return "Duplicate variable {} at line {}, already defined at line {}".format(token[count], lineno, variable_table[token[count]][1])
	if len(label_table):
		return "Variable {} at line {} should be declared at top".format(token[count], lineno) 
	if len(inst_lines):
		return "Variable {} at line {} should be declared at top".format(token[count], lineno) 
	if token[count] in label_table.keys():
		return "Conflict name {} at line {}, already defined as label at line {}".format(token[count], lineno, label_table[token[count]][1])
	variable_table[token[count]]=['', lineno]
	return ""

#Function to handle labels
def handle_label(token, lineno):
	global already_halt
	if bool(already_halt):
		return "Invalid instructions after hlt"
	label_name = token[0][:-1]
	if label_name in label_table.keys():
		return "Duplicate label {} at line {}, already defined at line {}".format(label_name, lineno, label_table[label_name][1])
	if label_name in variable_table.keys():
		return "Conflict name {} at line {}, already defined at line {}".format(label_name, lineno, variable_table[label_name][1])
	label_addr = len(inst_lines)
	label_table[label_name] = [dtob(int(label_addr)), lineno]

#Function to check all the syntax
def checkSyntax(line, lineno):
	token = line.split()
	total_count = len(token)
	count = 0
	if total_count == 0:
		#Blank Line
		return ""
	if bool(re.match(label,token[count])):
		#Label
		error = handle_label(token, lineno)
		if (error and error.split()):
			return error
		count += 1
	if total_count <= count:
		#Only Label
		return ""
	#Check if first one contain : but is invalid format
	if token[count][-1] == ':':
		return "Invalid name for Label {} at line {}".format(token[count], lineno)
	if token[count] == "var":
		#Variable
		return handle_variable(token, count, lineno)
	if token[count] in inst_table:
		#Instruction
		return inst_table[token[count]][1](token, count, lineno)
	else:
		return "Unknown Instruction {} found".format(token[count])

#Function to finally print after submit symbol tables
def final_update():
	syntax_error = False
	variable_addr = len(inst_lines)
	for x in variable_table:
		variable_table[x][0] = dtob(variable_addr)
		variable_addr += 1
	count = 0
	for line in inst_lines:
		final_inst = line[0]
		var_to_fill = line[3]
		if line[2] == 1: #case for load and store instructions
			inst_lines[count][0] += str(variable_table[var_to_fill][0])
		elif line[2] == 2: #case for jump instructions
			if not var_to_fill in label_table.keys():
				print("Label {} used but not defined at line {}\n".format(var_to_fill, line[1]))
				syntax_error = True
			else:
				inst_lines[count][0] += str(label_table[var_to_fill][0])
		count += 1	
	if bool(syntax_error):
		return
	for line in inst_lines:
		print(line[0])
	return

def main():
	global already_halt
	fileinput = sys.stdin.read()
	count = 0
	syntax_error = False
	already_halt = False
	asmLines = fileinput.split('\n')
	for line in asmLines:
		count += 1
		error = checkSyntax(line, count)
		if (error and error.strip()):
			syntax_error = True
			print(error)
	if bool(syntax_error):
		return
	if not bool(already_halt):
		print("No hlt instruction found in the program") 
	final_update()
	return
        
#instruction table syntax instruction: [opcode, parse_function, jump_flag condition(not using now)]
inst_table = {
        "add": ['10000', parse_A, 1 ],
        "sub": ['10001', parse_A, 1 ],
        "mov": ['10010', parse_G, 0 ],
        "ld":  ['10100', parse_D, 0 ],
        "st":  ['10101', parse_D, 0 ],
        "mul": ['10110', parse_A, 1 ],
        "div": ['10111', parse_C, 0 ],
        "rs":  ['11000', parse_B, 0 ],
        "ls":  ['11001', parse_B, 0 ],
        "xor": ['11010', parse_A, 0 ],
        "or":  ['11011', parse_A, 0 ],
        "and": ['11100', parse_A, 0 ],
        "not": ['11101', parse_C, 0 ],
        "cmp": ['11110', parse_C, 14],
        "jmp": ['11111', parse_E, 0 ],
        "jlt": ['01100', parse_E, 0 ],
        "jgt": ['01101', parse_E, 0 ],
        "je":  ['01111', parse_E, 0 ],
        "hlt": ['01010', parse_F, 0 ],
        }

if __name__ == "__main__":
	main()
