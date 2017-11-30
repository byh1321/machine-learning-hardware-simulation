import numpy as np

def gen_error_array():
	f = open('0th_file.txt','r')
	last = f.readlines()[-1]
	end = last.find('t')
	arr_size = 2 ** 16
	err_array = np.zeros(arr_size)
	f.close()	
	start = 0
	f = open('0th_file.txt','r')
	for i in range(0,arr_size):
		line = f.readline()
		#print(line)
		#end = line.find('t')
		#repeat = long(line[0:end])

		num1 = int(line.rfind('('))
		num2 = int(line.rfind(','))
		num3 = int(line.rfind(')'))

		value = int(line[num1+1:num2])
		amount = int(line[num2+2:num3])
		#print(num1, num2, num3, value, amount)
		for j in range(start,start + amount):
			print(j)
			err_array[j] = value

		start = j+1
	f.close()
	return err_array
	

'''num1 = line.rfind('(')
num2 = line.rfind(',')
num3 = line.rfind(')')

'1th element (difference, number): (-16384, 96)\n'

value = line[num1+1:num2]
amount = line[num2+2:num3-1]'''
