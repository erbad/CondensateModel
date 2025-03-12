from sys import argv
import subprocess

# Creates a file with just indeces of single chains.
# Then append it to the index.ndx file to create an
# index-complete.ndx file.

Nchains = int(argv[1])

folder = f'{Nchains}A100-4000B'

with open(f'{folder}/index-A.ndx', 'w') as fo:
	for N in range(1,Nchains+1):
		fo.write(f'\n [ A{N} ] \n')
		beg, end = 100 * (N-1) + 1, 100 * N 
		count = 0
		for i in range(beg, end+1):
			if count % 16 != 15:
				fo.write('{:<6}'.format(i))
			else:
				fo.write('{:<6}\n'.format(i))
			count+=1

	fo.write('\n')

CommandCat = f'cat {folder}/index.ndx {folder}/index-A.ndx > {folder}/index-complete.ndx'
print(subprocess.run(CommandCat, shell=True))
