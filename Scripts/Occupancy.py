import numpy as np
from sys import argv

# The script require a file A$n.xvg created for each chain with the command
# gmx pairdist -f pbc.xtc -s md.tpr -o Occupancy/A$n.xvg -ref A$n -sel S -refgrouping none -selgrouping all -n index-complete.ndx -b 2000000
# The file should be in a folder called Occupancy

Nchains = int(argv[1])

FileStat = f'{Nchains}A100-4000B/Chain-Avg_ErrOccupation.dat'

file = f'{Nchains}A100-4000B/Occupancy/A'

cutoff = 1.0 # nm

with open(FileStat, 'w') as fo:
      fo.write('{:<10}{:<15}{:<15}\n'.format('# Chain', 'AvgOcc', 'ErrOcc'))
      for N in range(1,Nchains+1):
            print(N)
            Bound = []
            fileA = file + f'{N}.xvg'
            with open(fileA, 'r') as fi:
                  for row in fi:
                        if row[0] not in ['#', '@']:
                              cols = row.split()
                              cols = np.array(row.split(), dtype='float')
                              Bound.append(sum(cols[1:] < cutoff) / 100)

            fo.write('{:<10}{:<15}{:<15}\n'.format(N, format(np.mean(Bound), '.3E'), format(np.std(Bound), '.3E')))