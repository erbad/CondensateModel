import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
from math import dist
from sys import argv
import pathlib

# COMMAND: python density.py ePP ePR
# To search the folder=ePP$ePP/ePR$ePR/rep_a/
# requires folder/pbc.xtc and folder/md.gro

##################################################
########## FUNCTIONS

def FitDensityProfile(x, rho_cond, rho_dil, z_ds, t):

    return .5 * (rho_cond + rho_dil) - .5 * (rho_cond - rho_dil) * np.tanh((np.abs(x) - z_ds)/t)

def SphereVolume(radius):

      return 4/3 * np.pi * radius**3

def ComputeDensity(traj, binning, Begin, End):
      
      # Quantities related to the simulation
      side = traj.dimensions[0] / 10  # nm
      r_co = 6 # nm
      C_sel, A100_sel, A10_sel = traj.select_atoms('name C'), traj.select_atoms('index 0 to 1999'),  traj.select_atoms('index 2000 to 3999')
      C_ind, A100_ind, A10_ind = list([i.index for i in C_sel]), list([i.index for i in A100_sel]), list([i.index for i in A10_sel])
      C_ind_from0 = np.array((np.array(C_ind) - len(A100_ind) - len(A10_ind)) / 2, dtype=int)
      A10_ind_from0 = np.array((np.array(A10_ind) - len(A100_ind)), dtype=int)

      # Quantities we want to get
      CM_ts, cluster_size = [], []
      HISTC, BINSC = [], []
      HISTA100, BINSA100 = [], []
      HISTA10, BINSA10 = [], []
      count = 0
      
      BaseBins = range(0, int(np.floor(np.sqrt(3) * side)), 5) # 1 bin each 5 sigmas in a range going from 0 to the length of box diagonal

      for t in range(Begin, End, 10):
            
            count+=1
            pos_C = np.array(traj.trajectory[t].positions[C_ind]) / 10 # nm
            pos_A100 = np.array(traj.trajectory[t].positions[A100_ind]) / 10 # nm
            pos_A10 = np.array(traj.trajectory[t].positions[A10_ind]) / 10 # nm

            # getting the biggest cluster
            G = nx.Graph()
            G.add_nodes_from(C_ind_from0)

            pos_in_box = pos_C.T / side
            pair_shift = pos_in_box[:, :, np.newaxis] - pos_in_box[:, np.newaxis, :]
            pair_shift = pair_shift - np.rint(pair_shift)
            
            # distance matrix
            dist_nd = np.linalg.norm(pair_shift, axis=0) * side
            Cij = np.dstack(np.where((dist_nd <= r_co) & (dist_nd > 0)))
            for C in Cij[0]:
                  G.add_edges_from([(C[0], C[1])])
            largest_cluster = max(nx.connected_components(G), key=len)
            cluster_size.append(len(largest_cluster))
            #print(t, len(largest_cluster))

            # computing its center of mass

            center_of_mass_cluster = np.mean([pos_C[C] for C in largest_cluster], axis=0)

            CM_ts.append(center_of_mass_cluster)

            # making the radial distribution function for C
            distancesC = [dist(center_of_mass_cluster, pos_C[C]) for C in C_ind_from0]


            histC, binsC = np.histogram(distancesC, bins=BaseBins, density=False) 

            BINSC = [.5 * (binsC[b] + binsC[b+1]) for b in range(len(binsC)-1)]
            hist_rewC = [histC[h]/ ((SphereVolume(binsC[h+1]) - SphereVolume(binsC[h]))) for h in range(len(histC))]
            HISTC.append(hist_rewC)

            # making the radial distribution function for A100
            distancesA100 = [dist(center_of_mass_cluster, pos_A100[A]) for A in A100_ind]
      
            histA100, binsA100 = np.histogram(distancesA100, bins=BaseBins, density=False)
            
            BINSA100 = [.5 * (binsA100[b] + binsA100[b+1]) for b in range(len(binsA100)-1)]
            hist_rewA100 = [histA100[h]/((SphereVolume(binsA100[h+1]) - SphereVolume(binsA100[h]))) for h in range(len(histA100))]
            HISTA100.append(hist_rewA100)

            # making the radial distribution function for A100
            distancesA10 = [dist(center_of_mass_cluster, pos_A10[A]) for A in A10_ind_from0]
            histA10, binsA10 = np.histogram(distancesA10, bins=BaseBins, density=False)

            BINSA10 = [.5 * (binsA10[b] + binsA10[b+1]) for b in range(len(binsA10)-1)]
            hist_rewA10 = [histA10[h]/((SphereVolume(binsA10[h+1]) - SphereVolume(binsA10[h]))) for h in range(len(histA10))]
            HISTA10.append(hist_rewA10)

            if t%250==0:
                  print('Step %s done!' %(t))
                  print(len(largest_cluster), center_of_mass_cluster)
      print('Analyzed %s frames' %count)
      
      return HISTC, np.array(BINSC), HISTA100, np.array(BINSA100), HISTA10, np.array(BINSA10), cluster_size, CM_ts

##################################################
########## MAIN

# Generating name of folders in Matteo directory according to simulation type
ePP = float(argv[1])    # ePP
ePR = int(argv[2])      # epR 


# Creating a folder in this directory
pwd = pathlib.Path().resolve()
folder = f'ePP{ePP}/ePR{ePR}' 


gro, xtc = f'{folder}/md.gro', f'{folder}/md.xtc'

traj = mda.Universe(gro, xtc)
side = traj.dimensions[0]
SimFrames = len(traj.trajectory)    # End of Density analysis frames
DensFrames = SimFrames - 3000       # Begin of Density analysis frames

print(SimFrames, DensFrames)

binning = 5

Nparts = 3

HISTC, BINSC, HISTA100, BINSA100, HISTA10, BINSA10, cluster_size, CM_ts = ComputeDensity(traj, binning, DensFrames, SimFrames)

colors = {'C':'tab:blue',
          'A100':'tab:red',
          'A10':'darkgoldenrod'}

# Fitting C density

fig, ax = plt.subplots(1,1, figsize=(6,6))

ax.set_xlim([0,50])
ax.set_xlim([0,50])

ax.set_xlabel(r'$r$ [$\sigma$]', fontsize=18)
ax.set_xticks(range(0,60,10))
ax.set_xticklabels(range(0,60,10), fontsize=14)

ax.set_ylabel(r'$\rho$ [$10^{-2}\sigma^{-3}$]', fontsize=18)
ax.set_yticks([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
ax.set_yticklabels(range(1,9,1), fontsize=14)

avg_histoC = np.mean(HISTC, axis=0)
for h in HISTC:
      ax.plot(np.array(BINSC), h, alpha=0.05, color=colors['C'], linewidth=0.6)
            
nbins = int((side / 10) // 2 // binning // 2) # half_side [nm] / bin_width / 2 (all integer)

print(nbins)

ax.plot(np.array(BINSC), avg_histoC, linewidth=4, color=colors['C'], label='Protein')


with open(folder + '/densityC.dat', 'w') as fo:
      for B, H in zip(BINSC, avg_histoC):
            fo.write('{:<15}{:<15}\n'.format(format(B, '.3E'), format(H, '.3E')))

# Fitting A100 density

avg_histoA100 = np.mean(HISTA100, axis=0)
for h in HISTA100:
      plt.plot(np.array(BINSA100), h, alpha=0.05, color=colors['A100'], linewidth=0.6)


ax.plot(np.array(BINSA100), avg_histoA100, linewidth=4, color=colors['A100'], label='RNA 100')
#ax.set_ylim([0, 3 * dens_condA])

with open(folder + '/densityA100.dat', 'w') as fo:
      for B, H in zip(BINSA100, avg_histoA100):
            fo.write('{:<15}{:<15}\n'.format(format(B, '.3E'), format(H, '.3E')))


# Fitting A100 density

avg_histoA10 = np.mean(HISTA10, axis=0)
for h in HISTA10:
      ax.plot(np.array(BINSA10), h, alpha=0.05, color=colors['A10'], linewidth=0.6)

ax.plot(np.array(BINSA10), avg_histoA10, linewidth=4, color=colors['A10'], label='RNA 10')

ax.legend(fontsize=15)
fig.savefig(folder + '/density.png', dpi=300, bbox_inches='tight')

with open(folder + '/densityA10.dat', 'w') as fo:
      for B, H in zip(BINSA10, avg_histoA10):
            fo.write('{:<15}{:<15}\n'.format(format(B, '.3E'), format(H, '.3E')))

