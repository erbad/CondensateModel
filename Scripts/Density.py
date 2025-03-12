import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from scipy.optimize import curve_fit
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
      C_sel, A_sel = traj.select_atoms('name C'), traj.select_atoms('name A')
      C_ind, A_ind = list([i.index for i in C_sel]), list([i.index for i in A_sel])
      C_ind_from0 = np.array((np.array(C_ind) - len(A_ind)) / 2, dtype=int)

      # Quantities we want to get
      CM_ts, cluster_size = [], []
      HISTC, BINSC = [], []
      HISTA, BINSA = [], []
      count = 0
      
      BaseBins = range(0, int(np.floor(np.sqrt(3) * side)), 5) # 1 bin each 5 sigmas in a range going from 0 to the length of box diagonal

      for t in range(Begin, End, 10):
            
            count+=1
            pos_C = np.array(traj.trajectory[t].positions[C_ind]) / 10 # nm
            pos_A = np.array(traj.trajectory[t].positions[A_ind]) / 10 # nm

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
            print(t, len(largest_cluster))

            # computing its center of mass

            center_of_mass_cluster = np.mean([pos_C[C] for C in largest_cluster], axis=0)

            CM_ts.append(center_of_mass_cluster)

            # making the radial distribution function for C
            distancesC = [dist(center_of_mass_cluster, pos_C[C]) for C in C_ind_from0]


            histC, binsC = np.histogram(distancesC, bins=BaseBins, density=False) 

            BINSC = [.5 * (binsC[b] + binsC[b+1]) for b in range(len(binsC)-1)]
            hist_rewC = [histC[h]/ ((SphereVolume(binsC[h+1]) - SphereVolume(binsC[h]))) for h in range(len(histC))]
            HISTC.append(hist_rewC)

            # making the radial distribution function for A
            distancesA = [dist(center_of_mass_cluster, pos_A[A]) for A in A_ind]
      
            histA, binsA = np.histogram(distancesA, bins=BaseBins, density=False)
            
            BINSA = [.5 * (binsA[b] + binsA[b+1]) for b in range(len(binsA)-1)]
            hist_rewA = [histA[h]/((SphereVolume(binsA[h+1]) - SphereVolume(binsA[h]))) for h in range(len(histA))]
            HISTA.append(hist_rewA)

            if t%250==0:
                  print('Step %s done!' %(t))
                  print(len(largest_cluster), center_of_mass_cluster)
      print('Analyzed %s frames' %count)
      
      return HISTC, np.array(BINSC), HISTA, np.array(BINSA), cluster_size, CM_ts

##################################################
########## MAIN

# Generating name of folders in Matteo directory according to simulation type
ePP = float(argv[1])    # ePP
ePR = int(argv[2])      # epR 


# Creating a folder in this directory
pwd = pathlib.Path().resolve()
folder = f'ePP{ePP}/ePR{ePR}/rep_a' 


gro, xtc = f'{folder}/md.gro', f'{folder}/md.xtc'

traj = mda.Universe(gro, xtc)
side = traj.dimensions[0]
SimFrames = len(traj.trajectory)
DensFrames = SimFrames - 3000

print(SimFrames, DensFrames)

binning = 5

Nparts = 3

with open(f'{folder}/FittingParams_C-Bin{binning}.dat', 'w') as fo:
      fo.write('{:<6} {:<24}{:<24}{:<24}{:<24}{:<24}{:<24}\n'.format('#Part', 'Cond Dens [sigma^-3]', 'Err Cond [sigma^-3]', 'Dil Dens [sigma^-3]', 'Err Dil [sigma^-3]', 'Rad Side [sigma]', 'Cond Surf [sigma]'))

with open(f'{folder}/FittingParams_A-Bin{binning}.dat', 'w') as fo:
      fo.write('{:<6} {:<24}{:<24}{:<24}{:<24}\n'.format('#Part', 'Cond Dens [sigma^-3]', 'Dil Dens [sigma^-3]', 'Rad Side [sigma]', 'Cond Surf [sigma]'))

for N in range(Nparts):
      print(f'Part {N+1}')
      
      Begin = DensFrames + N * 1000
      End = DensFrames + (N + 1) * 1000
      print(Begin, End)
      
      HISTC, BINSC, HISTA, BINSA, cluster_size, CM_ts = ComputeDensity(traj, binning, Begin, End)
      
      ##########
      # Fitting C density

      fig, (ax, bx) = plt.subplots(nrows=1,ncols=2, figsize=(10,5), sharey=True)

      ax.set_title('Binder Density', fontsize=20)

      with open( f'{folder}/Bins-DensityFrames-Bin{binning}-part{N+1}.dat', 'w') as fo:
            for B in range(len(BINSC)):
                  if B < len(BINSC):
                        fo.write('{:<10}'.format(np.around(BINSC[B], 3)))
                  else:
                        fo.write('{:<10}\n'.format(np.around(BINSC[B], 3)))

      avg_histoC = np.mean(HISTC, axis=0)
      with open(f'{folder}/Bins-DensityFrames-Bin{binning}-part{N+1}.dat', 'a') as fo:
            for h in HISTC:
                  plt.plot(np.array(BINSC), h, alpha=0.4, color='black', linewidth=0.6)
                  for H in range(len(h)):
                        if H < len(h):
                              fo.write('{:<10}'.format(np.around(h[H], 3)))
                        else:
                              fo.write('{:<10}\n'.format(np.around(h[H], 3)))

      dens_profileC = np.vstack((BINSC, avg_histoC)).T

      nbins = int((side / 10) // 2 // binning // 2) # half_side [nm] / bin_width / 2 (all integer)
      
      bounds=([0.5*dens_profileC[0,1], 0.5 * np.mean([dens_profileC[-i,1] for i in range(1, 10)]), 0, 0], 
            [1.5*dens_profileC[0,1], 1.5 * np.mean([dens_profileC[-i,1] for i in range(1, 10)]), side, side])

      fit_paramsC, fit_pcovC = curve_fit(FitDensityProfile, dens_profileC[:,0], dens_profileC[:,1], 
                                         bounds=([0.5*dens_profileC[0,1], 0, 0, 0], 
                                                 [1.5*dens_profileC[0,1], 0.1 * dens_profileC[0,1], side, side]))
      dens_condC, dens_dilC, Z_lC, surfC = fit_paramsC[0], fit_paramsC[1], fit_paramsC[2], fit_paramsC[3]

      fit_errC = np.sqrt(np.diag(fit_pcovC))
      err_dens_condC, err_dens_dilC = fit_errC[0], fit_errC[1]
      print(fit_paramsC)

      x_fit = np.linspace(start=0, stop=max(BINSC), num=5000)
      ax.plot(np.array(BINSC), avg_histoC, linewidth=4, color='tab:blue', label='Average')
      ax.plot(x_fit, FitDensityProfile(x_fit, *fit_paramsC), color='tab:red', linestyle='dotted', linewidth=4, markersize=0, label='Fit')

      with open(f'{folder}/densityC-Bin{binning}-part{N+1}.dat', 'w') as fo:
            for B, H in zip(BINSC, avg_histoC):
                  fo.write('{:<15}{:<15}\n'.format(format(B, '.3E'), format(H, '.3E')))

      with open(f'{folder}/FittingParams_C-Bin{binning}.dat', 'a') as fo:
            fo.write('{:<6} {:<24}{:<24}{:<24}{:<24}{:<24}{:<24}\n'.format(N+1,
                                                         format(dens_condC, '.3E'),
                                                         format(err_dens_condC, '.3E'),
                                                         format(dens_dilC, '.3E'),
                                                         format(err_dens_dilC, '.3E'),
                                                         np.around(Z_lC, 2),
                                                         np.around(surfC, 2)))
      
      ##########
      # Fitting A density
      bx.set_title('Polymer beads Density', fontsize=20)

      avg_histoA = np.mean(HISTA, axis=0)
      for h in HISTA:
            plt.plot(np.array(BINSA), h, alpha=0.4, color='black')
      dens_profileA = np.vstack((BINSA, avg_histoA)).T
      fit_paramsA, fit_pcovA = curve_fit(FitDensityProfile, dens_profileA[1:,0], dens_profileA[1:,1], 
                                         bounds=([0.5*dens_profileA[0,1], 1E-5*dens_profileA[0,1], 0, 0], 
                                                 [1.5*dens_profileA[0,1], 1E-2*dens_profileA[0,1], side, side]))
      dens_condA, dens_dilA, Z_lA, surfA = fit_paramsA[0], fit_paramsA[1], fit_paramsA[2], fit_paramsA[3]

      with open(f'{folder}/FittingParams_A-Bin{binning}.dat', 'a') as fo:
            fo.write('{:<6} {:<24}{:<24}{:<24}{:<24}\n'.format(N+1,
                                                              format(dens_condA, '.3E'),
                                                         format(dens_dilA, '.3E'),
                                                         np.around(Z_lA, 2),
                                                         np.around(surfA, 2)))

      x_fit = np.linspace(start=0, stop=max(BINSC), num=5000)

      bx.plot(np.array(BINSA), avg_histoA, linewidth=4, color='tab:blue', label='Average')
      bx.plot(x_fit, FitDensityProfile(x_fit, *fit_paramsA), color='tab:red', linestyle='dotted', linewidth=4, markersize=0, label='Fit')

      bx.legend()
      fig.savefig(f'{folder}/density-Bin{binning}-part{N+1}.png', dpi=300, bbox_inches='tight')

      ##########
      # Writing to output
      with open(f'{folder}/densityA-Bin{binning}-part{N+1}.dat', 'w') as fo:
            for B, H in zip(BINSA, avg_histoA):
                  fo.write('{:<15}{:<15}\n'.format(format(B, '.3E'), format(H, '.3E')))

