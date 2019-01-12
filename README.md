# (Polytech) HPC Project: Parallel Simulation of Neutrons
This repository contains my work for the HPC class I took at the French school of Engineering Polytech Sorbonne, taught by Pierre Fortin and Lokmane Abbas Turki.

The goal was to create a parallel version of a sequential code that studies the movement of neutrons through a thin plate. Each neutron is thrown against the plate and can end up in one of the following 3 states: reflected by the plate, absorbed by the plate or transmitted through the plate. We want to study the probability each case has of occurring thanks to the Monte Carlo method. Basically, the idea is to simulate the situation for a large number of neutrons and then look at the average results.

I started from the sequential version written by our teacher and applied 3 tools to turn it into a parallel version:
- the CUDA library to have a (mono) GPU version
- the MPI and OpenMP libraries to have a parallelized CPU version

## Some info about the repository
This repository contains the report and the slides I wrote (in French) for this projet and the various scripts with each parallelization method.
