# HPC Coursework

An implementation of a 2D lattice Boltzmann method using MPI.

* Source code is in the `d2q9-bgk.c` file
* Report is in `report.pdf`
* Results checking scripts are in the `check/` directory

## Compiling and running

To compile type `make`. The completed version of the coursework requires an MPI enabled version of ICC.

Input parameter and obstacle files are all specified on the command line of the `d2q9-bgk` executable.

Usage:

    $ ./d2q9-bgk <paramfile> <obstaclefile>
eg:

    $ ./d2q9-bgk input_256x256.params obstacles_256x256.dat

## Checking results

An automated result checking function is provided that requires you to load a particular Python module (`module load languages/anaconda2/5.0.1`). Running `make check` will check the output file (average velocities and final state) against some reference results. By default, it should look something like this:

    $ make check
    python check/check.py --ref-av-vels-file=check/128x128.av_vels.dat --ref-final-state-file=check/128x128.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat
    Total difference in av_vels : 5.270812566515E-11
    Biggest difference (at step 1219) : 1.000241556248E-14
      1.595203170657E-02 vs. 1.595203170658E-02 = 6.3e-11%

    Total difference in final_state : 5.962977334129E-11
    Biggest difference (at coord (6,2)) : 1.000588500943E-14
      3.329122639178E-02 vs. 3.329122639179E-02 = 3e-11%

    Both tests passed!

This script takes both the reference results and the results to check (both average velocities and final state). This is also specified in the makefile and can be changed like the other options:

    $ make check REF_AV_VELS_FILE=check/128x256.av_vels.dat REF_FINAL_STATE_FILE=check/128x256.final_state.dat
    python check/check.py --ref-av-vels-file=check/128x256.av_vels.dat --ref-final-state-file=check/128x256.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat
    ...

All the options for this script can be examined by passing the --help flag to it.

    $ python check/check.py --help
    usage: check.py [-h] [--tolerance TOLERANCE] --ref-av-vels-file
                    REF_AV_VELS_FILE --ref-final-state-file REF_FINAL_STATE_FILE
    ...

# Visualisation

You can view the final state of the simulation by creating a .png image file using a provided Gnuplot script:

    $ gnuplot final_state.plt
