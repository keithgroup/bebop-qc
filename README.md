# BEBOP

The bond energy/bond order population ([BEBOP](https://doi.org/10.1021/acs.jctc.2c00334)) program is a computational chemistry algorithm that computes accurate molecular energies at equilibrium and bond energies using well-conditioned Hartree-Fock orbital populations and bond orders from approximate quantum chemistry methods.

## Installation 

```bash
git clone https://github.com/keithgroup/bebop-qc
cd bebop-qc
pip install .
```

## Preparing BEBOP Input Files

BEBOP requires output from the MinPop algorithm when runnning Hartree-Fock. Below is an example of how to do this in Gaussian 16.

1. Optimize your molecular structure using your preferred level of theory (e.g., B3LYP/CBSB7).

    ```# Opt B3LYP/CBSB7```

2. Run Hartree-Fock on the optimized structure in Gaussian using these keywords:

    ```# SP ROHF/CBSB3 Pop=(Full) IOp(6/27=122)```

## Usage

Execute ``bebop``. Run this in the command line:

```bash
bebop -f {name_file} --be --sort --json > {name_file}.bop
```

where ``{name_file}`` is the Hartree-Fock Gaussian output file.

Some examples of BEBOP output files are found in the ``examples`` directory.

Some details of the parsers used in ``bebop`` source code.

```bash
$ bebop -h
usage: bebop [-h] -f F [--be] [--sort] [--json]

compute BEBOP atomization energies and bond energies (i.e., gross and net)

optional arguments:
  -h, --help  show this help message and exit
  -f F        name of the Gaussian Hartree-Fock output file
  --be        compute BEBOP bond energies (net and gross bond energies)
  --sort      sort the net BEBOP bond energies (from lowest to highest in energy)
  --json      save the job output into JSON
```

## Citation

If you use BEBOP in your research, please cite:

Barbaro Zulueta, Sonia V. Tulyani, Phillip R. Westmoreland, Michael J. Frisch, E. James Petersson, George A. Petersson, and John A. Keith
Journal of Chemical Theory and Computation **2022** *18* (8), 4774-4794
DOI: 10.1021/acs.jctc.2c00334

## License

Distributed under the MIT License.
See `LICENSE` for more information.
