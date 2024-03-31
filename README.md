# BEBOP

The bond energy/bond order population ([BEBOP](https://doi.org/10.1021/acs.jctc.2c00334)) program is a computational chemistry algorithm that computes accurate molecular energies at equilibrium and bond energies using well-conditioned Hartree-Fock orbital populations and bond orders from approximate quantum chemistry methods.

## Installation

```bash
git clone https://github.com/keithgroup/bebop-qc
cd bebop-qc
pip install .
```

## Preparing BEBOP Input Files

Currently, we have developed BEBOP-1 and BEBOP-2 models, and each of these models require output from the MinPop algorithm when running Hartree-Fock. Below is an example of how to do this in Gaussian 16 for each model.

1. Optimize your molecular structure using your preferred level of theory (e.g., B3LYP/CBSB7 and B3LYP/cc-pVTZ+1d).

##### BEBOP-1:

    # Opt B3LYP/CBSB7

##### BEBOP-2:

    # Opt B3LYP/cc-pVTZ+1d

2. Run Hartree-Fock on the optimized structure in Gaussian.

##### BEBOP-1 and BEBOP-2:

    # SP ROHF/CBSB3 Pop=(Full) IOp(6/27=122,6/12=3)

## Usage

Execute `bebop1` and `bebop2` for BEBOP-1 and BEBOP-2, respectively. Run this in the command line:

##### BEBOP-1:

```bash
bebop1 -f {name_file} --be --sort --json > {name_file}.bop
```

##### BEBOP-2

```bash
bebop2 -f {name_file} -parameter_folder {parameter_folder} --be --sort --json --ionicity > {name_file}.bop
```

where `{name_file}` is the Hartree-Fock Gaussian output file and `{parameter_folder}` is the name or path of the BEBOP-2 parameter folder. Note that BEBOP-2 parameters are stored in json files under the `opt_parameters` folders.

Some examples of BEBOP-1 and BEBOP-2 output files are found in the `examples` directory.

Some details of the parsers used in `bebop1` and `bebop2` source codes.

##### BEBOP-1:

```bash
$ bebop1 -h
usage: bebop1 [-h] -f F [--be] [--sort] [--json]

compute BEBOP atomization energies and bond energies (i.e., gross and net)

optional arguments:
  -h, --help  show this help message and exit
  -f F        name of the Gaussian Hartree-Fock output file
  --be        compute BEBOP bond energies (net and gross bond energies)
  --sort      sort the net BEBOP bond energies (from lowest to highest in energy)
  --json      save the job output into JSON
```

##### BEBOP-2:

```bash
$ bebop2 -h
usage: bebop2 [-h] -f F [-param_folder PARAM_FOLDER] [--be] [--sort] [--json] [--ionicity]

compute BEBOP atomization energies and bond energies (i.e., gross and net)

optional arguments:
  -h, --help            show this help message and exit
  -f F                  name of the Gaussian Hartree-Fock output file
  -param_folder PARAM_FOLDER
                        name of BEBOP-2's parameter path/folder (default: opt_parameters)
  --be                  compute BEBOP bond energies (net and gross bond energies)
  --sort                sort the net BEBOP bond energies (from lowest to highest in energy)
  --json                save the job output into JSON
  --ionicity            percent of ionicity per each bond and entire molecule
```

## Citations

Please cite:

**BEBOP-1**: Barbaro Zulueta, Sonia V. Tulyani, Phillip R. Westmoreland, Michael J. Frisch, E. James Petersson, George A. Petersson, and John A. Keith
Journal of Chemical Theory and Computation **2022** _18_ (8), 4774-4794
DOI: 10.1021/acs.jctc.2c00334

**BEBOP-2**: Barbaro Zulueta, George A. Petersson, and John A. Keith; Many-Body Bond and Charge Model Contributions to BEBOP Model **2024** (_in preparation_).

## License

Distributed under the MIT License.
See `LICENSE` for more information.
