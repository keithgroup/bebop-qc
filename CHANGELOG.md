# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2022-08-16

- Initial release!

## [2.0.0] - 2024-03-30

- Changing ``bebop`` folder to ``bebop1``
- Changed the MIT license date from 2022 to 2024 for bebop1
- Modified line 246 in ``bebop1/read_output.py`` from ``l[11:20].replace(' ','')``  to ``l[11:20].replace(' ','').replace('\n',''))``
- Changed name and version in bebop1 to 1.0.1
- Created  bebop2 folder containing python scripts and examples
- Move all the examples in version 1.0.0 to ``bebop1_examples``
- Created ``bebop2_examples`` containing BEBOP-2 examples
- Modify bebop-qc version from 1.0.0 to 2.0.0
- Change README.md file to include information on BEBOP-1 and BEBOP-2 usages and citations 

## [2.0.0] - 2024-05-09
-Modified line 382 of ``bebop2/read_output.py`` from ``p = ROHF.readlines()[93:]`` to ``p = ROHF.readlines()``