# DSC 291 - Large Scale Statistical Analysis SP22 Project

This repository hosts all documents used throughout the project. It follows the family-wise error (FWER) control procedure proposed in [List 2018][1]. The core idea is to use bootstrap method to estimate the distribution of p-values.

The codes in this repository is a Python implementation of the proposed procedure. [List 2019][1] has offered a Matlab and Stats implementation available [here][3]. 

The files in this repository is organized as follows:
1. `data` folder stores a example dataset on how matching grants change the donation behaviors of donors, as in [Karlan 2007][2]. `data_econ.csv` is the old version of data and `AMR merged.csv` is the updated version of the same data.
2. `proposal` and `final` folders store the figures, R scripts, report, and slides for presentation.
3. `mht_python` folder stores the Python implementation of the FWER control procedure. Please refer to the `README.md` file in the folder for how to use and run the codes.

Thank you for visiting this repository. Please contact me if there is any concern.

[1]: https://link.springer.com/article/10.1007/s10683-018-09597-5
[2]: https://www.aeaweb.org/articles?id=10.1257/aer.97.5.1774
[3]: https://github.com/seidelj/mht