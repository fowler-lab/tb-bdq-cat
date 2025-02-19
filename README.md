<div align="center">

# An improved catalogue for whole-genome sequencing prediction of bedaquiline resistance in M. tuberculosis using a reproducible algorithmic approach

</div>

This repository contains all data used and all figures generated in the study to build bedaquilne catalogues containing minor alleles.

---

To run the notebooks you will first have to clone the repository via

`git clone git@github.com:fowler-lab/tb-bdq-cat.git`

Navigate into project directory and setup an environment either with conda or pip:

`conda env create -f env.yml`

`conda activate BDQ_env`

or

`python -m venv bdq_venv`

`source bdq_venv/bin/activate` (On Windows, use: venv\Scripts\activate)

`pip install -r requirements.txt`

---

The `methods.ipynb`, `results.ipynb`, and `supplement.ipynb` notebooks correspond to the methods, sections, and supplementary sections in the manuscript. Notebook cells are presented in a similar order and structure as the manuscript.

---

All generated catalogues can be found in the catalogues folder and can all be parsed by Piezo. Catalogue naming (ie catomatic_1.csv) correspond to the catalogue names in the manuscript, and can be cross referenced.

---

All data used to build and test the catalogues can be found in the `data/` directory.
