# tb-bdq-catalogue

This repository contains all data used and all figures generated in the study to build bedaquilne catalogues containing minor alleles.

---

The results of the study can be found in the Results.ipynb notebook, presented in a similar order and structure as in the paper.

---

All generated catalogues can be found in the catalogues folder and can all be parsed by Piezo. Those built at varying FRS thresholds and containing all BDQ candidate genes are named accordingly. i.e The catalogue built at FRS = 0.25 is catalogue_FRS_25.csv. Metadata includes the R, S, and background counts used in the Fisher test. Catalogues built using cross validation are in the cv directory, and catalogues built without mmpL5 and mmpS5 exist in the rem_mmpL5 directory.

---

The catalogue building psuedo-heuristic algorithm can be found in protocols/BuildCatalogue.py.

---

Predictions using piezo require functions in protocols/Predict.py
