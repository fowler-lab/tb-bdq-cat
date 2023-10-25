import pandas as pd
from scipy import stats


class BuildCatalogue:
    def __init__(self, samples, mutations, FRS_threshold):
        mutations = mutations[(mutations.FRS >= FRS_threshold)]

        self.S, self.R, self.U = [], [], []

        self.run = True

        while self.run:
            self.build_S_arr(samples, mutations)

        self.mop_up(samples, mutations)

        self.catalogue = self.construct_catalogue()

    def build_S_arr(self, samples, mutations):
        mutations = mutations[~mutations.GENE_MUT.isin(self.S)]

        joined = pd.merge(samples, mutations, on=["UNIQUEID"], how="left")

        solos = joined.groupby("UNIQUEID").filter(lambda x: len(x) == 1)

        if len(solos) == 0:
            self.run = False

        s_iters = 0
        # for non WT or synonymous mutations
        for mut in solos[~solos.GENE_MUT.isna()].GENE_MUT.unique():
            pheno = self.fisher_binary(solos, mut)
            if pheno == "S":
                self.S.append(mut)
                s_iters += 1

        if s_iters == 0:
            self.run = False

    def mop_up(self, samples, mutations):
        mutations = mutations[~mutations.GENE_MUT.isin(self.S)]

        joined = pd.merge(samples, mutations, on=["UNIQUEID"], how="left")

        solos = joined.groupby("UNIQUEID").filter(lambda x: len(x) == 1)

        for mut in solos[~solos.GENE_MUT.isna()].GENE_MUT.unique():
            pheno = self.fisher_binary(solos, mut)
            if pheno == "R":
                self.R.append(mut)
            elif pheno == "U":
                self.U.append(mut)

    def fisher_binary(self, solos, mut):
        R_count = len(solos[(solos.PHENOTYPE == "R") & (solos.GENE_MUT == mut)])
        S_count = len(solos[(solos.PHENOTYPE == "S") & (solos.GENE_MUT == mut)])

        R_count_no_mut = len(solos[(solos.GENE_MUT.isna()) & (solos.PHENOTYPE == "R")])
        S_count_no_mut = len(solos[(solos.GENE_MUT.isna()) & (solos.PHENOTYPE == "S")])

        data = [[R_count, S_count], [R_count_no_mut, S_count_no_mut]]

        print(mut, [R_count, S_count], [R_count_no_mut, S_count_no_mut])

        _, p_value = stats.fisher_exact(data)

        print(mut + "p_value", p_value)

        if p_value < 0.05 or solos[solos.GENE_MUT == mut].PHENOTYPE.nunique() == 1:
            if R_count > S_count:
                return "R"
            else:
                return "S"
        else:
            return "U"

    def return_catalogue(self):
        return self.catalogue

    def construct_catalogue(self):
        catalogue = {}
        for i in self.S:
            catalogue[i] = "S"
        for i in self.R:
            catalogue[i] = "R"
        for i in self.U:
            catalogue[i] = "U"

        return catalogue
