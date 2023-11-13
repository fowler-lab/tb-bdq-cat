import pandas as pd
from scipy import stats
import json


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
        mutations = mutations[~mutations.GENE_MUT.isin(i["mut"] for i in self.S)]

        joined = pd.merge(samples, mutations, on=["UNIQUEID"], how="left")

        solos = joined.groupby("UNIQUEID").filter(lambda x: len(x) == 1)

        if len(solos) == 0:
            self.run = False

        s_iters = 0
        # for non WT or synonymous mutations
        for mut in solos[~solos.GENE_MUT.isna()].GENE_MUT.unique():
            pheno = self.fisher_binary(solos, mut)
            if pheno["pred"] == "S":
                self.S.append({"mut": mut, "evid": pheno["evid"]})
                s_iters += 1

        if s_iters == 0:
            self.run = False

    def mop_up(self, samples, mutations):
        mutations = mutations[~mutations.GENE_MUT.isin(i["mut"] for i in self.S)]

        joined = pd.merge(samples, mutations, on=["UNIQUEID"], how="left")

        solos = joined.groupby("UNIQUEID").filter(lambda x: len(x) == 1)

        for mut in solos[~solos.GENE_MUT.isna()].GENE_MUT.unique():
            pheno = self.fisher_binary(solos, mut)
            if pheno["pred"] == "R":
                self.R.append({"mut": mut, "evid": pheno["evid"]})
            elif pheno["pred"] == "U":
                self.U.append({"mut": mut, "evid": pheno["evid"]})

    def fisher_binary(self, solos, mut):
        R_count = len(solos[(solos.PHENOTYPE == "R") & (solos.GENE_MUT == mut)])
        S_count = len(solos[(solos.PHENOTYPE == "S") & (solos.GENE_MUT == mut)])

        R_count_no_mut = len(solos[(solos.GENE_MUT.isna()) & (solos.PHENOTYPE == "R")])
        S_count_no_mut = len(solos[(solos.GENE_MUT.isna()) & (solos.PHENOTYPE == "S")])

        data = [[R_count, S_count], [R_count_no_mut, S_count_no_mut]]

        _, p_value = stats.fisher_exact(data)

        if p_value < 0.05 or solos[solos.GENE_MUT == mut].PHENOTYPE.nunique() == 1:
            if R_count > S_count:
                return {
                    "pred": "R",
                    "evid": [[R_count, S_count], [R_count_no_mut, S_count_no_mut]],
                }
            else:
                return {
                    "pred": "S",
                    "evid": [[R_count, S_count], [R_count_no_mut, S_count_no_mut]],
                }
        else:
            return {
                "pred": "U",
                "evid": [[R_count, S_count], [R_count_no_mut, S_count_no_mut]],
            }

    def return_catalogue(self):
        return {
            mutation: {"PHENOTYPE": data["pred"]}
            for mutation, data in self.catalogue.items()
        }

    def construct_catalogue(self):
        catalogue = {}
        for i in self.S:
            catalogue[i["mut"]] = {"pred": "S", "evid": i["evid"]}
        for i in self.R:
            catalogue[i["mut"]] = {"pred": "R", "evid": i["evid"]}
        for i in self.U:
            catalogue[i["mut"]] = {"pred": "U", "evid": i["evid"]}

        return catalogue

    def insert_wildcards(self, wildcards):
        self.catalogue = {**self.catalogue, **wildcards}

    def return_piezo(self, genbank_ref, catalogue_name, version, drug, wildcards):
        """Formats a catalogue in a piezo compatible format"""
        self.insert_wildcards(wildcards)

        piezo = (
            pd.DataFrame.from_dict(self.catalogue, orient="index")
            .reset_index()
            .rename(
                columns={"index": "MUTATION", "pred": "PREDICTION", "evid": "EVIDENCE"}
            )
        )
        piezo["GENBANK_REFERENCE"] = genbank_ref
        piezo["CATALOGUE_NAME"] = catalogue_name
        piezo["CATALOGUE_VERSION"] = version
        piezo["CATALOGUE_GRAMMAR"] = "GARC1"
        piezo["PREDICTION_VALUES"] = "RUS"
        piezo["DRUG"] = drug
        piezo["SOURCE"] = json.dumps({})
        piezo["EVIDENCE"] = [
            json.dumps(
                {
                    "solo_R": i[0][0],
                    "solo_S": i[0][1],
                    "background_R": i[1][0],
                    "background_S": i[1][1],
                }
            )
            if i
            else json.dumps({})
            for i in piezo["EVIDENCE"]
        ]
        piezo["OTHER"] = json.dumps({})

        piezo = piezo[
            [
                "GENBANK_REFERENCE",
                "CATALOGUE_NAME",
                "CATALOGUE_VERSION",
                "CATALOGUE_GRAMMAR",
                "PREDICTION_VALUES",
                "DRUG",
                "MUTATION",
                "PREDICTION",
                "SOURCE",
                "EVIDENCE",
                "OTHER",
            ]
        ]

        return piezo
