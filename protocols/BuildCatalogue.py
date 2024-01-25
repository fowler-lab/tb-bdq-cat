import pandas as pd
from scipy import stats
import json


class BuildCatalogue:
    """
    Class for building a mutations catalogue using a Fisher's test on lone-occuring
    mutations.


    N.B - this is compatible with CRyPTIC tables only (merges made on 'UNIQUEID', for example) - but can
    easily be adapted
    """

    def __init__(self, samples, mutations, FRS_threshold, hardcoded=None):
        # apply FRS threshold to mutations
        mutations = mutations[(mutations.FRS >= FRS_threshold)]

        self.S, self.R, self.U = [], [], []

        #hardcode variant classifications - often helps to seed with phylogenetic mutations
        if hardcoded:
            for k,v in hardcoded.items():
                if v == 'S':
                    self.S.append({'mut':k, 'evid':{}})
                
        self.run = True

        while self.run:
            # while there are susceptible solos, call susceptible and remove
            self.build_S_arr(samples, mutations)

        # once the method gets jammed (ie no more susceptible solo mutations),
        # call all remaining solos (R and U) if there are any
        self.mop_up(samples, mutations)

        # build catalogue object from phenotype arrays
        self.catalogue = self.construct_catalogue()

    def build_S_arr(self, samples, mutations):

        # remove mutations predicted as susceptible from df (to potentially proffer additional, effective solos)
        mutations = mutations[~mutations.GENE_MUT.isin(i["mut"] for i in self.S)]

        # left join phenotypes with mutations
        joined = pd.merge(samples, mutations, on=["UNIQUEID"], how="left")
        # extract samples with only 1 mutation
        solos = joined.groupby("UNIQUEID").filter(lambda x: len(x) == 1)

        # method is jammed - end here.
        if len(solos) == 0:
            self.run = False

        s_iters = 0
        # for non WT or synonymous mutations
        for mut in solos[~solos.GENE_MUT.isna()].GENE_MUT.unique():
            # determine phenotype of mutation using Fisher's test
            pheno = self.fisher_binary(solos, mut)
            if pheno["pred"] == "S":
                # if susceptible, add mutation to phenotype array
                self.S.append({"mut": mut, "evid": pheno["evid"]})
                s_iters += 1

        if s_iters == 0:
            # if no susceptible solos (ie jammed) - move to mop up
            self.run = False

    def mop_up(self, samples, mutations):
        # remove mutations predicted as susceptible from df (to potentially proffer additional, effective solos)
        mutations = mutations[~mutations.GENE_MUT.isin(i["mut"] for i in self.S)]

        # left join phenotypes with mutations
        joined = pd.merge(samples, mutations, on=["UNIQUEID"], how="left")

        # extract samples with only 1 mutation
        solos = joined.groupby("UNIQUEID").filter(lambda x: len(x) == 1)

        # for non WT or synonymous mutations
        for mut in solos[~solos.GENE_MUT.isna()].GENE_MUT.unique():
            # determine phenotype of mutation using Fisher's test and add mutation to phenotype array (should be no S)
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

        # build contingency table ((R count, S count), (background R count, background S count))
        data = [[R_count, S_count], [R_count_no_mut, S_count_no_mut]]

        _, p_value = stats.fisher_exact(data)
            
        if p_value < 0.05 or solos[solos.GENE_MUT == mut].PHENOTYPE.nunique() == 1:
            # if variant frequency is 1 simply call the phenotype, otherwise call phenotype at 95% confidence
            if R_count > S_count:
                return {
                    "pred": "R",
                    "evid": [[R_count, S_count], [R_count_no_mut, S_count_no_mut], [p_value, _]],
                }
            else:
                return {
                    "pred": "S",
                    "evid": [[R_count, S_count], [R_count_no_mut, S_count_no_mut], [p_value, _]],
                }
        else:
            return {
                "pred": "U",
                "evid": [[R_count, S_count], [R_count_no_mut, S_count_no_mut], [p_value, _]],
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

    def return_piezo(
        self,
        genbank_ref,
        catalogue_name,
        version,
        drug,
        wildcards,
        grammar="GARC1",
        values="RUS",
    ):
        # insert piezo wildcards into catalogue object
        self.insert_wildcards(wildcards)

        piezo = (
            pd.DataFrame.from_dict(self.catalogue, orient="index")
            .reset_index()
            .rename(
                columns={"index": "MUTATION", "pred": "PREDICTION", "evid": "EVIDENCE", "p":"p_value"}
            )
        )
        piezo["GENBANK_REFERENCE"] = genbank_ref
        piezo["CATALOGUE_NAME"] = catalogue_name
        piezo["CATALOGUE_VERSION"] = version
        piezo["CATALOGUE_GRAMMAR"] = grammar
        piezo["PREDICTION_VALUES"] = values
        piezo["DRUG"] = drug
        piezo["SOURCE"] = json.dumps({})
        piezo["EVIDENCE"] = [
            json.dumps(
                {
                    "solo_R": i[0][0],
                    "solo_S": i[0][1],
                    "background_R": i[1][0],
                    "background_S": i[1][1],
                    "p_value":i[2][0]
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
