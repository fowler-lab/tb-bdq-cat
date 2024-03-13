import pandas as pd
from scipy import stats
import json
import os
import piezo

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


class BuildCatalogueAggregate:
    """
    Class for building a mutations catalogue using a Fisher's test on lone-occuring
    mutations.


    N.B - this is compatible with CRyPTIC tables only (merges made on 'UNIQUEID', for example) - but can
    easily be adapted
    """

    def __init__(
        self,
        samples,
        mutations,
        FRS_threshold,
        hardcoded=None,
        aggregates=None,
    ):
        # apply FRS threshold to mutations
        mutations = mutations[(mutations.FRS >= FRS_threshold)]

        self.S, self.R, self.U = [], [], []

        # hardcode variant classifications - often helps to seed with phylogenetic mutations
        if hardcoded:
            for k, v in hardcoded.items():
                if v == "S":
                    self.S.append({"mut": k, "evid": {}})

        self.aggregates = aggregates

        self.run = True

        while self.run:
            # while there are susceptible solos, call susceptible and remove
            self.build_S_arr(samples, mutations)

        # once the method gets jammed (ie no more susceptible solo mutations),
        # call all remaining solos (R and U) if there are any
        self.mop_up(samples, mutations)

        # build catalogue object from phenotype arrays
        self.catalogue = self.construct_catalogue()

        self.aggregate_ = False

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

    def build_contigency(self, solos, mut):
        R_count = len(solos[(solos.PHENOTYPE == "R") & (solos.GENE_MUT == mut)])
        S_count = len(solos[(solos.PHENOTYPE == "S") & (solos.GENE_MUT == mut)])

        R_count_no_mut = len(solos[(solos.GENE_MUT.isna()) & (solos.PHENOTYPE == "R")])
        S_count_no_mut = len(solos[(solos.GENE_MUT.isna()) & (solos.PHENOTYPE == "S")])

        return (
            R_count,
            S_count,
            R_count_no_mut,
            S_count_no_mut,
            [[R_count, S_count], [R_count_no_mut, S_count_no_mut]],
        )

    def fisher_binary(self, solos, mut):

        # build contingency table ((R count, S count), (background R count, background S count))

        R_count, S_count, R_count_no_mut, S_count_no_mut, data = self.build_contigency(
            solos, mut
        )

        _, p_value = stats.fisher_exact(data)

        if p_value < 0.05 or solos[solos.GENE_MUT == mut].PHENOTYPE.nunique() == 1:
            # if variant frequency is 1 simply call the phenotype, otherwise call phenotype at 95% confidence
            if R_count > S_count:
                return {
                    "pred": "R",
                    "evid": [
                        [R_count, S_count],
                        [R_count_no_mut, S_count_no_mut],
                        [p_value, _],
                    ],
                }
            else:
                return {
                    "pred": "S",
                    "evid": [
                        [R_count, S_count],
                        [R_count_no_mut, S_count_no_mut],
                        [p_value, _],
                    ],
                }
        else:
            return {
                "pred": "U",
                "evid": [
                    [R_count, S_count],
                    [R_count_no_mut, S_count_no_mut],
                    [p_value, _],
                ],
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

    @staticmethod
    def insert_wildcards(catalogue, wildcards):
        return {**catalogue, **wildcards}

    def build_piezo(
        self,
        genbank_ref,
        catalogue_name,
        version,
        drug,
        wildcards,
        aggregates=None,
        grammar="GARC1",
        values="RUS",
    ):

        self.genbank_ref = genbank_ref
        self.catalogue_name = catalogue_name
        self.version = version
        self.drug = drug
        self.wildcards = wildcards

        # if not being used in the aggregation step
        if not aggregates:
            # insert piezo wildcards into catalogue object
            self.catalogue = BuildCatalogue.insert_wildcards(
                self.catalogue, self.wildcards
            )

            # construct piezo table df
            piezo = (
                pd.DataFrame.from_dict(self.catalogue, orient="index")
                .reset_index()
                .rename(
                    columns={
                        "index": "MUTATION",
                        "pred": "PREDICTION",
                        "evid": "EVIDENCE",
                        "p": "p_value",
                    }
                )
            )
        # if being used to create the aggregation table
        else:
            piezo = (
                pd.DataFrame.from_dict(aggregates, orient="index")
                .reset_index()
                .rename(
                    columns={
                        "index": "MUTATION",
                        "pred": "PREDICTION",
                        "evid": "EVIDENCE",
                        "p": "p_value",
                    }
                )
            )

        piezo["GENBANK_REFERENCE"] = self.genbank_ref
        piezo["CATALOGUE_NAME"] = self.catalogue_name
        piezo["CATALOGUE_VERSION"] = self.version
        piezo["CATALOGUE_GRAMMAR"] = grammar
        piezo["PREDICTION_VALUES"] = values
        piezo["DRUG"] = self.drug
        piezo["SOURCE"] = json.dumps({})
        piezo["EVIDENCE"] = [
            (
                json.dumps(
                    {
                        "solo_R": i[0][0],
                        "solo_S": i[0][1],
                        "background_R": i[1][0],
                        "background_S": i[1][1],
                        "p_value": i[2][0],
                    }
                )
                if i
                else json.dumps({})
            )
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

        self.piezo_catalogue = piezo

        return self

    def return_piezo(self):
        return self.piezo_catalogue

    def aggregate(self, aggregates):
        # for now we will use piezo to understand the aggregate rules
        # therefore need to save the rules in a piezo table csv, which will be read in
        # then iteratre through each var in the catalogue and aggregate and classify

        self.aggregate_ = True

        if not os.path.exists("./temp"):
            os.makedirs("./temp")

        # aggregate variants by one rule at a time
        for rule in aggregates:
            # use R as a flag and save this one rule as a piezo table

            self.build_piezo(
                self.genbank_ref,
                self.catalogue_name,
                self.version,
                self.drug,
                self.wildcards,
                aggregates=BuildCatalogue.insert_wildcards(
                    {rule: {"pred": "R", "evid": {}}}, self.wildcards
                ),
            ).return_piezo().to_csv("./temp/aggregate_rule.csv", index=False)
            # read rule back in with piezo
            aggregate_rule = piezo.ResistanceCatalogue("./temp/aggregate_rule.csv")

            aggregated_vars = {}


            for k, v in self.catalogue.items():
                if k not in self.wildcards.keys() and k not in aggregates:
                    if aggregate_rule.predict(k)[self.drug] == "R":
                        aggregated_vars[k] = v["evid"]

            prediction = self.fishers_aggregate(aggregated_vars)

            self.catalogue = {**{rule:prediction}, **self.catalogue}

            for k in aggregated_vars:
                self.catalogue.pop(k, None)

        return self

    def aggregate_contigency(self, aggregated_variants):
        aggregated_contigency = [[0, 0], [0, 0]]
        for mutation, tables in aggregated_variants.items():
            aggregated_contigency[0][0] += tables[0][0]
            aggregated_contigency[0][1] += tables[0][1]
            aggregated_contigency[1][0] = max(aggregated_contigency[1][0], tables[1][0])
            aggregated_contigency[1][1] = max(aggregated_contigency[1][1], tables[1][1])

        return aggregated_contigency

    def fishers_aggregate(self, aggregated_variants):

        x = self.aggregate_contigency(aggregated_variants)

        _, p_value = stats.fisher_exact(x)

        if p_value < 0.05 or (x[0][0] + x[0][1]) == 1 or (bool(x[0][0]) != bool(x[0][1])):
            # if variant frequency is 1 simply call the phenotype, otherwise call phenotype at 95% confidence
            if x[0][0] > x[0][1]:
                return {
                    "pred": "R",
                    "evid": [
                        [x[0][0], x[0][1]],
                        [x[1][0], x[1][1]],
                        [p_value, _],
                    ],
                }
            else:
                return {
                    "pred": "S",
                    "evid": [
                        [x[0][0], x[0][1]],
                        [x[1][0], x[1][1]],
                        [p_value, _],
                    ],
                }
        else:
            return {
                "pred": "U",
                "evid": [
                    [x[0][0], x[0][1]],
                    [x[1][0], x[1][1]],
                    [p_value, _],
                ],
            }