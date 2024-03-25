"""The WHO catalogues use aggregated mutations. Let's try and add in functionality so we can do the same (should reduce the number of false negatives and allow for a direct comparison).

In the case of frameshifts and premature stops, for example, I see there being 2 levels at which the aggregation could happen. Position specific, or position agnostic. E.g we could aggregate frameshifts that occur at the same position, or we could aggregate all frameshifts (as those near the end of the chain may not do too much). We should ensure both are possible.

The easiest way to do this is probably using wildcards, and using piezo to read them.

We can either aggregate before classification, or after (as each row's evidence is in the catalogue). What would the difference be? Well, the former could potentially affect downstream classifications while the latter would be retrospective, and therefore wouldn't. However, if we do before then the method would encounter a variant and classify it potentially without access to other variants that fall under that aggregate umbrella - that's not good. So we should classify each variant individually and aggregate afterwards.

Problem with aggregating before classication is that regardless of aggregation there will only be a few solos from a particular gene to begin with, which will then determine the fate of the all the other mutations that fall under the aggregation rule.   

Thought process:   
We aggregate aftwards and then decide a phenotype. There are a few ways we could this - the simplest being if all mutations have the same phenotype, just call that phenotype for the aggregate. But we could rather run a Fisher's test. The next question is what is the contigency table made up of? We could use R vs S counts of mutations in the catalogue, but then you lose the solo distributions/any measure of frequency. So instead, we can aggregate the solo counts (from the evidence column in the catalogue). But then what about the background? I think the most logical is to take the mutation classified last's background rate - the point at which the last (aggregate) mutation was catalogued is effectively the same point at which clustering is done. So we should use this background, irrespective of those rates developed after the last relevant mutation was classified."""

import os
import json
import piezo
from scipy import stats
import pandas as pd


class BuildCatalogue:
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
        # Build contingency table
        R_count, S_count, R_count_no_mut, S_count_no_mut, data = self.build_contigency(solos, mut)
        _, p_value = stats.fisher_exact(data)

        # Determine prediction based on mutation's presence and significance level
        phenotype = solos[solos.GENE_MUT == mut].PHENOTYPE
        if phenotype.nunique() == 1:
            prediction = "R" if R_count > S_count else "S"
        elif p_value < 0.05:
            OR = (R_count * S_count_no_mut) / (S_count * R_count_no_mut)
            prediction = "R" if OR > 1 else "S"
            _ = OR
        else:
            prediction = "U"

        # Evidence structure
        evidence = [
            [R_count, S_count],
            [R_count_no_mut, S_count_no_mut],
            [p_value, _]
        ]

        return {"pred": prediction, "evid": evidence}

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
        column_mappings = {
            "index": "MUTATION",
            "pred": "PREDICTION",
            "evid": "EVIDENCE",
            "p": "p_value",
        }

        data = aggregates if aggregates else self.catalogue

        data = BuildCatalogue.insert_wildcards(data, self.wildcards)

        piezo = (
            pd.DataFrame.from_dict(data, orient="index")
            .reset_index()
            .rename(columns=column_mappings)
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

    def update(self, rules):
        # for now we will use piezo to understand the expert rule
        # therefore need to save the rules in a piezo table csv, which will be read in
        # then iteratre through each var in the catalogue and aggregate (if aggregate rule) and classify

        if not os.path.exists("./temp"):
            os.makedirs("./temp")

        # aggregate variants by one rule at a time
        for rule, phenotype in rules.items():
            # use R as a flag and save rule for piezo to read back in

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
            piezo_rule = piezo.ResistanceCatalogue("./temp/aggregate_rule.csv")

            if "*" in rule:
                # if an aggregation rule
                aggregated_vars = {
                    k: v["evid"]
                    for k, v in self.catalogue.items()
                    if k not in self.wildcards.keys()
                    and k not in rules
                    and piezo_rule.predict(k)[self.drug] == "R"
                }

                evid = self.aggregate_contigency(aggregated_vars)
                self.catalogue = {
                    **{rule: {"pred": phenotype, "evid": evid}},
                    **self.catalogue,
                }

                for k in aggregated_vars.keys():
                    self.catalogue.pop(k, None)
            else:
                # if not an aggregation rule
                self.catalogue.pop(rule, None)
                # remove old entry and replace with rule
                self.catalogue = {
                    **{rule: {"pred": phenotype, "evid": None}},
                    **self.catalogue,
                }

        os.remove("./temp/aggregate_rule.csv")

        return self

    def aggregate_contigency(self, aggregated_variants):
        aggregated_contigency = [[0, 0], [0, 0], [None, None]]
        for tables in aggregated_variants.values():
            aggregated_contigency[0][0] += tables[0][0]
            aggregated_contigency[0][1] += tables[0][1]
            aggregated_contigency[1][0] = max(aggregated_contigency[1][0], tables[1][0])
            aggregated_contigency[1][1] = max(aggregated_contigency[1][1], tables[1][1])

        return aggregated_contigency
