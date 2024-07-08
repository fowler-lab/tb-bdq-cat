import pandas as pd
from Bio import SeqIO
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from protocols.Predict import predict
from protocols.BuildCatalogue import BuildCatalogue
from protocols import Helpers
import random


DNA_Codons = {
    # 'M' - START, '_' - STOP
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TGT": "C",
    "TGC": "C",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TTT": "F",
    "TTC": "F",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
    "CAT": "H",
    "CAC": "H",
    "ATA": "I",
    "ATT": "I",
    "ATC": "I",
    "AAA": "K",
    "AAG": "K",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATG": "M",
    "AAT": "N",
    "AAC": "N",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGA": "R",
    "AGG": "R",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "AGT": "S",
    "AGC": "S",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TGG": "W",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "!",
    "TAG": "!",
    "TGA": "!",
}


def parse_vcf(vcf_file, gene, database):
    index = 0
    Dict = {}
    with open(vcf_file, "r") as f:
        for line in f:
            if line[0] != "#":
                line = line.split("\t")
                line[0] = line[0].split("/")
                if database == "shaheed":
                    line[0][5] = line[0][5].split(".")
                    uniqueid = line[0][5][0:8]
                elif database == "CRyPTIC1":
                    line[0][6] = line[0][6].split(".")
                    uniqueid = line[0][6][0:8]

                uniqueid = "".join([i + "." for i in uniqueid])[:-1]
                variant = line[1]
                line[-1] = line[-1].split(":")
                if database == "shaheed":
                    FRS = line[-1][3]
                    RRS = line[-1][2].split(",")
                elif database == "CRyPTIC1":
                    FRS = line[-1][4]
                    RRS = line[-1][3].split(",")
                mutation = "".join(line[3:5])
                ref = line[3]
                alt = line[4]
                print(alt)

                Dict[index] = {
                    "UNIQUEID": uniqueid,
                    "GENE": gene,
                    "GENOME_INDEX": int(variant),
                    "GENETIC_REF": ref,
                    "GENETIC_ALT": alt,
                    "MUTATION": mutation,
                    "RRS_vcf": RRS,
                    "FRS_vcf": FRS,
                }
                index += 1

    df = pd.DataFrame.from_dict(Dict)

    return df


def FilterMultiplePhenos(group):
    """
    If a sample contains more than 1 phenotype,
    keep the resistant phenotype (preferably with MIC) if there is one.
    """
    if len(group) == 1:
        return group

    # Prioritize rows with 'R' phenotype
    prioritized_group = (
        group[group["PHENOTYPE"] == "R"] if "R" in group["PHENOTYPE"].values else group
    )

    # Check for rows with METHOD_MIC values
    with_mic = prioritized_group.dropna(subset=["METHOD_MIC"])
    return with_mic.iloc[0:1] if not with_mic.empty else prioritized_group.iloc[0:1]


def mic_to_float(arr):
    float_mic = []
    for i in arr:
        try:
            float_mic.append(float(i))
        except ValueError:
            try:
                float_mic.append(float(i[1:]))
            except ValueError:
                float_mic.append(float(i[2:]))

    return float_mic


def DNA_mut_to_aa(vcf_df, genes):
    """genes must be a dictionary with complement bool as values"""

    vcf_df = shift_genome_index(vcf_df, genes)

    mut_products = []
    for i in vcf_df.index:
        for gene in genes.keys():
            if vcf_df["GENE"][i] == gene:
                if len(vcf_df["GENETIC_MUTATION"][i]) > 2:
                    mut_products.append(str(vcf_df["PRODUCT_POSITION"][i]) + "_indel")
                else:
                    codon = extract_codons(gene, genes[gene])[1][
                        int(vcf_df["PRODUCT_POSITION"][i])
                    ]
                    pos_in_codon = vcf_df["GENOME_INDEX_SHIFTED"][i] % 3
                    codon[pos_in_codon] = vcf_df["GENETIC_MUTATION"][i][-1]
                    mut_products.append(
                        str(vcf_df["MUTATION"][i][:-1] + DNA_Codons["".join(codon)])
                    )

    vcf_df["MUTATION"] = mut_products
    return vcf_df


def shift_genome_index(vcf_df, genes):
    """genes must be a dictionary with complement bool as values"""

    genome_index_shifted = []
    for i in vcf_df.index:
        for gene in genes.keys():
            if vcf_df["GENE"][i] == gene:
                if genes[gene]:
                    genome_index_shifted.append(
                        np.max(extract_codons(gene, genes[gene])[0])
                        - vcf_df["GENOME_INDEX"][i]
                    )  # range is for the complement strand (the gene is 'backwards')
                else:
                    genome_index_shifted.append(
                        vcf_df["GENOME_INDEX"][i]
                        - np.min(extract_codons(gene, genes[gene])[0])
                    )

    vcf_df["GENOME_INDEX_SHIFTED"] = genome_index_shifted

    return vcf_df


def extract_codons(gene, complement=False):
    for record in SeqIO.parse(f"../../Data/genetic/{gene}.fasta", "fasta"):
        record = record

    try:
        idx_range = [int(i) for i in record.id.split(":")[1].split("-")]
    except ValueError:
        idx_range = [
            int(i[1:]) for i in record.id.split(":")[1].split("-")
        ]  # may have a 'c' before the number

    complements = {"A": "T", "T": "A", "C": "G", "G": "C"}
    if complement:
        seq = ""
        for i in record.seq:
            seq += complements[i]
        seq = [i for i in seq]
    else:
        seq = [i for i in record.seq]

    codons = []
    for i in range(0, len(seq), 3):
        codons.append(seq[i : i + 3])

    return idx_range, codons


def RSIsolateTable(df, genes):
    """returns df of number of isolates for each phenotype"""
    table = {}
    table["Total"] = {
        "R": df[df.PHENOTYPE == "R"].UNIQUEID.nunique(),
        "S": df[df.PHENOTYPE == "S"].UNIQUEID.nunique(),
        "Total": df.UNIQUEID.nunique(),
    }
    for i in genes:
        d = df[df.GENE == i]
        table[i] = {
            "R": d[d.PHENOTYPE == "R"].UNIQUEID.nunique(),
            "S": d[d.PHENOTYPE == "S"].UNIQUEID.nunique(),
            "Total": d[d.PHENOTYPE == "R"].UNIQUEID.nunique()
            + d[d.PHENOTYPE == "S"].UNIQUEID.nunique(),
        }

    return pd.DataFrame.from_dict(table).T


def RSVariantTable(df, genes):
    table = {}
    table["Total"] = {
        "R": df[df.PHENOTYPE == "R"].UNIQUEID.count(),
        "S": df[df.PHENOTYPE == "S"].UNIQUEID.count(),
        "Total": df.UNIQUEID.count(),
    }
    for i in genes:
        d = df[df.GENE == i]
        table[i] = {
            "R": d[d.PHENOTYPE == "R"].UNIQUEID.count(),
            "S": d[d.PHENOTYPE == "S"].UNIQUEID.count(),
            "Total": d[d.PHENOTYPE == "R"].UNIQUEID.count()
            + d[d.PHENOTYPE == "S"].UNIQUEID.count(),
        }

    return pd.DataFrame.from_dict(table).T


def CombinedDataTable(all):
    df = RSIsolateTable(all, all.GENE.unique())
    df1 = RSIsolateTable(all[all.FRS < 0.9], all.GENE.unique())
    df2 = RSVariantTable(all, all.GENE.unique())
    df3 = RSVariantTable(all[all.FRS < 0.9], all.GENE.unique())
    df = pd.concat([df, df1, df2, df3], axis=1)

    df.columns = pd.MultiIndex.from_tuples(
        zip(
            [
                "All",
                "",
                "",
                "Minor alleles",
                "",
                "",
                "All",
                "",
                "",
                "Minor alleles",
                "",
                "",
            ],
            df.columns,
        )
    )

    return df


def extract_solos(gene, df):
    solos, solo_ids = {}, []
    id_list = df.UNIQUEID.unique().tolist()
    for i in id_list:
        id_only = df[df.UNIQUEID == i]
        if (id_only.MUTATION.nunique() == 1) & (id_only.GENE.tolist()[0] == gene):
            solo_ids.append(i)
            if id_only.MUTATION.tolist()[0] in solos.keys():
                solos[id_only.MUTATION.tolist()[0]].append(
                    id_only.METHOD_MIC.tolist()[0]
                )
            else:
                solos[id_only.MUTATION.tolist()[0]] = id_only.METHOD_MIC.tolist()

    return solos, solo_ids


def extract_mics(gene, df, hz_thresh=3, output_counts=False, solo_ids=None):
    mut_counts = df[["GENE", "MUTATION"]].value_counts().reset_index(name="count")
    if output_counts:
        print(mut_counts)

    mic_dict = {}
    for i in mut_counts[
        (mut_counts["count"] >= hz_thresh) & (mut_counts.GENE == gene)
    ].index:
        mic_dict[mut_counts["MUTATION"][i]] = df[
            (df.GENE == gene)
            & (df.MUTATION == mut_counts["MUTATION"][i])
            & (~df.UNIQUEID.isin(solo_ids))
        ].METHOD_MIC.tolist()

    return mic_dict


def order_x(target_x, x):
    """Orders a second axis of discrete values relative to the first.
    Inserts blanks if that x value is unmatched"""

    ordered_x = {}
    no_overlap = {}
    for k in target_x.keys():
        if k in x.keys():
            ordered_x[k] = x[k]
        else:
            ordered_x[k] = []

    for k in x.keys():
        if k not in ordered_x.keys():
            ordered_x[k] = x[k]

    return ordered_x


def tabulate(mic_dict, solo_dict, solo_ids, df, y_axis_keys, ecoff, minor=False):
    """this can only be run on an mic_dict that has passed through mutation_mic_plot()
    due to y axis mappings.
    Could have also calculate these values by fancy indexing the dfs"""
    var_r, var_s = 0, 0
    for dict in (mic_dict, solo_dict):
        # if dict == mic_dict:

        for v in dict.values():
            for mic in v:
                if float(mic) > y_axis_keys[ecoff]:
                    var_r += 1
                else:
                    var_s += 1
        if dict == solo_dict:
            solo_r, solo_s = 0, 0
            for v in dict.values():
                for mic in v:
                    if float(mic) > y_axis_keys[ecoff]:
                        solo_r += 1
                    else:
                        solo_s += 1

    minor_r, minor_s = 0, 0
    if not minor:
        for id in solo_ids:
            if df[df.UNIQUEID == id].MUTATION.nunique() > 1:
                if float(df[df.UNIQUEID == id].MIC_FLOAT.unique().tolist()[0]) > 1.0:
                    minor_r += 1
                else:
                    minor_s += 1

    table = {
        "variants": {"R": var_r, "S": var_s},
        "solos": {"R": solo_r, "S": solo_s},
        "minor": {"R": minor_r, "S": minor_s},
    }

    return pd.DataFrame.from_dict(table).T


def plot_catalogue_counts(all, catalogue):
    sns.set_context("notebook")

    genes_S, genes_R = [], []
    for i in catalogue[catalogue.PREDICTION == "S"].index:
        gene = all[all.GENE_MUT == catalogue["GENE_MUT"][i]].GENE.tolist()[0]
        genes_S.append(gene)
    for i in catalogue[catalogue.PREDICTION == "R"].index:
        gene = all[all.GENE_MUT == catalogue["GENE_MUT"][i]].GENE.tolist()[0]
        genes_R.append(gene)
    plt.figure(figsize=(7, 5))
    df = pd.concat(
        axis=0,
        ignore_index=True,
        objs=[
            pd.DataFrame.from_dict({"Gene": genes_S, "phenotype": "S"}),
            pd.DataFrame.from_dict({"Gene": genes_R, "phenotype": "R"}),
        ],
    ).sort_values(["Gene"], ascending=True, key=lambda col: col.str.lower())

    # Count occurrences of each phenotype for each gene
    counts = df.groupby(["Gene", "phenotype"]).size().reset_index(name="counts")

    # Sort the counts dataframe
    sorted_genes = counts.sort_values(by="counts", ascending=True)["Gene"].unique()

    # Plotting
    plt.figure(figsize=(7, 5))
    ax = sns.histplot(
        data=df,
        x="Gene",
        hue="phenotype",
        multiple="dodge",
        order=sorted_genes,
        discrete=True,
    )
    plt.ylabel("Number of Catalogued Mutations")
    sns.despine()
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability if needed

    # Accessing and modifying the legend to remove the bounding box
    legend = ax.legend()
    legend.set_frame_on(False)

    plt.show()


def plot_catalogue_counts_h(all, catalogue):
    sns.set_context("notebook")

    genes_S, genes_R = [], []
    for i in catalogue[catalogue.PREDICTION == "S"].index:
        gene = all[all.GENE_MUT == catalogue["MUTATION"][i]].GENE.tolist()[0]
        genes_S.append(gene)
    for i in catalogue[catalogue.PREDICTION == "R"].index:
        gene = all[all.GENE_MUT == catalogue["MUTATION"][i]].GENE.tolist()[0]
        genes_R.append(gene)

    # Create a DataFrame with genes and phenotypes
    df = pd.concat(
        axis=0,
        ignore_index=True,
        objs=[
            pd.DataFrame.from_dict({"Gene": genes_S, "phenotype": "S"}),
            pd.DataFrame.from_dict({"Gene": genes_R, "phenotype": "R"}),
        ],
    ).sort_values(
        ["Gene", "phenotype"], ascending=[True, True], key=lambda col: col.str.lower()
    )

    df_counts = (
        df.groupby(["Gene", "phenotype"]).size().unstack(fill_value=0).reset_index()
    )

    gene_order = [
        "Rv0678",
        "pepQ",
        "mmpS5",
        "mmpL5",
        "atpE",
    ]

    sns.set_palette("muted")

    plt.figure(figsize=(16, 3))
    ax = sns.barplot(
        data=df_counts,
        x="R",
        y="Gene",
        color="#1b9e77",
        label="R",
        order=gene_order,
        edgecolor="black",
                alpha=0.7

    )
    sns.barplot(
        data=df_counts,
        x="S",
        y="Gene",
        color="#7570b3",
        ax=ax,
        order=gene_order,
        label="S",
        edgecolor="black",
        alpha=0.6

        
    )


    for p in ax.patches:
        width = p.get_width()
        plt.annotate(
            f"{width:.0f}",
            (width + 2, p.get_y() + p.get_height() / 2),
            ha="left",
            va="center",
        )

    ax.set(xlabel="Number of catalogued mutations", ylabel="Genes")
    ax.legend(title="Phenotype").set_frame_on(False)

    sns.despine()

    plt.show()



def plot_metrics(performance):
    df = pd.DataFrame(performance, index=[0])

    sns.set_theme("paper")
    sns.set_style("white")

    plt.figure(figsize=(7, 3))

    ax = sns.barplot(df)

    ax.set_ylabel("Metric Value (%)", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=10,
        )
    ax.set_ylim(0, 100)

    sns.despine()
    plt.show()


def compare_metrics(performance_comparison):
    df = (
        pd.DataFrame(performance_comparison)
        .T.reset_index()
        .melt(id_vars="index", var_name="Metric", value_name="Value")
    )
    df.rename(columns={"index": "Dataset"}, inplace=True)

    sns.set_theme(style="white")
    plt.figure(figsize=(7, 3))

    ax = sns.barplot(x="Metric", y="Value", hue="Dataset", data=df, palette=["#1b9e77", "#7570b3"])

    ax.set_ylabel("Metric Value (%)", fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            xytext=(0, 10),
            textcoords="offset points",
        )

    ax.set_ylim(0, 100)
    ax.legend(fontsize="small", title_fontsize="small", frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.show()

def compare_metrics_2charts(performance):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))  # Create subplots for two charts

    for i, frs_value in enumerate(performance.keys()):
        df = pd.DataFrame(performance[frs_value]).T.reset_index().melt(id_vars="index", var_name="Metric", value_name="Value")
        df.rename(columns={"index": "Dataset"}, inplace=True)

        ax = sns.barplot(x="Metric", y="Value", hue="Dataset", data=df, palette=["#1b9e77", "#7570b3"], ax=axes[i])

        ax.set_ylabel("Metric Value (%)", fontsize=12)
        ax.set_xlabel("Metric", fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=10)

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                xytext=(0, 10),
                textcoords="offset points",
            )

        ax.set_ylim(0, 100)
        ax.set_title(f"FRS {frs_value}", fontsize=14)  # Set title for each subplot
        ax.legend(fontsize="small", title_fontsize="small", frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.show()




def plot_metrics_std(performance, stds):
    df = pd.DataFrame(performance, index=[0])

    sns.set_theme("paper")
    sns.set_style("white")

    plt.figure(figsize=(7, 3))

    ax = sns.barplot(df)

    for k, v in stds.items():
        ax.errorbar(x=k, y=performance[k], yerr=v, color="black", zorder=10, capsize=3)

    ax.set_ylabel("Metric Value (%)", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_ylim(0, 100)

    sns.despine()
    plt.show()


def mutation_mic_plot(df, ids, gene, ecoff, fig_size):
    """Displays scatter plot of mutation vs MIC.
    Args: mic_dict --> a dictionary of mic lists for each mutation"""

    all_sha = df[df.UNIQUEID.isin(ids)]

    all_sha = all_sha[all_sha.FRS >= 0.1]

    solos, solo_ids = Helpers.extract_solos(gene, all_sha)

    mut_counts = all_sha[["GENE", "MUTATION"]].value_counts().reset_index(name="count")

    mic_dict = {}

    for i in mut_counts[(mut_counts["count"] >= 3) & (mut_counts.GENE == gene)].index:
        mic_dict[mut_counts["MUTATION"][i]] = all_sha[
            (all_sha.GENE == gene)
            & (all_sha.MUTATION == mut_counts["MUTATION"][i])
            & (~all_sha.UNIQUEID.isin(solo_ids))
        ].METHOD_MIC.tolist()

    y_axis_keys = {0.125: 0, 0.25: 1, 0.5: 2, 1.0: 3, 2.0: 4, 4.0: 5, 8.0: 6}

    values_array = np.array(
        [0.06, 0.015, 0.008, 0.03, 0.25, 0.12, 0.5, 1.0, 2.0, 8.0, 4.0, 0.125]
    )

    # Create y_axis_keys dictionary
    y_axis_keys = {
        value: index
        for index, value in enumerate(sorted(np.array(all_sha.MIC_FLOAT.unique())))
    }

    ordered_solos = {}
    for k in mic_dict.keys():
        if k in solos.keys():
            ordered_solos[k] = solos[k]
        else:
            ordered_solos[k] = []

    def prep_xy(dict):
        x_axis_keys = {}
        count = 0
        for mut in dict.keys():
            x_axis_keys[mut] = str(count)
            count += 1

        mic_dict_num = {}
        for k, v in dict.items():
            for i in range(len(v)):
                try:
                    v[i] = float(v[i])
                except ValueError:
                    try:
                        v[i] = float(v[i][1:])
                    except ValueError:
                        v[i] = float(v[i][2:])
            mic_dict_num[x_axis_keys[k]] = v

        for k, v in mic_dict_num.items():
            for i in range(len(v)):
                v[i] = y_axis_keys[v[i]]

        mic_rand_y = {}
        for k, v in mic_dict_num.items():
            mic_rand_y[k] = []
            for i in v:
                count = v.count(i)
                if count > 1:
                    rand = random.randrange(600) / 1000
                    mic_rand_y[k].append(i - 0.3 + rand)
                else:
                    mic_rand_y[k].append(i)

        x, y = [], []
        for k, v in mic_rand_y.items():
            v_rounded = [round(i, 0) for i in v]
            for i in range(len(v_rounded)):
                count = v_rounded.count(v_rounded[i])
                if count > 1:
                    rand = random.randrange(800) / 1000
                    x.append(float(k) - 0.4 + rand)
                else:
                    x.append(float(k))
                y.append(v[i])

        return x, y, x_axis_keys, y_axis_keys

    x, y, x_axis_keys, y_axis_keys = prep_xy(mic_dict)
    x_solo, y_solo, _, _ = prep_xy(solos)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.scatter(x, y, c="white", edgecolor="blue")
    ax.scatter(x_solo, y_solo, color="blue")
    ax.set_xticks([int(i) for i in x_axis_keys.values()])
    ax.set_xticklabels(x_axis_keys.keys(), rotation="vertical")
    val = -0.5
    for i in range(len(x_axis_keys)):
        ax.axvline(val, color="black", lw=0.5)
        val += 1
    ax.set_xlim(-0.5, len(x_axis_keys) - 0.5)
    ax.set_yticks([i for i in y_axis_keys.values()])
    ax.set_yticklabels(y_axis_keys.keys())
    ecoff = y_axis_keys[ecoff] + 0.5
    ax.axhline(ecoff, color="black", lw=1)

    ax.set_xlabel("Mutation")
    ax.set_ylabel("MIC (mg/L)")
    plt.show()


def FRS_vs_metric(df, cov=True):
    # Create a line plot using seaborn
    data = df
    plt.figure(figsize=(8, 4))
    sns.lineplot(x="FRS", y="Sensitivity", data=data, label="Sensitivity", color="blue")
    sns.lineplot(x="FRS", y="Specificity", data=data, label="Specificity", color="red")
    if cov:
        sns.lineplot(
            x="FRS", y="Coverage", data=data, label="Isolate Coverage", color="green"
        )

    yticks = [
        0,
        20,
        40,
        60,
        80,
        100,
    ]
    xticks = [
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    plt.yticks(yticks)
    plt.xticks(xticks)

    # Add labels and legend
    plt.xlabel("Fraction Read Support (FRS)")
    plt.ylabel("Metric (%)")
    plt.legend(loc="best", frameon=False, bbox_to_anchor=(0.85, 0.40))

    # Add final values at the end of each line
    # for line in plt.gca().lines[:-1]:
    for line in plt.gca().lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        start_value = y_data[0]
        final_value = y_data[-1]
        # Annotate the start value
        plt.annotate(f"~{start_value:.2f}", (x_data[0], start_value), textcoords="offset points", xytext=(-23, -3), ha="center")
        # Annotate the end value
        plt.annotate(f"~{final_value:.2f}", (x_data[-1], final_value), textcoords="offset points", xytext=(25, -3), ha="center")

    plt.axvline(x=0.75, color="gray", linestyle="--", label="FRS=0.75")
    plt.text(0.68, 30, "WHOv2 build threshold", color="gray", ha="left", va="top")

    plt.axvline(x=0.25, color="gray", linestyle="--", label="FRS=0.25")
    plt.text(0.23, 30, "WHOv2 evaluation threshold", color="gray", ha="left", va="top")


    sns.despine(top=True, right=True)
    plt.grid(False)

    # Show the plot
    plt.ylim(40, 100)
    plt.show()