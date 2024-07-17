import json
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


PHENOTYPE = "PHENOTYPE"
UNIQUEID = "UNIQUEID"
GENE = "GENE"
METHOD_MIC = "METHOD_MIC"
FRS = "FRS"


def filter_multiple_phenos(group):
    """
    If a sample contains more than one phenotype,
    keep the resistant phenotype (preferably with MIC) if there is one.

    Parameters:
    group (pd.DataFrame): A dataframe containing sample data with phenotypes.

    Returns:
    pd.DataFrame: A filtered dataframe prioritizing resistant phenotypes.
    """
    if len(group) == 1:
        return group

    # Prioritize rows with 'R' phenotype
    prioritized_group = (
        group[group[PHENOTYPE] == "R"] if "R" in group[PHENOTYPE].values else group
    )

    # Check for rows with METHOD_MIC values
    with_mic = prioritized_group.dropna(subset=[METHOD_MIC])
    return with_mic.iloc[0:1] if not with_mic.empty else prioritized_group.iloc[0:1]


def combined_data_table(all_data):
    """
    Combines multiple data tables into a single dataframe with multi-level columns.

    Parameters:
    all_data (pd.DataFrame): The input dataframe containing all sample data.

    Returns:
    pd.DataFrame: A combined dataframe with multi-level columns.
    """
    df_all = generate_isolate_or_variant_table(
        all_data, all_data[GENE].unique(), unique=True
    )
    df_minor_alleles = generate_isolate_or_variant_table(
        all_data[all_data[FRS] < 0.9], all_data[GENE].unique(), unique=True
    )
    df_variants = generate_isolate_or_variant_table(
        all_data, all_data[GENE].unique(), unique=False
    )
    df_variants_minor_alleles = generate_isolate_or_variant_table(
        all_data[all_data[FRS] < 0.9], all_data[GENE].unique(), unique=False
    )

    combined_df = pd.concat(
        [df_all, df_minor_alleles, df_variants, df_variants_minor_alleles], axis=1
    )

    combined_df.columns = pd.MultiIndex.from_tuples(
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
            combined_df.columns,
        )
    )
    return combined_df


def data_table(all_data):
    """
    Creates a single dataframe with multi-level columns for sample and mutation counts.

    Parameters:
    all_data (pd.DataFrame): The input dataframe containing all sample data.

    Returns:
    pd.DataFrame: A combined dataframe with multi-level columns.
    """
    df_all = generate_isolate_or_variant_table(
        all_data, all_data[GENE].unique(), unique=True
    )

    df_variants = generate_isolate_or_variant_table(
        all_data, all_data[GENE].unique(), unique=False
    )

    combined_df = pd.concat([df_all, df_variants], axis=1)

    combined_df.columns = pd.MultiIndex.from_tuples(
        zip(
            [
                "Samples",
                "",
                "",
                "Mutations",
                "",
                "",
            ],
            combined_df.columns,
        )
    )
    return combined_df


def generate_isolate_or_variant_table(df, genes, unique):
    """
    Generates a table of counts for isolates or variants based on phenotypes for each gene.

    Parameters:
    df (pd.DataFrame): The input dataframe containing sample data.
    genes (list): A list of genes to include in the table.
    unique (bool): If True, count unique isolates; otherwise, count total variants.

    Returns:
    pd.DataFrame: A dataframe with counts for each gene and phenotype.
    """
    table = {
        "Total": {
            "R": (
                df[df[PHENOTYPE] == "R"][UNIQUEID].nunique()
                if unique
                else df[df[PHENOTYPE] == "R"][UNIQUEID].count()
            ),
            "S": (
                df[df[PHENOTYPE] == "S"][UNIQUEID].nunique()
                if unique
                else df[df[PHENOTYPE] == "S"][UNIQUEID].count()
            ),
            "Total": df[UNIQUEID].nunique() if unique else df[UNIQUEID].count(),
        }
    }

    for gene in genes:
        gene_df = df[df[GENE] == gene]
        table[gene] = {
            "R": (
                gene_df[gene_df[PHENOTYPE] == "R"][UNIQUEID].nunique()
                if unique
                else gene_df[gene_df[PHENOTYPE] == "R"][UNIQUEID].count()
            ),
            "S": (
                gene_df[gene_df[PHENOTYPE] == "S"][UNIQUEID].nunique()
                if unique
                else gene_df[gene_df[PHENOTYPE] == "S"][UNIQUEID].count()
            ),
            "Total": (
                gene_df[gene_df[PHENOTYPE] == "R"][UNIQUEID].nunique()
                + gene_df[gene_df[PHENOTYPE] == "S"][UNIQUEID].nunique()
                if unique
                else gene_df[gene_df[PHENOTYPE] == "R"][UNIQUEID].count()
                + gene_df[gene_df[PHENOTYPE] == "S"][UNIQUEID].count()
            ),
        }

    return pd.DataFrame.from_dict(table).T


def confusion_matrix(labels, predictions, classes):
    """
    Creates a confusion matrix for given labels and predictions with specified classes.

    Parameters:
    labels (list): Actual labels.
    predictions (list): Predicted labels.
    classes (list): List of all classes.

    Returns:
    np.ndarray: Confusion matrix.
    """
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    for label, prediction in zip(labels, predictions):
        if label in class_to_index and prediction in class_to_index:
            cm[class_to_index[label], class_to_index[prediction]] += 1

    return cm


def plot_truthtables(truth_table, U_to_S=False, figsize=(2.5, 1.5), fontsize=12):
    """
    Plots a truth table as a confusion matrix to denote each cell.

    Parameters:
    truth_table (pd.DataFrame): DataFrame containing the truth table values.
                                The DataFrame should have the following structure:
                                - Rows: True labels ("R" and "S")
                                - Columns: Predicted labels ("R", "S", and optionally "U")
    U_to_S (bool): Whether to separate the "U" values from the "S" column. If True,
                   an additional column for "U" values will be used.
    figsize (tuple): Figure size for the plot.
    fontsize (int): Font size for the text in the plot.

    Returns:
    None
    """
    fig = plt.figure(figsize=figsize)
    axes = plt.gca()

    if not U_to_S:
        axes.add_patch(Rectangle((2, 0), 1, 1, fc="#377eb8", alpha=0.5))
        axes.add_patch(Rectangle((2, 1), 1, 1, fc="#377eb8", alpha=0.5))

        axes.set_xlim([0, 3])
        axes.set_xticks([0.5, 1.5, 2.5])
        axes.set_xticklabels(["S", "R", "U"], fontsize=fontsize)
    else:
        axes.set_xlim([0, 2])
        axes.set_xticks([0.5, 1.5])
        axes.set_xticklabels(["S+U", "R"], fontsize=fontsize)

    axes.add_patch(Rectangle((0, 0), 1, 1, fc="#e41a1c", alpha=0.7))
    axes.add_patch(Rectangle((1, 0), 1, 1, fc="#4daf4a", alpha=0.7))
    axes.add_patch(Rectangle((1, 1), 1, 1, fc="#fc9272", alpha=0.7))
    axes.add_patch(Rectangle((0, 1), 1, 1, fc="#4daf4a", alpha=0.7))

    axes.set_ylim([0, 2])
    axes.set_yticks([0.5, 1.5])
    axes.set_yticklabels(["R", "S"], fontsize=fontsize)

    axes.text(
        1.5,
        0.5,
        int(truth_table["R"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        1.5,
        1.5,
        int(truth_table["R"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        1.5,
        int(truth_table["S"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        0.5,
        int(truth_table["S"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )

    if not U_to_S:
        axes.text(
            2.5,
            0.5,
            int(truth_table["U"]["R"]),
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        axes.text(
            2.5,
            1.5,
            int(truth_table["U"]["S"]),
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    plt.show()


def compare_metrics(performance_comparison, figsize=(8, 6)):
    """
    Plots a comparison of performance metrics for different datasets.

    Parameters:
    performance_comparison (dict): A dictionary where keys are dataset names and values are dictionaries
                                   of performance metrics. The inner dictionaries should have metric names
                                   as keys and their corresponding values as values.
                                   Example:
                                   {
                                       "Dataset1": {"Metric1": value1, "Metric2": value2},
                                       "Dataset2": {"Metric1": value3, "Metric2": value4},
                                   }
    figsize (tuple): Figure size in inches, default is (8, 6).

    Returns:
    None
    """
    # Refactor dict into a df for seaborn
    df = (
        pd.DataFrame(performance_comparison)
        .T.reset_index()
        .melt(id_vars="index", var_name="Metric", value_name="Value")
    )
    df.rename(columns={"index": "Dataset"}, inplace=True)

    sns.set_theme(style="white")
    plt.figure(figsize=figsize)
    # Plot metrics as bars
    ax = sns.barplot(
        x="Metric", y="Value", hue="Dataset", data=df, palette=["#1b9e77", "#7570b3"]
    )

    ax.set_ylabel("Value (%)")
    ax.set_xlabel("Metric")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            xytext=(0, 10),
            textcoords="offset points",
        )
    # Clean up plot
    ax.set_ylim(0, 100)
    ax.legend(fontsize="small", title_fontsize="small", frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.show()


def compare_metrics_groups(performance, figsize=(8, 6)):
    """
    Plots a comparison of performance metrics for different datasets, across different experiments.

    Parameters:
    performance (dict): A dictionary where keys are subset names (e.g., different experiments) and values are
                        dictionaries containing dataset names as keys and another dictionary as values.
                        This innermost dictionary should have metric names as keys and their corresponding values as values.
                        Example:
                        {
                            "Experiment1": {
                                "Dataset1": {"Metric1": value1, "Metric2": value2},
                                "Dataset2": {"Metric1": value3, "Metric2": value4}
                            },
                            "Experiment2": {
                                "Dataset1": {"Metric1": value5, "Metric2": value6},
                                "Dataset2": {"Metric1": value7, "Metric2": value8}
                            }
                        }
    figsize (tuple): Figure size in inches, default is (8, 6).

    Returns:
    None
    """
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=figsize)  # Create subplots for two charts

    # for each experiment:
    for i, experiment in enumerate(performance.keys()):
        # refactor dict into dataframe for seaborn

        df = (
            pd.DataFrame(performance[experiment])
            .T.reset_index()
            .melt(id_vars="index", var_name="Metric", value_name="Value")
        )
        df.rename(columns={"index": "Dataset"}, inplace=True)
        # plot metrics as bars
        ax = sns.barplot(
            x="Metric",
            y="Value",
            hue="Dataset",
            data=df,
            palette=["#1b9e77", "#7570b3", "#fb8072"],
            ax=axes[i],
        )

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
        ax.set_title(f"{experiment}", fontsize=14)  # Set title for each subplot
        ax.legend(fontsize="small", title_fontsize="small", frameon=False)
    # clean up plot
    sns.despine()
    plt.tight_layout()
    plt.show()


def str_to_dict(s):
    """Convert strings to dictionary - helpful for evidence column"""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def plot_tricolour_venn(
    subsets, labels, figsize=(8, 6), x_offsets=(0.03, 0.03), y_offsets=(-0.045, -0.01)
):
    """
    Generates a tricolour Venn diagram for variants present in two different groups.

    Parameters:
    subsets (tuple): A tuple of three values representing the sizes of the subsets for the Venn diagram.
                     Example: (size_in_first_set_only, size_in_second_set_only, size_in_both_sets)
    labels (tuple): A tuple of two strings representing the labels for the two sets.
                    Example: ("WHO Catalogue", "MMM Catalogue")
    figsize (tuple): A tuple specifying the size of the figure.
                     Example: (8, 6)
    x_offsets (tuple): A tuple specifying the x position offsets for the labels.
                       Example: (0.03, 0.03)
    y_offsets (tuple): A tuple specifying the y position offsets for the labels.
                       Example: (-0.045, -0.01)

    Returns:
    None
    """
    plt.figure(figsize=figsize)
    v = venn2(subsets=subsets, set_labels=labels)
    v.get_patch_by_id("10").set_color("#1b9e77")  # Color for WHO only
    v.get_patch_by_id("01").set_color("#7570b3")  # Color for MMM only
    v.get_patch_by_id("11").set_color("#d95f02")  # Color for intersection

    # Adjust position and font size for the first label
    label = v.set_labels[0]
    x, y = label.get_position()
    label.set_position((x + x_offsets[0], y + y_offsets[0]))
    label.set_fontsize(12)

    # Adjust position and font size for the second label
    label = v.set_labels[1]
    x, y = label.get_position()
    label.set_position((x + x_offsets[1], y + y_offsets[1]))
    label.set_fontsize(12)

    # Adjust font size for subset labels
    for text in v.subset_labels:
        text.set_fontsize(22)

    plt.show()


def FRS_vs_metric(df, cov=True):
    """
    Plots a comparison of performance metrics (Sensitivity, Specificity, and optionally Coverage)
    against Fraction Read Support (FRS).

    Parameters:
    df (pandas.DataFrame): DataFrame containing the performance metrics with columns "FRS",
                           "Sensitivity", "Specificity", and optionally "Coverage".
    cov (bool): If True, includes Coverage in the plot. Defaults to True.

    Returns:
    None

    """
    plt.figure(figsize=(8, 4))

    # Plot Sensitivity and Specificity
    sns.lineplot(x="FRS", y="Sensitivity", data=df, label="Sensitivity", color="blue")
    sns.lineplot(x="FRS", y="Specificity", data=df, label="Specificity", color="red")

    # Plot Coverage if specified
    if cov:
        sns.lineplot(
            x="FRS", y="Coverage", data=df, label="Isolate Coverage", color="green"
        )

    # Set x and y ticks
    yticks = [0, 20, 40, 60, 80, 100]
    xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.yticks(yticks)
    plt.xticks(xticks)

    # Add labels and legend
    plt.xlabel("Fraction Read Support (FRS)")
    plt.ylabel("Metric (%)")
    plt.legend(loc="best", frameon=False, bbox_to_anchor=(0.85, 0.40))

    # Annotate the start and end values
    for line in plt.gca().lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        start_value = y_data[0]
        final_value = y_data[-1]
        plt.annotate(
            f"~{start_value:.2f}",
            (x_data[0], start_value),
            textcoords="offset points",
            xytext=(-23, -3),
            ha="center",
        )
        plt.annotate(
            f"~{final_value:.2f}",
            (x_data[-1], final_value),
            textcoords="offset points",
            xytext=(25, -3),
            ha="center",
        )

    # Add vertical lines and text annotations
    plt.axvline(x=0.75, color="gray", linestyle="--", label="FRS=0.75")
    plt.text(0.68, 30, "WHOv2 build threshold", color="gray", ha="left", va="top")

    plt.axvline(x=0.25, color="gray", linestyle="--", label="FRS=0.25")
    plt.text(0.15, 30, "WHOv2 evaluation threshold", color="gray", ha="left", va="top")

    # Despine and grid settings
    sns.despine(top=True, right=True)
    plt.grid(False)

    # Show plot
    plt.ylim(40, 100)
    plt.show()


def plot_catalogue_counts(df, figsize=(6, 2.5)):
    """
    Plots a horizontal bar chart showing the counts of R, S, and U predictions per gene.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'MUTATION' and 'PREDICTION'.
    figsize (tuple): Figure size in the format (width, height).

    """
    # Extract the gene names
    df["GENE"] = df["MUTATION"].apply(lambda x: x.split("@")[0])

    # Count the occurrences of each prediction type per gene
    count_data = df.groupby(["GENE", "PREDICTION"]).size().unstack(fill_value=0)

    colors = {"S": "#e41a1c", "R": "#4daf4a", "U": "#377eb8"}

    # Plot the chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = count_data.plot(
        kind="barh",
        stacked=True,
        color=[colors["R"], colors["S"], colors["U"]],
        edgecolor="none",
        width=0.8,
        ax=ax,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(frameon=False, fontsize="small")
    ax.set_xlabel("Count")
    ax.set_ylabel("Gene")

    # Add data labels for R counts
    for i, (gene, row) in enumerate(count_data.iterrows()):
        if "R" in row:
            ax.text(
                row["R"] + 2, i, f'{row["R"]}', va="center", ha="left", color="black"
            )

    plt.show()


def plot_catalogue_proportions(
    catalogue, background=None, figsize=None, order=True, title=None
):
    """
    Plots the proportions with confidence intervals for mutations in the given catalogue.

    Parameters:
    catalogue (dict): A dictionary where keys are mutation identifiers and values are dictionaries
                      containing mutation details, including 'proportion' and 'confidence' intervals.
    background (float): A value on which to draw the vertical background line. Defaults to None.
    figsize (tuple): A tuple representing the figure size. Default is (10, 20).
    order (bool): Whether to order by proportion. Default: True
    title (str): Title of the plot. Defaults to None.
    """
    rows = []
    for mutation, details in catalogue.items():
        evid = details["evid"][0]
        rows.append(
            {
                "Mutation": mutation,
                "Proportion": evid["proportion"],
                "Lower_bound": evid["confidence"][0],
                "Upper_bound": evid["confidence"][1],
                "Interval": evid["confidence"][1] - evid["confidence"][0],
                "Background": background,
            }
        )
    df = pd.DataFrame(rows)

    # Sort DataFrame by Proportion
    if order:
        df = df.sort_values(by=["Proportion", "Interval"], ascending=[False, False])

    dataframes = []
    length = 106
    start = 0
    end = length
    while True:
        if len(df) > end:
            dataframes.append(df[start:end])
            start += length
            end += length
        else:
            dataframes.append(df[start:])
            break

    figures = []
    axes = []

    for df2 in dataframes:
        # Plotting
        if figsize is None:
            height = len(df2) / 9.85 + 0.9
            fig, ax = plt.subplots(figsize=(4, height))
        else:
            fig, ax = plt.subplots(figsize=figsize)
        xerr = [
            abs(df2["Proportion"] - df2["Lower_bound"]),
            abs(df2["Upper_bound"] - df2["Proportion"]),
        ]

        for i in range(len(df2)):
            ax.plot(
                [df2["Lower_bound"].iloc[i], df2["Upper_bound"].iloc[i]],
                [i, i],
                color="#377eb8",
                lw=1,
            )
            ax.plot(
                df2["Proportion"].iloc[i], i, marker="|", color="#377eb8", markersize=10
            )
            if background is not None:
                ax.axvline(
                    x=df2["Background"].iloc[i], color="red", linestyle="--", lw=1
                )

        ax.set_yticks(np.arange(len(df2)))
        ax.set_yticklabels([i if len(i) < 20 else i[:20] for i in df2["Mutation"]])
        ax.set_title(title)
        for item in ax.get_yticklabels():
            if figsize is None:
                item.set_fontsize(7)
            else:
                item.set_fontsize(9)

        for item in [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels():
            item.set_fontsize(9)

        plt.xlabel("proportion resistant")
        # plt.ylabel("mutation")
        plt.tight_layout()
        plt.xlim(-0.05, 1.05)
        ax.set_ylim(-0.5, len(df2) - 0.5)  # Adjust y-axis limits to fit the data
        sns.despine()
        figures.append(fig)
        axes.append(ax)

    return figures, axes


def plot_catalogue_counts_from_df(
    df, background=None, figsize=(10, 20), order=True, title=None
):
    """
    Plots the proportions with confidence intervals for mutations in the given dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'MUTATION', 'PREDICTION', and 'EVIDENCE'.
    background (float): A value on which to draw the vertical background line. Defaults to None.
    figsize (tuple): A tuple representing the figure size. Default is (10, 20).
    order (bool): Whether to order by proportion. Default: True
    title (str): Title of the plot. Defaults to None.
    """
    rows = []
    for index, row in df.iterrows():
        R, S = row["R"], row["S"]
        wilson_intervals = wilson(R, S)
        rows.append(
            {
                "Mutation": row["MUTATION"],
                "Proportion": wilson_intervals["proportion"],
                "Lower_bound": wilson_intervals["lower_bound"],
                "Upper_bound": wilson_intervals["upper_bound"],
                "Interval": wilson_intervals["upper_bound"]
                - wilson_intervals["lower_bound"],
                "Background": background,
            }
        )
    plot_df = pd.DataFrame(rows)

    # Sort DataFrame by Proportion
    if order:
        plot_df = plot_df.sort_values(
            by=["Proportion", "Interval"], ascending=[False, False]
        )

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    xerr = [
        abs(plot_df["Proportion"] - plot_df["Lower_bound"]),
        abs(plot_df["Upper_bound"] - plot_df["Proportion"]),
    ]

    for i in range(len(plot_df)):
        ax.plot(
            [plot_df["Lower_bound"].iloc[i], plot_df["Upper_bound"].iloc[i]],
            [i, i],
            color="darkBlue",
            lw=2,
        )
        ax.plot(
            plot_df["Proportion"].iloc[i], i, marker="s", color="darkBlue", markersize=5
        )
        if background is not None:
            ax.axvline(
                x=plot_df["Background"].iloc[i], color="red", linestyle="--", lw=1
            )

    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels([i if len(i) < 20 else i[:20] for i in plot_df["Mutation"]])
    ax.set_title(title)

    plt.xlabel("Proportions with Wilson CIs")
    plt.ylabel("Mutation")
    plt.tight_layout()
    plt.xlim(0, 1.05)
    ax.set_ylim(-0.5, len(plot_df) - 0.5)  # Adjust y-axis limits to fit the data
    sns.despine()

    plt.show()


def mic_to_float(arr):
    """
    Converts an array of MIC (Minimum Inhibitory Concentration) values to floats.

    Parameters:
    arr (list of str): List of MIC values as strings, which may include symbols or text.

    Returns:
    list of float: List of MIC values converted to floats.
    """
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


def background_vs_metric(df, cov=True):
    """
    Plots a comparison of performance metrics (Sensitivity, Specificity, and optionally Coverage)
    against a defined background rage.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the performance metrics with columns "Background",
                           "Sensitivity", "Specificity", and optionally "Coverage".
    cov (bool): If True, includes Coverage in the plot. Defaults to True.

    Returns:
    None

    """
    plt.figure(figsize=(8, 4))

    # Plot Sensitivity and Specificity
    sns.lineplot(
        x="Background", y="Sensitivity", data=df, label="Sensitivity", color="blue"
    )
    sns.lineplot(
        x="Background", y="Specificity", data=df, label="Specificity", color="red"
    )

    # Plot Coverage if specified
    if cov:
        sns.lineplot(
            x="Background",
            y="Coverage",
            data=df,
            label="Isolate Coverage",
            color="green",
        )

    # Set x and y ticks
    yticks = [0, 20, 40, 60, 80, 100]
    xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.yticks(yticks)
    plt.xticks(xticks)

    # Add labels and legend
    plt.xlabel("Background")
    plt.ylabel("Metric (%)")
    plt.legend(loc="best", frameon=False, bbox_to_anchor=(0.85, 0.40))

    # Annotate the start and end values
    for line in plt.gca().lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        start_value = y_data[0]
        final_value = y_data[-1]
        plt.annotate(
            f"~{start_value:.2f}",
            (x_data[0], start_value),
            textcoords="offset points",
            xytext=(-23, -3),
            ha="center",
        )
        plt.annotate(
            f"~{final_value:.2f}",
            (x_data[-1], final_value),
            textcoords="offset points",
            xytext=(25, -3),
            ha="center",
        )

    # Despine and grid settings
    sns.despine(top=True, right=True)
    plt.grid(False)

    # Show plot
    plt.ylim(0, 110)
    plt.show()


def plot_stacked_positions(
    grouped_counts,
    all_grouped_positions,
    colors,
    high_count_threshold=90,
    figsize=(20, 8),
    bar_width=1.5,
    line_counts=None,
    line_color="black",
    line_label="Threshold Line",
):
    """
    Plots stacked bars based on the provided counts and positions with the specified styling.

    Parameters:
    grouped_counts (dict): Dictionary with legend keys and their corresponding count values (as pandas Series).
    all_grouped_positions (iterable): Iterable of all grouped codon positions.
    colors (list): List of colors for the bars, must be the same length as grouped_counts.
    high_count_threshold (int): Threshold to create an inset for bars with counts above this value.
    figsize (tuple): Figure size in the format (width, height).
    bar_width (float): Width of the bars.
    line_counts (dict): Dictionary with positions as keys and counts as values to draw a line within each bar at the specified count.
    line_color (str): Color of the lines.
    line_label (str): Label for the line in the legend.
    """

    if len(colors) != len(grouped_counts):
        raise ValueError("The length of colors must match the length of grouped_counts")

    # Create a DataFrame from the grouped counts
    df = pd.DataFrame(grouped_counts).reindex(all_grouped_positions, fill_value=0)

    # Plot the data
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind="bar", stacked=True, color=colors, width=bar_width, ax=ax)

    ax.set_ylabel("Number of isolates", fontsize=15)
    ax.set_xlabel("Codon position in Rv0678", fontsize=15)

    handles, labels = ax.get_legend_handles_labels()

    if line_counts:
        custom_line = plt.Line2D([0], [0], color=line_color, linestyle="-", linewidth=2)
        handles.append(custom_line)
        labels.append(line_label)

    ax.legend(handles=handles, labels=labels, loc="upper left", frameon=False)

    # Increase the number of x-tick labels
    ax.set_xticks(range(0, len(all_grouped_positions), 5))
    # ax.set_xticklabels(all_grouped_positions)
    ax.xaxis.set_tick_params(rotation=0)

    # Set the x-axis limits to remove the gaps
    ax.set_xlim([0.5, len(all_grouped_positions) - 0.5])

    # Highlight the really long bars
    mutation_counts_grouped = df.sum(axis=1)
    high_counts = mutation_counts_grouped[
        mutation_counts_grouped > high_count_threshold
    ]
    ax.set_ylim(0, high_count_threshold)

    # Draw lines within the bars at specified counts
    if line_counts:
        for position, count in line_counts.items():
            if position in all_grouped_positions:
                pos_index = all_grouped_positions.index(position)
                ax.plot(
                    [pos_index - bar_width / 2, pos_index + bar_width / 2],
                    [count, count],
                    color=line_color,
                    linestyle="-",
                    linewidth=2,
                )

    # Create an inset plot
    ax_inset = inset_axes(
        ax,
        width="7%",
        height="58%",
        loc="upper right",
    )

    df_high_counts = df.loc[high_counts.index]
    df_high_counts.plot(
        kind="bar",
        stacked=True,
        color=colors,
        width=bar_width,
        ax=ax_inset,
        legend=False,
    )

    ax_inset.set_xlabel("Codon position", fontsize=10)
    ax_inset.set_ylabel("Number of isolates", fontsize=10)
    ax_inset.set_xticks(range(len(high_counts.index)))
    ax_inset.set_xticklabels(high_counts.index, rotation=0)

    # Draw lines within the bars at specified counts for the inset plot
    if line_counts:
        for position, count in line_counts.items():
            if position in high_counts.index:
                pos_index = high_counts.index.tolist().index(position)
                ax_inset.plot(
                    [pos_index - bar_width / 2, pos_index + bar_width / 2],
                    [count, count],
                    color=line_color,
                    linestyle="-",
                    linewidth=2,
                )

    # Remove top and right spines from both plots
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        ax_inset.spines[spine].set_visible(False)
    return (fig, ax)


def wilson(R, S):
    """Calculates wilson confidence intervals for supplied counts"""
    successes, n = R, R + S
    proportion = successes / n
    z = norm.ppf(1 - 0.05 / 2)
    denom = 1 + (z**2 / n)
    centre_adjusted_prob = proportion + (z**2 / (2 * n))
    adjusted_sd = z * np.sqrt((proportion * (1 - proportion) / n) + (z**2 / (4 * n**2)))

    lower = (centre_adjusted_prob - adjusted_sd) / denom
    upper = (centre_adjusted_prob + adjusted_sd) / denom

    return pd.Series(
        [proportion, lower, upper], index=["proportion", "lower_bound", "upper_bound"]
    )
