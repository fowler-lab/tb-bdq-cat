from sklearn.metrics import confusion_matrix

from mglearn.tools import heatmap
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
import numpy as np
from protocols.BuildCatalogue import BuildCatalogue
import pandas as pd
import piezo
import math


def predict(iso_df, catalogue, U_to_R=False, Print=True):
    ids = iso_df.UNIQUEID.unique().tolist()
    labels, predictions = [], []
    for i in ids:
        df = iso_df[iso_df.UNIQUEID == i]
        labels.append(df.PHENOTYPE.tolist()[0])
        mut_predictions = []
        for var in df.index:
            mut_prediction = catalogue[
                catalogue.GENE_MUT == df["GENE_MUT"][var]
            ].phenotype
            # if mutation isn't in catalogue, predict as U
            if len(mut_prediction) > 0:
                mut_predictions.append(mut_prediction.tolist()[0])
            else:
                mut_predictions.append("U")
        # if there is a single resistant mutation in that sample, predict the isolate as resistant
        if "R" in mut_predictions:
            predictions.append("R")
        else:
            # for samples with no R mutations, if there is a U, predict as U (becuase it could be R, so cant predict as S)
            if "U" in mut_predictions:
                if U_to_R:
                    predictions.append("R")
                else:
                    predictions.append("U")
            else:
                predictions.append("S")

    FN_id = []
    for i in range(len(labels)):
        if (predictions[i] == "S") & (labels[i] == "R"):
            FN_id.append(ids[i])

    cm = confusion_matrix(labels, predictions)

    if "U" not in predictions:
        cm = cm[:2][:2]
    else:
        cm = cm[:2]
    if Print:
        print(cm)

    sensitivity = cm[0][0] / (cm[0][0] + cm[0][1])
    specificity = cm[1][1] / (cm[1][1] + cm[1][0])
    isolate_cov = (len(labels) - predictions.count("U")) / len(labels)
    mut_cov = catalogue.GENE_MUT.nunique() / iso_df.GENE_MUT.nunique()

    if Print:
        print("Catalogue coverage of isolates:", isolate_cov)
        print("Catalogue coverage of mutations:", mut_cov)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

    return {
        "cm": cm,
        "isolate_cov": isolate_cov,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "FN_ids": FN_id,
    }


def piezo_predict(iso_df, catalogue_file, drug, U_to_R=False, U_to_S=False, Print=True):
    catalogue = piezo.ResistanceCatalogue(catalogue_file)

    ids = iso_df.UNIQUEID.unique().tolist()
    labels, predictions = [], []

    for i in ids:
        df = iso_df[iso_df.UNIQUEID == i]
        labels.append(df.PHENOTYPE.tolist()[0])
        mut_predictions = []
        for var in df.GENE_MUT:
            try:
                mut_predictions.append(catalogue.predict(var)[drug])
            except TypeError:
                mut_predictions.append("S")

        if "R" in mut_predictions:
            predictions.append("R")

        else:
            if "U" in mut_predictions:
                if U_to_R:
                    predictions.append("R")
                elif U_to_S:
                    predictions.append("S")
                else:
                    predictions.append("U")
            else:
                predictions.append("S")

    FN_id, FP_id = [], []
    for i in range(len(labels)):
        if (predictions[i] == "S") & (labels[i] == "R"):
            FN_id.append(ids[i])
        elif (predictions[i] == "R") & (labels[i] == "S"):
            FP_id.append(ids[i])

    cm = confusion_matrix(labels, predictions)

    if "U" not in predictions:
        cm = cm[:2][:2]
    else:
        cm = cm[:2]
    if Print:
        print(cm)

    sensitivity = cm[0][0] / (cm[0][0] + cm[0][1])
    specificity = cm[1][1] / (cm[1][1] + cm[1][0])
    isolate_cov = (len(labels) - predictions.count("U")) / len(labels)

    if Print:
        print("Catalogue coverage of isolates:", isolate_cov)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

    return [cm, isolate_cov, sensitivity, specificity, FN_id, FP_id]



def piezo_predict_cv(
    all,
    samples,
    mutations,
    FRS,
    n_splits,
    test_size,
    seed,
    genbank_ref,
    catalogue_name,
    version,
    drug,
    wildcards,
    path,
):
    X = all.UNIQUEID.unique()

    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    train_indices, test_indices = [], []
    for i, (train_index, test_index) in enumerate(rs.split(X)):
        train_indices.append(train_index)
        test_indices.append(test_index)

    isolate_cov, specificities, sensitivities, cms = [], [], [], []
    for fold in range(len(train_indices)):
        train_ids, test_ids = [], []
        for i in train_indices[fold]:
            train_ids.append(X[i])
        for j in test_indices[fold]:
            test_ids.append(X[j])

        train_samples = samples[samples.UNIQUEID.isin(train_ids)]
        train_mutations = mutations[mutations.UNIQUEID.isin(train_ids)]
        test_df = all[all.UNIQUEID.isin(test_ids)]

        BuildCatalogue(train_samples, train_mutations, FRS).build_piezo(
            genbank_ref, catalogue_name, version, drug, wildcards
        ).return_piezo().to_csv(f"{path}catalogue_FRS_{FRS}_cv.csv", index=False)

        cm, _cov, _sens, _spec, _FN_ids= piezo_predict(
            test_df, f"{path}catalogue_FRS_{FRS}_cv.csv", drug
        )
        isolate_cov.append(_cov)
        specificities.append(_spec)
        # just for the sake of plotting the bar charts - a nan would make plotting impossible
        sensitivities.append(_sens)
        cms.append(cm)

    isolate_cov = (np.mean(isolate_cov), np.std(isolate_cov))
    specificity = (np.mean(specificities), np.std(specificities))
    sensitivity = (np.mean(sensitivities), np.std(sensitivities))

    print("isolate_cov", isolate_cov)
    print("specificity", specificity)
    print("sensitivity", sensitivity)

    mean = np.mean(cms, axis=0)
    std = np.std(cms, axis=0)

    labels = [[], []]

    for i in range(len(mean)):
        for j in range(len(mean[i])):
            labels[i].append("%s" % int(mean[i][j]))

    df_cm = pd.DataFrame(mean, index=["R", "S"], columns=["R", "S", "U"])
    return df_cm, labels, sensitivity, specificity, isolate_cov


def predict_cv(
    all,
    samples,
    mutations,
    FRS,
    n_splits,
    test_size,
    seed,
):
    X = all.UNIQUEID.unique()

    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    train_indices, test_indices = [], []
    for i, (train_index, test_index) in enumerate(rs.split(X)):
        train_indices.append(train_index)
        test_indices.append(test_index)

    isolate_cov, specificities, sensitivities, cms = [], [], [], []
    for fold in range(len(train_indices)):
        train_ids, test_ids = [], []
        for i in train_indices[fold]:
            train_ids.append(X[i])
        for j in test_indices[fold]:
            test_ids.append(X[j])

        train_samples = samples[samples.UNIQUEID.isin(train_ids)]
        train_mutations = mutations[mutations.UNIQUEID.isin(train_ids)]
        test_df = all[all.UNIQUEID.isin(test_ids)]

        catalogue = BuildCatalogue(
            train_samples, train_mutations, FRS
        ).return_catalogue()
        catalogue = (
            pd.DataFrame.from_dict(catalogue, orient="index")
            .reset_index(0)
            .rename(columns={"index": "GENE_MUT", 0: "phenotype"})
        )
        catalogue.index.rename("GENE_MUT", inplace=True)

        performance = predict(test_df, catalogue, Print=False)
        isolate_cov.append(performance["isolate_cov"])
        specificities.append(performance["specificity"])
        sensitivities.append(performance["sensitivity"])
        cms.append(performance["cm"])

    isolate_cov = (np.mean(isolate_cov), np.std(isolate_cov))
    specificity = (np.mean(specificities), np.std(specificities))
    sensitivity = (np.mean(sensitivities), np.std(sensitivities))

    print("isolate_cov", isolate_cov)
    print("specificity", specificity)
    print("sensitivity", sensitivity)

    mean = np.mean(cms, axis=0)
    std = np.std(cms, axis=0)

    labels = [[], []]

    for i in range(len(mean)):
        for j in range(len(mean[i])):
            labels[i].append("%s" % int(mean[i][j]))

    df_cm = pd.DataFrame(mean, index=["R", "S"], columns=["R", "S", "U"])
    return df_cm, labels
