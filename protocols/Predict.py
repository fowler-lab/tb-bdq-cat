from sklearn.metrics import confusion_matrix

from mglearn.tools import heatmap
import matplotlib.pyplot as plt


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

    """predictions_filt, labels_filt = [], []
    for i in range(len(predictions)):
        if predictions[i] != "U":
            predictions_filt.append(predictions[i])
            labels_filt.append(labels[i])"""

    FN_id = []
    for i in range(len(labels)):
        if (predictions[i] == "S") & (labels[i] == "R"):
            FN_id.append(ids[i])

    cm = confusion_matrix(labels, predictions)

    if "U" not in predictions:
        cm = cm[:2][:2]
    else:
        cm = cm[:2]
    """if "U" not in predictions:
        cm = cm[:2][:2]
        cm_hm = heatmap(
            cm,
            xlabel="predicted_label",
            ylabel="true_label",
            xticklabels=["R", "S"],
            yticklabels=["R", "S"],
            fmt="%d",
        )
        # cm_hm = plt.matshow(cm)
    else:
        cm = cm[:2]
        cm_hm = heatmap(
            cm,
            xlabel="predicted_label",
            ylabel="true_label",
            xticklabels=["R", "S", "U"],
            yticklabels=["R", "S"],
            fmt="%d",
        )
        # cm_hm = plt.matshow(cm)

    plt.gca().invert_yaxis()"""
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
