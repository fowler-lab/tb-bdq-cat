import piezo
import pandas as pd
from protocols import utils
from sklearn.model_selection import ShuffleSplit


PHENOTYPE = "PHENOTYPE"
UNIQUEID = "UNIQUEID"
GENE = "GENE"
METHOD_MIC = "METHOD_MIC"
FRS = "FRS"
MUTATION = "MUTATION"


def piezo_predict(iso_df, catalogue_file, drug, U_to_R=False, U_to_S=False, Print=True):
    """
    Predicts drug resistance based on genetic mutations using a resistance catalogue.

    Parameters:
    iso_df (pd.DataFrame): DataFrame containing isolate data with UNIQUEID, PHENOTYPE, and GENE_MUT columns.
    catalogue_file (str): Path to the resistance catalogue file.
    drug (str): The drug for which resistance predictions are to be made.
    U_to_R (bool, optional): If True, treat 'U' predictions as 'R'. Defaults to False.
    U_to_S (bool, optional): If True, treat 'U' predictions as 'S'. Defaults to False.
    Print (bool, optional): If True, prints the confusion matrix, coverage, sensitivity, and specificity. Defaults to True.

    Returns:
    list: Confusion matrix, isolate coverage, sensitivity, specificity, and false negative IDs.
    """
    # Load and parse the catalogue with piezo
    catalogue = piezo.ResistanceCatalogue(catalogue_file)

    # Ensure the UNIQUEID and PHENOTYPE columns are used correctly
    ids = iso_df['UNIQUEID'].unique().tolist()
    labels = iso_df.groupby('UNIQUEID')['PHENOTYPE'].first().reindex(ids).tolist()
    predictions = []

    for id_ in ids:
        # For each sample
        df = iso_df[iso_df['UNIQUEID'] == id_]
        # Predict phenotypes for each mutation via lookup
        mut_predictions = []
        for var in df['MUTATION']:
            if pd.isna(var):
                predict = 'S'
            else:
                predict = catalogue.predict(var)
            if isinstance(predict, dict):
                mut_predictions.append(predict[drug])
            else:
                mut_predictions.append(predict)

        # Make sample-level prediction from mutation-level predictions. R > U > S
        if "R" in mut_predictions:
            predictions.append("R")
        elif "U" in mut_predictions:
            if U_to_R:
                predictions.append("R")
            elif U_to_S:
                predictions.append("S")
            else:
                predictions.append("U")
        else:
            predictions.append("S")

    # Log false negative samples
    FN_id = [
        id_
        for id_, label, pred in zip(ids, labels, predictions)
        if pred == "S" and label == "R"
    ]

    # Generate confusion matrix for performance analysis
    cm = utils.confusion_matrix(labels, predictions, classes=["R", "S", "U"])

    if "U" not in predictions:
        cm = cm[:2, :2]
    else:
        cm = cm[:2, :]

    if Print:
        print(cm)
    
    # Calculate performance metrics
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    isolate_cov = (len(labels) - predictions.count("U")) / len(labels)

    if Print:
        print("Catalogue coverage of isolates:", isolate_cov)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

    return [cm, isolate_cov, sensitivity, specificity, FN_id]
