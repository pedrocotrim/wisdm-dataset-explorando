from dataset_contents import Person
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from statistics import mode


def voting_split(full_df: pd.DataFrame, classifier: RandomForestClassifier, class_column: str, qty: int = 5, biometrics: bool = False) -> tuple[bool | str, bool | str]:
    unique_classes = full_df[class_column].unique()
    results_pred = []
    results_true = []
    for i in unique_classes:
        sub_df: pd.DataFrame = full_df[full_df[class_column] == i]
        sub_df = sub_df.drop(columns=[class_column])
        sub_df = sub_df[:len(sub_df)-(len(sub_df) % qty)]
        for j in range(len(sub_df)//qty):
            curr_slice = sub_df[(j*qty):(j*qty)+qty]
            results = classifier.predict(curr_slice)
            most_common = mode(results)
            if biometrics:
                probs = classifier.predict_proba(curr_slice)[:, 1]
                results_pred.append(np.mean(probs))
            else:
                results_pred.append(most_common)
            results_true.append(i)
    return results_pred, results_true
