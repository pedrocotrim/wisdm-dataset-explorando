import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dataset_contents import Person, people, sensors, activities
from sklearn.metrics import roc_curve
from voting_split import voting_split
from dataclasses import dataclass


def compute_eer_changjiang(labels, prediction):
    """
    Computes the Equal Error Rate based on the ROC curve, using interpolation to
    find the point closest to the intersection between the curve and the line in which FPR==TPR.
    Credits: https://yangcha.github.io/EER-ROC/
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    fpr, tpr, thresholds = roc_curve(labels, prediction, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer


def pick_impostors(id: str, people: list[Person]) -> list[Person]:
    """
    Picks impostors for the authentication of one person, as described in the paper.
    """
    return random.choices([i for i in people if i['id'] != id], k=18)


def biometric_train_test_split(person: Person, impostors: list[Person], activity: str, sensors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Makes a custom train/test split, following the procedure described in the paper.
    """
    full_person: pd.DataFrame = None
    for sensor in sensors:
        full_person = person[sensor].query(f"`ACTIVITY`=='{activity}'").drop(columns=['class', 'ACTIVITY']).add_suffix(sensor).reset_index(drop=True) if full_person is None else full_person.merge(
            person[sensor].query(f"`ACTIVITY`=='{activity}'").drop(columns=['class', 'ACTIVITY']).add_suffix(sensor).reset_index(drop=True), left_index=True, right_index=True)
    full_person['class'] = True

    full_impostors: list[pd.DataFrame] | pd.DataFrame = []
    for impostor in impostors:
        full_impostor: pd.DataFrame = None
        for sensor in sensors:
            full_impostor = impostor[sensor].query(f"`ACTIVITY`=='{activity}'").drop(columns=['class', 'ACTIVITY']).add_suffix(sensor).reset_index(drop=True) if full_impostor is None else full_impostor.merge(
                impostor[sensor].query(f"`ACTIVITY`=='{activity}'").drop(columns=['class', 'ACTIVITY']).add_suffix(sensor).reset_index(drop=True), left_index=True, right_index=True)
        full_impostor = full_impostor.dropna()
        if len(full_impostor) >= 3:
            full_impostor = full_impostor.sample(n=3)
        full_impostor['class'] = False
        full_impostors.append(full_impostor)
    full_person = full_person.sample(frac=1)
    full_impostors = pd.concat(full_impostors).sample(frac=1)
    return pd.concat([full_person[:len(full_person)//2], full_impostors[:len(full_impostors)//2]]).sample(frac=1.0), pd.concat([full_person[len(full_person)//2:], full_impostors[len(full_impostors)//2:]]).sample(frac=1.0)


def main() -> None:
    biometry_models_single = {activity: [] for activity in activities}
    biometry_models_vote = {activity: [] for activity in activities}
    biometry_models_test = {activity: [] for activity in activities}
    for person in people:
        impostors = pick_impostors(person['id'], people)
        for activity in activities:
            train, test = biometric_train_test_split(
                person, impostors, activity, sensors)
            if len(train) == 0 or len(test) == 0:
                continue
            train_target = train['class']
            # print(train_target.unique())
            train = train.drop(columns=['class'])
            test_target = test['class']
            test_full = test.copy()
            test = test.drop(columns=['class'])
            biometry_classifier = RandomForestClassifier(
                10, max_features='sqrt')
            biometry_classifier.fit(train, train_target)
            test_classifier = RandomForestClassifier(
                50, max_features='log2')
            biometry_classifier.fit(train, train_target)
            test_classifier.fit(train, train_target)
            y_true = test_target
            if len(y_true.unique()) != 2:
                # print(person['guid'],combination,activity)
                continue
            prediction = biometry_classifier.predict_proba(test)[:, 1]
            test_prediction = test_classifier.predict_proba(test)[:, 1]
            eer = compute_eer_changjiang(test_target, prediction)
            test_eer = compute_eer_changjiang(test_target, test_prediction)
            vote_pred, vote_true = voting_split(
                test_full, biometry_classifier, 'class', biometrics=True)
            eer_vote = compute_eer_changjiang(vote_true, vote_pred)
            biometry_models_vote[activity].append(eer_vote)
            biometry_models_single[activity].append(eer)
            biometry_models_test[activity].append(test_eer)

    print("unfiltered")
    for activity in biometry_models_single:
        print(activities[activity], np.mean(
            biometry_models_single[activity])*100,
            np.mean(
            biometry_models_vote[activity])*100,
            np.mean(
            biometry_models_test[activity])*100)
    print("filtered")
    for activity in biometry_models_single:
        biometry_models_single[activity] = list(filter(lambda x: x!=0,biometry_models_single[activity]))
        biometry_models_test[activity] = list(filter(lambda x: x!=0,biometry_models_test[activity]))
        biometry_models_vote[activity] = list(filter(lambda x: x!=0,biometry_models_vote[activity]))
        print(activities[activity], np.mean(
            biometry_models_single[activity])*100,
            np.mean(
            biometry_models_vote[activity])*100,
            np.mean(
            biometry_models_test[activity])*100)

if __name__ == "__main__":
    main()
