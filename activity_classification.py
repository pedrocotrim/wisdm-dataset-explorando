from typing import Optional
import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from dataset_contents import people, sensors, activities, Person
from voting_split import voting_split
from dataclasses import dataclass


def activity_classification_df_builder(people: list[Person], sensors: list[str], activities: dict[str,str]) -> pd.DataFrame:
    '''Merges all sensors into one dataframe and sets up the correct activity class'''
    people_df: list[pd.DataFrame] = []
    for person in people:
        for activity in activities:
            curr_df = None
            for sensor in sensors:
                curr_df = person[sensor].query(f"`ACTIVITY`=='{activity}'").drop(columns=['class', 'ACTIVITY']).add_suffix(sensor).reset_index(drop=True) if curr_df is None else curr_df.merge(
                    person[sensor].query(f"`ACTIVITY`=='{activity}'").drop(columns=['class', 'ACTIVITY']).add_suffix(sensor).reset_index(drop=True), left_index=True, right_index=True)
            curr_df['ACTIVITY'] = activity
            people_df.append(curr_df)
    return pd.concat(people_df).dropna()


"""
Section on parameter optimization
======================================================================================================================================
classifier = RandomForestClassifier()
optimize_rf_params = {'criterion':['gini','entropy','log_loss'], 'max_features':['sqrt','log2',None], 'n_estimators':(1,50)}
rkf = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
random_search = RandomizedSearchCV(estimator=classifier,param_distributions=optimize_rf_params,n_jobs=-1,scoring='accuracy',cv=rkf)
random_search.fit(activities_df,activities_target)
best_model = random_search.best_estimator_
print(random_search.best_score_)
print(best_model.get_params())
======================================================================================================================================
"""
@dataclass
class Results:
    confusion_matrix: np.ndarray
    metrics: list[list[float]]
    activities: dict[str,str]
    metrics_df: pd.DataFrame


def main(voting: Optional[bool] = False) -> Results:
    kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    activities_df = activity_classification_df_builder(
        people, sensors, activities)
    activities_target = activities_df['ACTIVITY']
    activities_df = activities_df.drop(columns=['ACTIVITY'])
    activity_rfc = RandomForestClassifier(50, max_features='log2')
    if voting == False:
        score = cross_validate(activity_rfc, activities_df,
                            activities_target, cv=kf, scoring='accuracy')
        mean_accuracy = np.average(score['test_score'])
        print(f"KFold Cross validation accuracy: {mean_accuracy}")

        activity_predict = cross_val_predict(
            activity_rfc, activities_df, activities_target, cv=kf)
        activity_cm = confusion_matrix(
            activities_target, activity_predict, labels=list(activities))

        activity_metrics = []
        for activity in activities:
            _, recall, _, _ = precision_recall_fscore_support(np.array(
                activities_target) == activity, np.array(activity_predict) == activity)
            activity_metrics.append([activities[activity], recall[0], recall[1]])
        metrics_df = pd.DataFrame(activity_metrics, columns=[
                                "activity", 'specificity', 'sensitivity'])
        metrics_df.loc[len(metrics_df.index)] = [
            'avg', metrics_df['specificity'].mean(), metrics_df['sensitivity'].mean()]
        metrics_df.set_index('activity', inplace=True)

    if voting == True:
        X_train, X_test, y_train, y_test = train_test_split(
            activities_df, activities_target, random_state=35)
        activity_rfc.fit(X_train, y_train)
        activities_full = X_test
        activities_full['ACTIVITY'] = y_test
        pred_vote, true_vote = voting_split(
            activities_full, activity_rfc, 'ACTIVITY')
        vote_acc = accuracy_score(true_vote, pred_vote)
        print(f"Voting accuracy: {vote_acc}")
        activity_cm = confusion_matrix(
            true_vote, pred_vote, labels=list(activities))
        activity_metrics = []
        for activity in activities:
            _, vote_recall, _, _ = precision_recall_fscore_support(
                np.array(true_vote) == activity, np.array(pred_vote) == activity)
            activity_metrics.append(
                [activities[activity], vote_recall[0], vote_recall[1]])
        metrics_df = pd.DataFrame(activity_metrics, columns=[
                                       "activity", 'specificity', 'sensitivity'])
        metrics_df.loc[len(metrics_df.index)] = [
            'avg', metrics_df['specificity'].mean(), metrics_df['sensitivity'].mean()]
        metrics_df.set_index('activity', inplace=True)
    
    retorno = Results(activity_cm,activity_metrics,activities,metrics_df)
    return retorno