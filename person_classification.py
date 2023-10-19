from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from dataset_contents import activities, Person, people, sensors
from voting_split import voting_split

def person_classification_df_builder(people: list[Person], sensors: list[str], activity: str) -> pd.DataFrame:
    result_df: pd.DataFrame = None
    for person in people:
        df: pd.DataFrame = None
        for sensor in sensors:
            df = person[sensor].query(f"`ACTIVITY`=='{activity}'").drop(columns=['class','ACTIVITY']).add_suffix(sensor).reset_index(drop=True) if df is None else df.merge(
                person[sensor].query(f"`ACTIVITY`=='{activity}'").drop(columns=['class','ACTIVITY']).add_suffix(sensor).reset_index(drop=True), left_index=True, right_index=True)
        df['class']=person['guid']
        result_df = df if result_df is None else pd.concat([result_df, df])
    return result_df.dropna().sample(frac=1.0)


def main(voting: Optional[bool] = False) -> None:
    kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    people_metrics: dict[str,list[str|float]] = {'Activity':[],'Accuracy':[]}
    if voting == True:
        people_vote_metrics: dict[str,list[str|float]] = {'Activity':[],'Accuracy':[]}
    for activity in activities:
        people_df = person_classification_df_builder(people,sensors,activity)
        people_target = people_df['class']
        people_df = people_df.drop(columns=['class'])
        people_classifier = RandomForestClassifier(10, max_features='sqrt')
        score = cross_validate(people_classifier,people_df,people_target,cv=kf,scoring='accuracy')
        people_metrics['Activity'].append(activities[activity])
        people_metrics['Accuracy'].append(np.mean(score['test_score']))
        if voting == True:
            X_train, X_test, y_train, y_test = train_test_split(
                people_df, people_target, random_state=35)
            people_full = X_test
            people_full['class'] = y_test
            people_classifier.fit(X_train,y_train)
            pred_vote, true_vote = voting_split(people_full,people_classifier,'class')
            people_vote_metrics['Activity'].append(activities[activity])
            people_vote_metrics['Accuracy'].append(accuracy_score(true_vote,pred_vote))
            
        