import pandas as pd
from scipy.io import arff
from itertools import combinations

Person = dict[str, str | pd.DataFrame]  # TypeAlias


def processed_dataframe(raw: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    """
    Receives a raw dataframe resulting from reading the arff file and returns a
    copy of that dataframe with the class and ACTIVITY columns decoded and without unused columns.
    """
    result = raw.drop(columns=cols_to_drop)
    result.columns = [col.replace('"', "") for col in result]
    result['ACTIVITY'] = result['ACTIVITY'].str.decode('utf-8')
    result['class'] = result['class'].str.decode('utf-8')
    return result


columns_to_be_dropped = []
for n in ['X', 'Y', 'Z']:
    columns_to_be_dropped.append(f'"{n}VAR"')
    for i in range(13):
        columns_to_be_dropped.append(f'"{n}MFCC{i}"')
for n in combinations(['X', 'Y', 'Z'], 2):
    columns_to_be_dropped.append(f'"{"".join(n)}COS"')
    columns_to_be_dropped.append(f'"{"".join(n)}COR"')

sensors = ["phone_accel", "watch_accel", "phone_gyro", "watch_gyro"]
activities = {'A': 'walking',
              'B': 'jogging',
              'C': 'stairs',
              'D': 'sitting',
              'E': 'standing',
              'F': 'typing',
              'G': 'teeth',
              'H': 'soup',
              'I': 'chips',
              'J': 'pasta',
              'K': 'drinking',
              'L': 'sandwich',
              'M': 'kicking',
              'O': 'catch',
              'P': 'dribbling',
              'Q': 'writing',
              'R': 'clapping',
              'S': 'folding'}

people: list[Person] = [{
    'id': i,
    'phone_accel': processed_dataframe(pd.DataFrame(arff.loadarff(f"wisdm-dataset/arff_files/phone/accel/data_{i}_accel_phone.arff")[0]), columns_to_be_dropped),
    'phone_gyro': processed_dataframe(pd.DataFrame(arff.loadarff(f"wisdm-dataset/arff_files/phone/gyro/data_{i}_gyro_phone.arff")[0]), columns_to_be_dropped),
    'watch_accel': processed_dataframe(pd.DataFrame(arff.loadarff(f"wisdm-dataset/arff_files/watch/accel/data_{i}_accel_watch.arff")[0]), columns_to_be_dropped),
    'watch_gyro': processed_dataframe(pd.DataFrame(arff.loadarff(f"wisdm-dataset/arff_files/watch/gyro/data_{i}_gyro_watch.arff")[0]), columns_to_be_dropped)
} for i in range(1600, 1651) if i != 1614]  # person 1614 missing in files

every_sensor_combination = [[i]for i in sensors] + \
    [list(i) for i in list(combinations(sensors, 2))]+[sensors]
