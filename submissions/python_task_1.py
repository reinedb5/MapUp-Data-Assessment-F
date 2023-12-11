import pandas as pd
import numpy as np


##### Task 1 (1)

def generate_car_matrix(df: pd.DataFrame) -> pd.DataFrame:
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')
    car_matrix = car_matrix.fillna(0)
    np.fill_diagonal(car_matrix.values, 0)
    return car_matrix



df = pd.read_csv("datasets/dataset-1.csv")
result_matrix = generate_car_matrix(df)
print(result_matrix)


##### Task 1 (2)


def get_type_count(df)->dict:
    df['car_type'] = pd.cut(df['car'],
                            bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'])
     # Count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()
    # Sorting the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))
    return sorted_type_counts


df = pd.read_csv("datasets/dataset-1.csv")
result = get_type_count(df)
print(result)


##### Task 1 (3)



def get_bus_indexes(df)->list:
    bus_mean = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    return bus_indexes



df = pd.read_csv("datasets/dataset-1.csv")
result = get_bus_indexes(df)
print(result)


##### Task 1 (4)


def filter_routes(df) -> list:
    # Group by 'route' and calculating the average of 'truck' values
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filtering routes where the average 'truck' value is greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sorting the list of routes
    sorted_filtered_routes = sorted(filtered_routes)

    return sorted_filtered_routes



df = pd.read_csv("datasets/dataset-1.csv")
result = filter_routes(df)
print(result)


##### Task 1 (5)


import numpy as np


def multiply_matrix(matrix)->pd.DataFrame:
    # Applying custom conditions to modify matrix values
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Rounding the values to 1 decimal place
    modified_matrix_rounded = modified_matrix.round(1)

    return modified_matrix_rounded


df = pd.read_csv("datasets/dataset-1.csv")
car_matrix = generate_car_matrix(df)
result_matrix = multiply_matrix(car_matrix)
print(result_matrix)


##### Task 1 (6)

from datetime import datetime, timedelta


def time_check(df) -> pd.Series:
    result = pd.Series(True, index=df.groupby(['id', 'id_2']).size().index)

    for (id, id_2), group in df.groupby(['id', 'id_2']):
        try:
            start_time = pd.to_datetime(group['startDay'] + ' ' + group['startTime'])
            end_time = pd.to_datetime(group['endDay'] + ' ' + group['endTime'])

            # Setting a minimum valid year (adjust as needed)
            min_valid_year = 1970

            # Replacing invalid timestamps with NaN
            start_time.loc[start_time.dt.year < min_valid_year] = pd.NaT
            end_time.loc[end_time.dt.year < min_valid_year] = pd.NaT

            # Dropping rows with NaN values
            group = group.dropna(subset=['startDay', 'startTime', 'endDay', 'endTime'])

        except pd.errors.OutOfBoundsDatetime as e:
            print(f"Error in group ({id}, {id_2}): {e}")
            print("Problematic start times:")
            print(group.loc[start_time.dt.year < min_valid_year, ['startDay', 'startTime']])
            print("Problematic end times:")
            print(group.loc[end_time.dt.year < min_valid_year, ['endDay', 'endTime']])
            result[(id, id_2)] = False
            continue

        date_range = pd.date_range(start=group['startDay'].min(), end=group['endDay'].max(), freq='T')

        if len(date_range) != len(group):
            result[(id, id_2)] = False

        if not set(date_range.dayofweek).issubset(set(range(7))):
            result[(id, id_2)] = False

    return result



# Loading the dataset-2.csv as a DataFrame
df = pd.read_csv("datasets/dataset-2.csv")

# Checking the timestamps in the dataset
result = time_check(df)

# Printing the result
print(result)