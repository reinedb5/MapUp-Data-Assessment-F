import pandas as pd

##### Task 2 (1): Distance Matrix Calculation

def calculate_distance_matrix(df)->pd.DataFrame():
    # Create a dictionary to store distances
    distances = {}

    # Initialize the dictionary with known distances
    for _, row in df.iterrows():
        distances[(row['id_start'], row['id_end'])] = row['distance']
        distances[(row['id_end'], row['id_start'])] = row['distance']  # Bidirectional

    # Get all unique IDs
    nodes = sorted(list(set(df['id_start'].unique()).union(df['id_end'].unique())))

    # Initialize an empty DataFrame for the distance matrix
    distance_matrix = pd.DataFrame(index=nodes, columns=nodes)

    # Calculate distances between IDs
    for source in nodes:
        for target in nodes:
            if source == target:
                distance_matrix.loc[source, target] = 0
            else:
                key = (source, target)
                if key in distances:
                    distance_matrix.loc[source, target] = distances[key]
                else:
                    # If no direct route, set distance to infinity
                    distance_matrix.loc[source, target] = float('inf')

    # Update distances based on known routes
    for intermediate in nodes:
        for source in nodes:
            for target in nodes:
                if distance_matrix.loc[source, intermediate] + distance_matrix.loc[intermediate, target] < distance_matrix.loc[source, target]:
                    distance_matrix.loc[source, target] = distance_matrix.loc[source, intermediate] + distance_matrix.loc[intermediate, target]

    return distance_matrix



df = pd.read_csv("datasets/dataset-3.csv")
print(calculate_distance_matrix(df))


# In[12]:


def unroll_distance_matrix(df)->pd.DataFrame():
    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate over the rows of the distance matrix
    for i, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        # Append to the list of DataFrames
        dfs.append(pd.DataFrame({'id_start': [id_start], 'id_end': [id_end], 'distance': [distance]}))

    # Concatenate all DataFrames into a single DataFrame
    unrolled_df = pd.concat(dfs, ignore_index=True)

    return unrolled_df


# #### Task 2 (2): Unroll Distance Matrix



df = pd.read_csv("datasets/dataset-3.csv")
unrolled_df = unroll_distance_matrix(df)
print(unrolled_df)


# #### Task 2 (3): Finding IDs within Percentage Threshold


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate over the rows of the distance matrix
    for id_start in df.index:
        for id_end, distance in df.loc[id_start].items():
            if id_start != id_end:  # Exclude same id_start to id_end
                # Append to the list of DataFrames
                dfs.append(pd.DataFrame({'id_start': [id_start], 'id_end': [id_end], 'distance': [distance]}))

    # Concatenate all DataFrames into a single DataFrame
    unrolled_df = pd.concat(dfs, ignore_index=True)

    return unrolled_df


df = pd.read_csv("datasets/dataset-3.csv")
unrolled_df = unroll_distance_matrix(df)
result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id=1001418)
print(result_df)


# #### Task 2 (4): Calculate Toll Rate


def calculate_toll_rate(df)->pd.DataFrame():
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Initialize new columns for toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df


df = pd.read_csv("datasets/dataset-3.csv")
unrolled_df = unroll_distance_matrix(df)
df_with_toll_rates = calculate_toll_rate(unrolled_df)
print(df_with_toll_rates)


# #### Task 2 (5): Calculate Time-Based Toll Rates

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    if 'start_time' not in df.columns:
        raise KeyError("Column 'start_time' not found in the DataFrame.")

    df['start_day'] = df['start_time'].dt.strftime('%A')
    df['start_time'] = df['start_time'].dt.time
    df['end_day'] = df['end_time'].dt.strftime('%A')
    df['end_time'] = df['end_time'].dt.time

    df['discount_factor'] = 1.0

    def apply_discount_factor(row):
        if row['start_day'] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if time(0, 0, 0) <= row['start_time'] < time(10, 0, 0):
                return 0.8
            elif time(10, 0, 0) <= row['start_time'] < time(18, 0, 0):
                return 1.2
            elif time(18, 0, 0) <= row['start_time'] <= time(23, 59, 59):
                return 0.8
        elif row['start_day'] in ['Saturday', 'Sunday']:
            return 0.7
        return 1.0

    df['discount_factor'] = df.apply(apply_discount_factor, axis=1)

    return df


df = pd.read_csv("datasets/dataset-3.csv")
reference_id = 1001400
result_df = find_ids_within_ten_percentage_threshold(df, reference_id)
print(result_df)
df_with_time_based_toll_rates = calculate_time_based_toll_rates(df[df['id_start'].isin(result_df['id_start'])])
print(df_with_time_based_toll_rates)