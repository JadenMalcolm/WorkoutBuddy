import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('pre_labels.csv')

manual_labels = {
    1: [0, 1, 1, 1, 1],
    2: [1, 0, 0, 0, 0],
    3: [1, 0, 1, 0, 0],
    4: [0, 0, 0, 0, 0],
    5: [0, 1, 1, 1, 1],
    6: [1, 0, 0, 0, 0],
    7: [1, 0, 1, 0, 0],
    8: [0, 0, 0, 0, 0],
    9: [0, 1, 1, 1, 1],
    10: [1, 0, 0, 0, 0],
    11: [1, 0, 1, 0, 0],
}

# Define the label columns
label_columns = ['good_form', 'not_deep_enough', 'back_bent', 'knees_too_forward', 'heels_up']

# Iterate through the DataFrame and set the labels according to the sequence_id
for sequence_id, labels in manual_labels.items():
    # Convert labels to a numpy array with dtype int to prevent float conversion
    labels_array = np.array(labels, dtype=int)
    df.loc[df['sequence_id'] == sequence_id, label_columns] = labels_array

# After updating the labels, convert the columns to integers if there are no NaN values
for column in label_columns:
    # Convert the column to integers, but only if there are no NaN values
    if df[column].notnull().all():
        df[column] = df[column].astype(int)

# Save the updated DataFrame to a new CSV file
df.to_csv('squat_labels.csv', index=False)
if 'Unnamed: 8' in df.columns:
    df = df.drop('Unnamed: 8', axis=1)

df.to_csv('squat_labels.csv', index=False)
df = pd.read_csv('squat_labels.csv', dtype={col: int for col in label_columns})

