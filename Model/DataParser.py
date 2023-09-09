import csv
import json

csv_file_path = 'OrderedData.csv'

keypoints_data = []

# Define a dictionary to map keypoint indices to body part labels
keypoint_labels = {
    1: "L_Toe",
    2: "L_Ankle",
    3: "L_Knee",
    4: "L_Pelvis",
    5: "L_Hip",
    6: "L_Shoulder",
    7: "L_Elbow",
    8: "L_Wrist",
    9: "R_Toe",
    10: "R_Ankle",
    11: "R_Knee",
    12: "R_Pelvis",
    13: "R_Hip",
    14: "R_Shoulder",
    15: "R_Elbow",
    16: "R_Wrist",
    17: "Bottom_spine",
    18: "Middle_spine",
    19: "Top_Spine",
}
x = 0
# Read CSV file and extract keypoints data
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip header row
    for line in csv_reader:
        json_data = json.loads(line[5])  # Extract the JSON data from the 6th column (index 5)
        cx = json_data.get("cx", None)
        cy = json_data.get("cy", None)
        line_index = int(line[4])  # Extract the line number as the index
        x += 1
        keypoints_data.append({
                "cx": cx,
                "cy": cy,
                "label": keypoint_labels[line_index + 1],
        })

    for data in keypoints_data:
        print(json.dumps(data))
