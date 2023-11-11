import re

def remove_consecutive_commas(input_file, output_file):
    with open(input_file, 'r', newline='') as infile:
        content = infile.read()

    cleaned_content = re.sub(r',+', ',', content)

    with open(output_file, 'w', newline='') as outfile:
        outfile.write(cleaned_content)

remove_consecutive_commas("squat_labels.csv", "cleaned_labels.csv")