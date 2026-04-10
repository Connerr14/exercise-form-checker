import pandas as pd
import os

#  File Configuration, and setting the headers.
INPUT_FILE = 'squat_dataset.csv'
OUTPUT_FILE = 'labeled_squat_data.csv'
HEADERS = [
    'lean', 'asymmetry', 'right_angle', 'left_angle', 
    'force_right', 'force_left', 'force_diff', 'body_ratio', 'label'
]

""" This function categorizes the line depending on its value"""
def categorize_rep(row):
    # Check for safety errors
    if row['force_diff'] == 0:
        return 'Force_Imbalance'
    if row['asymmetry'] > 15:
        return 'Uneven_Weight'
    

    # Baseline lean is 10 degrees. We add 5 degrees of "allowance" 
    # for every unit of body_ratio (Femur/Torso).
    # Long-legged users get a higher threshold; short-legged users get a stricter one.
    dynamic_lean_limit = 10 + (row['body_ratio'] * 5)

    if row['lean'] > dynamic_lean_limit:
        return 'Forward_Lean'
    if row['lean'] > 12:
        return 'Forward_Lean'

    # Gets the average angle
    avg_angle = (row['right_angle'] + row['left_angle']) / 2
    
    # Check for stationary markers
    if avg_angle > 145:
        return 'Standing'
    if avg_angle < 90:
        return 'At_Bottom'

    # Movement directions using smoothed diff
    if row['angle_diff'] < -0.1: 
        return 'Descending'
    elif row['angle_diff'] > 0.1:
        return 'Ascending'
    
    return 'Too_Shallow'

def main():
    # Ensuring the input file is found
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    
    # Getting the csv file
    df = pd.read_csv(INPUT_FILE, names=HEADERS, header=0)
    df.columns = df.columns.str.strip()

    # Temporal smoothing logic
    df['avg_angle_raw'] = (df['right_angle'] + df['left_angle']) / 2
    
    # Apply a rolling mean over 5 frames (0.15 seconds) (removes high-frequency noise from the camera)
    df['smoothed_angle'] = df['avg_angle_raw'].rolling(window=5, center=True).mean()
    
    # fill NaN values at start/end with the raw angle so we don't lose data
    df['smoothed_angle'] = df['smoothed_angle'].fillna(df['avg_angle_raw'])

    #  Calculate difference based on the SMOOTHED signal
    df['angle_diff'] = df['smoothed_angle'].diff()
    df['label'] = df.apply(categorize_rep, axis=1)

    # Removing the helper columns
    df = df.drop(columns=['avg_angle_raw', 'smoothed_angle'])
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Smoothed data saved to {OUTPUT_FILE}")
    print(df['label'].value_counts())

if __name__ == "__main__":
    main()