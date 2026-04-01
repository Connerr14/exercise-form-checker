import pandas as pd
import os

#  File Configuration
INPUT_FILE = 'squat_dataset.csv'
OUTPUT_FILE = 'labeled_squat_data.csv'

HEADERS = [
    'lean', 'asymmetry', 'right_angle', 'left_angle', 
    'force_right', 'force_left', 'force_diff', 'label'
]

def categorize_rep(row):
    # Check for safety errors
    if row['force_diff'] == 0:
        return 'Force_Imbalance'
    if row['asymmetry'] > 15:
        return 'Uneven_Weight'
    if row['lean'] > 12:
        return 'Forward_Lean'

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
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE, names=HEADERS, header=0)
    df.columns = df.columns.str.strip()

    # Temporal smoothing logic
    # Get raw average angle
    df['avg_angle_raw'] = (df['right_angle'] + df['left_angle']) / 2
    
    # Apply a rolling mean over 5 frames (0.15 seconds)
    # This removes high-frequency noise from the camera
    df['smoothed_angle'] = df['avg_angle_raw'].rolling(window=5, center=True).mean()
    
    # fill NaN values at start/end with the raw angle so we don't lose data
    df['smoothed_angle'] = df['smoothed_angle'].fillna(df['avg_angle_raw'])

    #  Calculate difference based on the SMOOTHED signal
    df['angle_diff'] = df['smoothed_angle'].diff()

    df['label'] = df.apply(categorize_rep, axis=1)

    df = df.drop(columns=['avg_angle_raw', 'smoothed_angle'])
    df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 30)
    print(f"Success! Smoothed data saved to {OUTPUT_FILE}")
    print(df['label'].value_counts())

if __name__ == "__main__":
    main()