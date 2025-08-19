import pandas as pd
import ast  # For safely parsing list-like strings
import numpy as np 
from env.cyborg_env import CyborgInsectEnv
import os 


def parse_pose(pose_str):
    return ast.literal_eval(pose_str)  # Converts '[x, y, heading]' to [x, y, heading]

def encode_action(stim_dir, freq):
    """
    Convert stimulation direction and frequency to action index (0-12).
    
    Args:
        stim_dir: Stimulation direction ("Left", "Right), "Both", "")
        freq_idx: Frequency index (10, 20, 30, 40)
    
    Returns:
        action_idx: Integer from 0 -> 12
    """
    stim_dirs = ["Left", "Right", "Both"]
    
    # Check if inputs are valid
    if stim_dir not in stim_dirs:
        return 12  # No Action 
    
    # Find direction index
    d_idx = stim_dirs.index(stim_dir)
    
    # Calculate action index
    action_idx = d_idx * 4 + freq/10
    
    return action_idx


###### DEFINE PATH #########
#Image height and lengths
image_height = 800
image_length = 1200

# Generate parameter t for one full sine cycle
t = np.linspace(0, 2 * np.pi, 1000)

# x runs from 50 to image_length - 50
x = 100 + (image_length - 150) * t / (2 * np.pi)

# y is a sine wave with amplitude image_height
y = 400 + (image_height/2 - 100) * np.sin(t)
path = np.column_stack((x, y))
path_int = np.round(path).astype(np.int32)


env = CyborgInsectEnv(
    path=path_int  # Example path
)

def process_csv_file(file_path, output_folder="formatted_data"):
    """Process a single CSV file and save the transformed data"""
    # Load your raw CSV file
    raw_df = pd.read_csv(file_path)
    
    # Preallocate lists
    heading_error_list = []
    path_distance_list = []
    progress_list = []
    action_list = []
    
    # Loop through your data
    for idx, row in raw_df.iterrows():

        j = str(row["arduino_data"])
        if isinstance(j, str) and j.strip():
            try: 
                direction, number = j.split(", ")
                freq = int(number)
            except ValueError: 
                direction = ""
                freq = ""

        action = encode_action(direction, freq)
        # Parse pose
        pose = parse_pose(row["pose"])
        x, y, heading = pose
        env.position = np.array([x, y])
        env.heading = heading
        
        state = env._get_state()
        disc_heading = state[0]
        min_distance = state[1]
        progress_along_path = state[2]
        
        # Only keep rows with valid actions
        if action >= 0:
            heading_error_list.append(disc_heading)
            path_distance_list.append(min_distance)
            progress_list.append(progress_along_path)
            action_list.append(action)
    
    # Build new DataFrame
    new_df = pd.DataFrame({
        "heading_error": heading_error_list,
        "path_distance": path_distance_list,
        "progress": progress_list,
        "action": action_list,
    })
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate output filename
    base_filename = os.path.basename(file_path)
    name, ext = os.path.splitext(base_filename)
    output_filepath = os.path.join(output_folder, f"formatted_{name}{ext}")
    
    # Save as new CSV for BC training
    new_df.to_csv(output_filepath, index=False)
    print(f"Processed: {file_path} -> {output_filepath}")
    
    return output_filepath

def process_all_csv_in_folder(input_folder, output_folder="formatted_data"):
    """Process all CSV files in a folder"""
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return []
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    output_files = []
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        # try:
        output_file = process_csv_file(file_path, output_folder)
        output_files.append(output_file)
        # except Exception as e:
        #     print(f"Error processing {csv_file}: {str(e)}")
    
    print(f"Successfully processed {len(output_files)} files")
    return output_files

# Usage
input_folder = r"G:\biorobotics\data\ClosedLoopControl\MiscDataCollection\500msDurationReformat"
output_folder = r"G:\biorobotics\data\ClosedLoopControl\MiscDataCollection\BC_Folder"
processed_files = process_all_csv_in_folder(input_folder, output_folder)