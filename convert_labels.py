import os

# Base folder path where all label folders are stored
base_path = "labels"

# Mapping dictionary
species_to_id = {
    "Zebra": 0,
    "Tiger": 1,
    "Rhinoceros": 2,
    "Ostrich": 3,
    "Lion": 4,
    "Leopard": 5,
    "Horse": 6,
    "Jaguar": 7,
    "Harbor seal": 8,
    "Goat": 9,
    "Giraffe": 10,
    "Fox": 11,
    "Elephant": 12,
    "Eagle": 13,
    "Deer": 14,
    "Crab": 15,
    "Chicken": 16,
    "Caterpillar": 17,
    "Cheetah": 18,
    "Butterfly": 19
}

# Walk through all folders inside "labels"
for species_folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, species_folder)
    
    if not os.path.isdir(folder_path):
        continue  # skip if not a folder
    
    # Process each .txt file in the species folder
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".txt"):
            continue
        
        file_path = os.path.join(folder_path, file_name)
        
        # Read lines from file
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                continue  # skip invalid lines
            
            label_name = parts[0]
            # Convert label to corresponding number
            if label_name in species_to_id:
                label_num = species_to_id[label_name]
                parts[0] = str(label_num)
            else:
                print(f"⚠️ Unknown label '{label_name}' in {file_path}")
                continue
            
            new_lines.append(" ".join(parts))
        
        # Overwrite the file with converted labels
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))
        
        print(f"✅ Converted: {file_path}")

print("\n🎯 Conversion complete for all label files!")
