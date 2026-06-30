import json
import random
import os

def create_subset(layer1_path, layer2_path, output_dir, subset_size=5000, seed=42):
    """
    Subsets the Recipe1M dataset by randomly sampling a smaller number of recipes 
    that are guaranteed to have images (exist in both layer1 and layer2).
    """
    random.seed(seed)
    
    print(f"Loading layer2 from {layer2_path}...")
    try:
        with open(layer2_path, 'r') as f:
            layer2_data = json.load(f)
    except Exception as e:
        print(f"Error loading {layer2_path}: {e}")
        return

    print(f"Loading layer1 from {layer1_path}...")
    try:
        with open(layer1_path, 'r') as f:
            layer1_data = json.load(f)
    except Exception as e:
        print(f"Error loading {layer1_path}: {e}")
        return

    # Create mappings for quick access
    # layer2 usually contains a list of objects: [{"id": "...", "images": [...]}, ...]
    print("Mapping layer2 IDs...")
    layer2_dict = {item['id']: item for item in layer2_data}
    
    print("Mapping layer1 IDs...")
    # layer1 contains: [{"id": "...", "title": "...", "ingredients": [...], "instructions": [...]}, ...]
    layer1_dict = {item['id']: item for item in layer1_data}
    
    # Find common IDs (Recipes that have both text and images)
    common_ids = list(set(layer1_dict.keys()).intersection(set(layer2_dict.keys())))
    print(f"Found {len(common_ids)} recipes that have both text and images.")
    
    if len(common_ids) < subset_size:
        print(f"Warning: Requested subset size ({subset_size}) is larger than available common recipes ({len(common_ids)}). Using {len(common_ids)}.")
        subset_size = len(common_ids)
        
    # Randomly sample IDs
    print(f"Sampling {subset_size} recipes...")
    sampled_ids = random.sample(common_ids, subset_size)
    
    # Build the small subsets
    layer1_small = [layer1_dict[rid] for rid in sampled_ids]
    layer2_small = [layer2_dict[rid] for rid in sampled_ids]
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the small JSONs
    layer1_out = os.path.join(output_dir, "layer1_small.json")
    layer2_out = os.path.join(output_dir, "layer2_small.json")
    
    print(f"Saving {layer1_out}...")
    with open(layer1_out, 'w') as f:
        json.dump(layer1_small, f, indent=2)
        
    print(f"Saving {layer2_out}...")
    with open(layer2_out, 'w') as f:
        json.dump(layer2_small, f, indent=2)
        
    # Also create a list of image filenames to easily copy them later
    print("Extracting image filenames for the subset...")
    image_filenames = []
    for item in layer2_small:
        if 'images' in item:
            for img in item['images']:
                if isinstance(img, dict) and 'id' in img:
                    image_filenames.append(img['id'])
                elif isinstance(img, str):
                    image_filenames.append(img)
                    
    img_list_out = os.path.join(output_dir, "subset_image_list.txt")
    with open(img_list_out, 'w') as f:
        for img_name in image_filenames:
            f.write(f"{img_name}\n")
            
    print("\n--- Subset Creation Complete! ---")
    print(f"1. Saved layer1_small.json ({len(layer1_small)} items)")
    print(f"2. Saved layer2_small.json ({len(layer2_small)} items)")
    print(f"3. Saved subset_image_list.txt (List of {len(image_filenames)} images you need to copy)")
    print("\nYou can use a bash/python script to read 'subset_image_list.txt' and copy only those images from your massive image folder to a new 'images_small' folder.")

if __name__ == "__main__":
    # USER: Update these paths to where you actually downloaded your Recipe1M dataset
    RAW_LAYER1_PATH = "path/to/downloaded/layer1.json"
    RAW_LAYER2_PATH = "path/to/downloaded/layer2.json"
    OUTPUT_DIRECTORY = "data_subset" # Will be created in current directory
    
    # Change subset_size to how many recipes you want to train on (e.g., 5000, 10000)
    create_subset(
        layer1_path=RAW_LAYER1_PATH, 
        layer2_path=RAW_LAYER2_PATH, 
        output_dir=OUTPUT_DIRECTORY, 
        subset_size=5000 
    )
