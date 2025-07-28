import argparse
import numpy as np
import json
import os


def prepare_semi_supervised_data(input_file, labeled_count, unlabeled_count, output_dir, seed=1):
    """Prepare dataset for Semi-supervised learning from COCO format annotations
    
    Args:
        input_file: Path to input COCO format JSON file
        labeled_count: Exact number of images for labeled subset
        unlabeled_count: Exact number of images for unlabeled subset
        output_dir: Directory to save output files
        seed: Random seed for dataset split
    """

    def _save_anno(filename, images, annotations):
        """Save annotation to JSON file."""
        print(f">> Processing {filename} ({len(images)} images, {len(annotations)} annotations)")
        
        new_anno = {
            "images": images,
            "annotations": annotations,
            "licenses": anno.get("licenses", []),
            "categories": anno["categories"],
            "info": anno.get("info", {})
        }
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(new_anno, f)
        
        print(f">> Saved {filename} ({len(images)} images, {len(annotations)} annotations)")

    # Set random seed
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load input annotations
    print(f">> Loading annotations from {input_file}")
    with open(input_file, 'r') as f:
        anno = json.load(f)
    
    image_list = anno["images"]
    total_images = len(image_list)
    
    print(f">> Total images available: {total_images}")
    print(f">> Requested labeled: {labeled_count}, unlabeled: {unlabeled_count}")
    
    # Validate input
    if labeled_count + unlabeled_count > total_images:
        raise ValueError(f"Requested images ({labeled_count + unlabeled_count}) exceed available images ({total_images})")
    
    # Randomly select images for labeled and unlabeled sets
    total_needed = labeled_count + unlabeled_count
    selected_indices = np.random.choice(range(total_images), size=total_needed, replace=False)
    
    # Split into labeled and unlabeled
    labeled_indices = set(selected_indices[:labeled_count])
    unlabeled_indices = set(selected_indices[labeled_count:])
    
    # Separate images
    labeled_images = []
    unlabeled_images = []
    labeled_image_ids = set()
    unlabeled_image_ids = set()
    
    for i, image in enumerate(image_list):
        if i in labeled_indices:
            labeled_images.append(image)
            labeled_image_ids.add(image["id"])
        elif i in unlabeled_indices:
            unlabeled_images.append(image)
            unlabeled_image_ids.add(image["id"])
    
    # Separate annotations based on image IDs
    labeled_annotations = []
    unlabeled_annotations = []
    
    for annotation in anno["annotations"]:
        if annotation["image_id"] in labeled_image_ids:
            labeled_annotations.append(annotation)
        elif annotation["image_id"] in unlabeled_image_ids:
            unlabeled_annotations.append(annotation)
    
    # Save labeled and unlabeled datasets
    _save_anno("labeled.json", labeled_images, labeled_annotations)
    _save_anno("unlabeled.json", unlabeled_images, unlabeled_annotations)
    
    print(f">> Successfully created semi-supervised split with seed {seed}")


if __name__ == "__main__":
    input_file = "data2/coco_ann/train/annotations/coco_instances.json"  # Path to your COCO annotations file
    labeled_count = 250  # Number of labeled images
    unlabeled_count = 1000  # Number of unlabeled images
    output_dir = "data2/ssl_split"  # Directory to save the output files
    seed = 42  # Random seed for reproducibility
    prepare_semi_supervised_data(
        input_file, labeled_count, unlabeled_count, output_dir, seed
    )
    print(">> Semi-supervised dataset preparation complete.")