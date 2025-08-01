import os
import json
import logging
from pathlib import Path
from collections import defaultdict
import supervisely as sly
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coco_to_supervisely_conversion.log'),
        logging.StreamHandler()
    ]
)

# Configuration
coco_predictions_path = 'evaluation/cocoDt.json'  # COCO predictions file
output_dir = 'output/sly_project/ann'  # Supervisely annotations output
meta_path = 'output/sly_project/meta.json'  # Path to save ProjectMeta
confidence_threshold = 0.05  # Minimum confidence threshold for predictions

# Create output directory
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(meta_path), exist_ok=True)

def load_coco_data(coco_path):
    """Load COCO predictions file."""
    logging.info(f"Loading COCO predictions from {coco_path}...")
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    
    logging.info(f"Loaded {len(coco_data.get('annotations', []))} predictions")
    logging.info(f"Found {len(coco_data.get('images', []))} images")
    logging.info(f"Found {len(coco_data.get('categories', []))} categories")
    
    return coco_data

def create_supervisely_meta(coco_categories):
    """Create Supervisely ProjectMeta from COCO categories."""
    logging.info("Creating Supervisely ProjectMeta from COCO categories...")
    
    obj_classes = []
    category_id_to_obj_class = {}
    
    # Sort categories by ID for consistent ordering
    sorted_categories = sorted(coco_categories, key=lambda x: x['id'])
    
    for category in sorted_categories:
        cat_id = category['id']
        cat_name = category['name']
        
        # Create object class for bounding box detection
        obj_class = sly.ObjClass(cat_name, sly.Rectangle)
        obj_classes.append(obj_class)
        category_id_to_obj_class[cat_id] = obj_class
        
        logging.debug(f"Created object class: {cat_name} (id: {cat_id})")
    
    # Create tag meta for confidence scores
    tag_metas = [
        sly.TagMeta('confidence', sly.TagValueType.ANY_NUMBER)
    ]
    
    # Create ProjectMeta
    project_meta = sly.ProjectMeta(obj_classes=obj_classes, tag_metas=tag_metas)
    
    logging.info(f"Created ProjectMeta with {len(obj_classes)} object classes")
    
    return project_meta, category_id_to_obj_class

def convert_coco_bbox_to_sly_rectangle(coco_bbox):
    """Convert COCO bbox format [x, y, width, height] to Supervisely Rectangle."""
    x, y, width, height = coco_bbox
    
    # COCO bbox: [x_min, y_min, width, height]
    # Supervisely Rectangle: (top, left, bottom, right)
    left = int(x)
    top = int(y)
    right = int(x + width)
    bottom = int(y + height)
    
    return sly.Rectangle(top=top, left=left, bottom=bottom, right=right)

def process_predictions(coco_data, project_meta, category_id_to_obj_class):
    """Process COCO predictions and create Supervisely annotations."""
    
    # Create mappings
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Group predictions by image_id
    predictions_by_image = defaultdict(list)
    
    # Filter predictions by confidence threshold
    filtered_count = 0
    for pred in coco_data['annotations']:
        if pred.get('score', 0) >= confidence_threshold:
            predictions_by_image[pred['image_id']].append(pred)
        else:
            filtered_count += 1
    
    logging.info(f"Filtered out {filtered_count} predictions below confidence threshold {confidence_threshold}")
    
    # Get confidence tag meta
    confidence_tag_meta = project_meta.get_tag_meta('confidence')
    
    # Process each image
    converted_count = 0
    total_objects = 0
    
    for image_id, predictions in tqdm(predictions_by_image.items(), desc="Converting annotations"):
        # Get image info
        if image_id not in image_id_to_info:
            logging.warning(f"Image ID {image_id} not found in images list, skipping...")
            continue
        
        image_info = image_id_to_info[image_id]
        img_filename = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        img_size = (img_height, img_width)  # Supervisely uses (height, width)
        
        # Create Supervisely labels for this image
        sly_labels = []
        
        for pred in predictions:
            # Get category/class
            category_id = pred['category_id']
            if category_id not in category_id_to_obj_class:
                logging.warning(f"Category ID {category_id} not found in categories, skipping...")
                continue
            
            obj_class = category_id_to_obj_class[category_id]
            
            # Convert bbox
            rectangle = convert_coco_bbox_to_sly_rectangle(pred['bbox'])
            
            # Create confidence tag
            confidence_value = round(pred.get('score', 0), 4)  # Round to 4 decimal places
            confidence_tag = sly.Tag(confidence_tag_meta, value=confidence_value)
            
            # Create label with tag
            sly_label = sly.Label(
                geometry=rectangle,
                obj_class=obj_class,
                tags=[confidence_tag]
            )
            sly_labels.append(sly_label)
        
        # Create Supervisely annotation
        annotation = sly.Annotation(img_size, sly_labels)
        
        # Save annotation
        # Output filename: image filename + .json
        output_filename = f"{img_filename}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(annotation.to_json(), f, indent=2)
        
        converted_count += 1
        total_objects += len(sly_labels)
        
        logging.debug(f"Converted {img_filename} -> {output_filename} ({len(sly_labels)} objects)")
    
    # Process images without predictions (create empty annotations)
    all_image_ids = set(image_id_to_info.keys())
    images_with_predictions = set(predictions_by_image.keys())
    images_without_predictions = all_image_ids - images_with_predictions
    
    logging.info(f"Creating empty annotations for {len(images_without_predictions)} images without predictions...")
    
    for image_id in images_without_predictions:
        image_info = image_id_to_info[image_id]
        img_filename = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        img_size = (img_height, img_width)
        
        # Create empty annotation
        annotation = sly.Annotation(img_size)
        
        # Save annotation
        output_filename = f"{img_filename}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(annotation.to_json(), f, indent=2)
        
        converted_count += 1
    
    return converted_count, total_objects

def main():
    """Main conversion function."""
    
    # Load COCO data
    coco_data = load_coco_data(coco_predictions_path)
    
    # Create Supervisely meta
    project_meta, category_id_to_obj_class = create_supervisely_meta(coco_data['categories'])
    
    # Save ProjectMeta
    with open(meta_path, 'w') as f:
        json.dump(project_meta.to_json(), f, indent=2)
    logging.info(f"Saved ProjectMeta to {meta_path}")
    
    # Process predictions
    converted_count, total_objects = process_predictions(
        coco_data, project_meta, category_id_to_obj_class
    )
    
    # Summary
    logging.info("=" * 50)
    logging.info("CONVERSION SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Input COCO file: {coco_predictions_path}")
    logging.info(f"Confidence threshold: {confidence_threshold}")
    logging.info(f"Total images processed: {converted_count}")
    logging.info(f"Total objects converted: {total_objects}")
    logging.info(f"Average objects per image: {total_objects/converted_count:.2f}" if converted_count > 0 else "N/A")
    logging.info(f"Object classes: {len(project_meta.obj_classes)}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"ProjectMeta saved to: {meta_path}")
    logging.info("=" * 50)
    
    print(f"\nConversion completed!")
    print(f"Supervisely annotations saved to: {output_dir}")
    print(f"ProjectMeta saved to: {meta_path}")

if __name__ == "__main__":
    main()