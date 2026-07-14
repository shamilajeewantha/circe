import json
import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "inputs")
JSON_PATH = os.path.join(BASE_DIR, "data", "json", "detections.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

def batch_process():
    # 1. Load Master Annotations Map
    if not os.path.exists(JSON_PATH):
        print(f"Error: Master annotations JSON file not found at {JSON_PATH}")
        return
        
    with open(JSON_PATH, "r") as f:
        master_annotations = json.load(f)

    # Ensure output pipeline path exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Grab all images in the input directory
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

    if not images:
        print(f"No valid images found in {INPUT_DIR}")
        return

    print(f"Found {len(images)} images to process...")

    # 3. Process the batch loop
    for image_name in images:
        if image_name not in master_annotations:
            print(f"Skipping '{image_name}': No mapping coordinates inside JSON.")
            continue

        img_path = os.path.join(INPUT_DIR, image_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {image_name}")
            continue

        img_height, img_width = img.shape[:2]
        detections = master_annotations[image_name]

        for idx, detection in enumerate(detections):
            ymin, xmin, ymax, xmax = detection["box_2d"]
            label = detection["label"]

            # Scaled Denormalization coordinates
            start_x = int((xmin / 1000) * img_width)
            start_y = int((ymin / 1000) * img_height)
            end_x = int((xmax / 1000) * img_width)
            end_y = int((ymax / 1000) * img_height)

            # Styling Configurations
            box_color = (0, 165, 255)  # Alert Amber
            thickness = max(2, int(img_width * 0.003)) # Scales thickness with resolution

            # Box overlay
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, thickness)

            # Text typography badge scale properties
            font_scale = img_width * 0.0006
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            label_ymin = max(start_y, label_size[1] + 10)
            
            # Badge Background Block
            cv2.rectangle(
                img, 
                (start_x, label_ymin - label_size[1] - 10), 
                (start_x + label_size[0] + 10, label_ymin + base_line - 10), 
                box_color, 
                cv2.FILLED
            )
            
            # Write text overlay
            cv2.putText(
                img, 
                label, 
                (start_x + 5, label_ymin - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )

        # 4. Write generated file to output folder using the same file handle name
        output_path = os.path.join(OUTPUT_DIR, f"detected_{image_name}")
        cv2.imwrite(output_path, img)
        print(f"Successfully processed: {image_name} -> saved to data/outputs/detected_{image_name}")

if __name__ == "__main__":
    batch_process()