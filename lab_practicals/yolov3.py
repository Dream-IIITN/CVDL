import cv2
import numpy as np
from ultralytics import YOLO

# Load the pre-trained YOLOv3 model
model = YOLO("yolov3.pt")  # YOLOv3 model from Ultralytics

# Load the image
image_path = "yolov3x.jpg"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Run YOLOv3 inference on the image
results = model(image)

# Class names for YOLO (COCO dataset)
CLASS_NAMES = model.names  # Dictionary mapping class ID to label

# Process detection results
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Iterate through detections and draw bounding boxes
    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)  # Convert bounding box coordinates to integers
        class_name = CLASS_NAMES.get(class_id, "Unknown")  # Get class label

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with confidence score
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save the annotated image
output_path = "output.jpg"
cv2.imwrite(output_path, image)
print(f"Annotated image saved at {output_path}")

# Display the image
cv2.imshow("YOLOv3 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
