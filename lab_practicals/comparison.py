import cv2
import torch
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T

# Load YOLO model
yolo_model = YOLO('yolov8n.pt')

# Load Faster R-CNN model
rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
rcnn_model.eval()

def preprocess_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model
    yolo_results = yolo_model(frame)

    # Run Faster R-CNN
    image_tensor = preprocess_image(frame)
    with torch.no_grad():
        rcnn_outputs = rcnn_model(image_tensor)

    # Draw YOLO results (Green boxes)
    for result in yolo_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"YOLO {class_id} | {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Faster R-CNN results (Blue boxes)
    for i, (box, score, label) in enumerate(zip(rcnn_outputs[0]['boxes'], 
                                                rcnn_outputs[0]['scores'], 
                                                rcnn_outputs[0]['labels'])):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"RCNN {label.item()} | {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("YOLO vs Faster R-CNN", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
