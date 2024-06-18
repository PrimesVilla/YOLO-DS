import hashlib
import imutils
from ultralytics import YOLO
import cv2
import math
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectDetection:
    def __init__(self, capture, confidence_threshold=0.80):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names
        self.confidence_threshold = confidence_threshold

    def load_model(self):
        model = YOLO("YOLO-DS/runs/detect/train3/weights/best.pt")
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img, stream=True)
        return results

    def plot_boxes(self, results, img):
        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, current_class = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                current_class = int(current_class)
                if conf > self.confidence_threshold:
                    detections.append(([x1, y1, x2, y2], conf, current_class))
        return detections, img

    def id_to_color(self, id_str):
        hash_object = hashlib.md5(id_str.encode())
        hash_hex = hash_object.hexdigest()
        r = int(hash_hex[:2], 16) % 256
        g = int(hash_hex[2:4], 16) % 256
        b = int(hash_hex[4:6], 16) % 256
        return (r, g, b)

    def track_detect(self, detections, img, tracker):
        tracks = tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            color = self.id_to_color(str(track_id))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

        return img

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        assert cap.isOpened()

        tracker = DeepSort()

        while True:
            ret, img = cap.read()

            if not ret:
                break

            img = imutils.resize(img, width=640)  # Resize frame for faster processing
            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = self.track_detect(detections, frames, tracker)

            cv2.imshow('Image', detect_frame)
            if cv2.waitKey(5) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage:
detector = ObjectDetection(capture=0)
detector()
