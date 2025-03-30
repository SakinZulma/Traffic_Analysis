from ultralytics import YOLO
import cv2
import numpy as np
import sys
from collections import deque
import string
import sys
from fast_plate_ocr import ONNXPlateRecognizer
import pandas as pd
import re
from itertools import product
import pandas as pd




video_test = './videos/car.mp4'
CAMERA_ANGLE = "Right"
OUTPUT_VIDEO = True


class VehicleTracker:
    def __init__(self, frame_height, entry_zone_height=350):
        self.tracks = {}  # track_id: {positions, counted, last_seen, lp_info}
        self.current_vehicles = []
        self.lp_history = {}
        self.frame_height = frame_height
        self.entry_line = self.frame_height/2
        self.exit_line = entry_zone_height
        
    def update_tracks(self, vehicles, frame_nmr, frame):
        current_ids = set()
        
        # Get current vehicle bounding boxes
        vehicle_bboxes = [v['bbox'] for v in vehicles]
               
        
        for vehicle in vehicles:
            track_id = vehicle['id']
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'positions': deque(maxlen=50),
                    'counted': False,
                    'last_seen': frame_nmr,
                    'lp_info': {'text': None}}
                
            self.tracks[track_id]['positions'].append(vehicle['center'])
            self.tracks[track_id]['last_seen'] = frame_nmr         
            
            current_ids.add(track_id)
        
        # Remove old tracks
        expired = [tid for tid, data in self.tracks.items() if frame_nmr - data['last_seen'] > 100]
        for tid in expired:
            del self.tracks[tid]

    

    def get_direction(self, track_id):
        positions = self.tracks[track_id]['positions']
        entry = -1
        threshold = 10  
        motion_threshold = 10
        entry_count = 0
        leaving_count = 0
        Condition_One = False
        Condition_Two = False
    
        if len(positions) < 20: 
            return "Unknown"
    
        if not isinstance(positions, list):  
            positions = list(positions)
    
        if all(isinstance(pos, tuple) and len(pos) == 2 for pos in positions[-20:]):
            if all(pos[1] == positions[-1][1] for pos in positions[-20:]):
                Condition_One = True
                
        recent_positions = np.array(positions[-20:])
        y_var = np.var(recent_positions[:, 1])
        x_var = np.var(recent_positions[:, 0])

        if x_var < threshold and y_var < threshold:
            Condition_Two = True
        else:
            Condition_Two = False

        if Condition_One or Condition_Two:
            return "Stationary"
        
    
        for i in range(len(positions) - 1):  
            past_position = positions[i][1]
            current_position = positions[i + 1][1]
    
            # Select a reference position to compare with
            reference_position = positions[max(i - 20, 0)][1]
            
            # Detect movement across the line
            if past_position >= self.entry_line: 
                if current_position > reference_position:
                    entry = 1  # Moving forward (crossing forward)
                elif current_position < reference_position:
                    entry = 0  # Moving backward (crossing backward)
                    
            if CAMERA_ANGLE == "Right":
                # Track movement direction using x-coordinates
                if positions[i + 1][0] < positions[i][0]:  
                    entry_count += 1
                elif positions[i + 1][0] > positions[i][0]:
                    leaving_count += 1
            else:
                # Track movement direction using x-coordinates
                if positions[i + 1][0] > positions[i][0]:  
                    entry_count += 1
                elif positions[i + 1][0] < positions[i][0]:
                    leaving_count += 1
    
    
        # Determine movement based on x-axis counts
        if entry_count > leaving_count:
            motion_decision = "Entering"
        elif leaving_count > entry_count:
            motion_decision = "Leaving"
        else:
            motion_decision = "Unknown"
    
        # Determine crossing based on y-axis
        if entry == 1:
            line_decision = "Entering"
        elif entry == 0:
            line_decision = "Leaving"
        else:
            line_decision = "Unknown"


        if motion_decision == line_decision:
            return motion_decision  

    
        if line_decision == "Unknown":
            return motion_decision 
    
        # Give equal weight to motion and line
        if motion_decision != "Unknown" and line_decision != "Unknown":
            return motion_decision if ((abs(entry_count - leaving_count)) >= motion_threshold) else line_decision
    
        return "Unknown"


def is_inside(lp_box, vehicle_box):
    lx1, ly1, lx2, ly2 = lp_box
    vx1, vy1, vx2, vy2 = vehicle_box
    return (vx1 < lx1 < lx2 < vx2) and (vy1 < ly1 < ly2 < vy2)


def write_csv(results, output_path):
    """
    Write the tracking results to a CSV file.
    The CSV includes frame number, vehicle type, vehicle score, vehicle ID, timestamp,
    vehicle bounding box, direction, license plate bounding box, license plate bbox score,
    and license plate text.
    """
    header = ['frame_nmr', 'vehicle_type', 'vehicle_score', 'car_id', 'timestamp','car_bbox', 'direction' , 'license_number']
    
    with open(output_path, 'w') as f:
        f.write(','.join(header) + "\n")
        
        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                row = results[frame_nmr][car_id]
                
                # Handle car bounding box
                car_bbox = row.get('car_bbox', None)
                if isinstance(car_bbox, list):
                    car_bbox = " ".join(map(str, car_bbox))  # Convert list to string
                
                # Write row to CSV
                f.write("{},{},{},{},{},{},{},{}\n".format(
                    frame_nmr,
                    row.get('vehicle_type', 'None'),  # Handle missing vehicle_type
                    row.get('vehicle_score', 'None'),  # Handle missing vehicle_score
                    car_id,
                    row.get('timestamp', 'None'),  # Handle missing timestamp
                    car_bbox if car_bbox else 'None',  # Handle missing car_bbox
                    row.get('direction', 'None'),  # Handle missing direction
                    row['license_plate'].get('text', 'None') if row.get('license_plate') else 'None'
                ))



def read_license_plates(license_plate_crop):
    text_list = m.run(license_plate_crop)
    
    # Define constants
    PATTERNS = [
        re.compile(r'^[A-Z]{3}\d{3}[A-Z]{2}$'),  # LLLNNNLL
        re.compile(r'^[A-Z]{2}\d{3}[A-Z]{3}$')   # LLNNNLLL
    ]
    
    AMBIGUOUS_CHARS = {
        '0': {'O'}, 'O': {'0'},
        '1': {'I', 'L'}, 'I': {'1'}, 'L': {'1'},
        '2': {'Z'}, 'Z': {'2'},
        '3': {'B'}, 'B': {'3', '8'}, '8': {'B'},
        '5': {'S'}, 'S': {'5'},'A': {'4'}, '4': {'A'}, 
        'G':{'6'}, '6': {'G'} 
    }

    def get_candidates(text):
        """Generate all possible valid variations"""
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(cleaned) != 8:
            return []

        # Generate possible substitutions for each character
        char_options = []
        for c in cleaned:
            substitutions = AMBIGUOUS_CHARS.get(c, set()) | {c}
            char_options.append(sorted(substitutions, key=lambda x: x != c))

        # Generate permutations with original text first
        return [''.join(comb) for comb in product(*char_options)]

    # Process all OCR results
    for text in text_list:
        # Try candidates in order of likelihood (original first)
        for candidate in get_candidates(text):
            if any(p.match(candidate) for p in PATTERNS):
                if candidate != text:
                    print(f"Corrected {text} → {candidate}")
                return candidate

    return None
    
def license_plate_processing(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Deskewing
    coords = cv2.findNonZero(thresh)  
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed


# --- Configuration & Initialization ---
cap = cv2.VideoCapture(video_test)
if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# Get frame dimensions and fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

tracker = VehicleTracker(frame_height, frame_width)       

direction = ""


# Flags to control time-based processing

frame_nmr = - 1

# Load YOLO models (using .track for detection and tracking)
vehicle_detector = YOLO('./models/class.pt')

license_plate_detector = YOLO('./models/license.pt')
m = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model')


results = {}  
frame_count = 0




# --- Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame_nmr += 1
    current_time = frame_count / fps
    timestamp = cv2.getTickCount( ) / cv2.getTickFrequency()

    results[frame_nmr] = {}
    current_vehicles = []
    
    # Vehicle detection and tracking
    vehicle_results = vehicle_detector.track(frame, persist=True, conf=0.5, iou=0.5)[0]
    if vehicle_results.boxes.id is not None:
        boxes = vehicle_results.boxes.xyxy.cpu().numpy()
        vehicle_ids = vehicle_results.boxes.id.cpu().numpy().astype(int)
        classes = vehicle_results.boxes.cls.cpu().numpy().astype(int)
        scores = vehicle_results.boxes.conf.cpu().numpy()

        for box, vid, cls, score in zip(boxes, vehicle_ids, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            current_vehicles.append({
                'id': vid,
                'bbox': [x1, y1, x2, y2],
                'center': ((x1+x2)/2, (y1+y2)/2),
                'class': "Car" if cls == 0 else "Korope" if cls == 1 else "Bus",
                'score': float(score)})

    tracker.update_tracks(current_vehicles, frame_nmr, frame)


    # --- License Plate Detection on Cropped Vehicle Region ---
    license_plate_text = ""
    license_plate_score = 0
    lp_bbox = []

    # Get license plate
    lp_results = license_plate_detector(frame, conf=0.5)[0]
    if lp_results.boxes is not None:
        for lp in lp_results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, lp)
            best_match = None
            
            for vehicle in current_vehicles:
                vbox = vehicle['bbox']
                if is_inside(lp, vbox):
                    best_match = vehicle['id']

            if best_match:
                print("Best Match Found")
                lp_crop = frame[y1:y2, x1:x2]
                
                lp_processed = license_plate_processing(lp_crop)

                lp_text = read_license_plates(lp_processed)
                
                if lp_text:
                    tracker.tracks[best_match]['lp_info'] = {
                        'text': lp_text}

    for vehicle in current_vehicles:
        vid = vehicle['id']
        x1, y1, x2, y2 = vehicle['bbox']
        
        # Get direction and color coding
        direction = tracker.get_direction(vid)
        
        if OUTPUT_VIDEO == True:
            color = ((0, 255, 0) if direction == "Entering" else (0, 0, 255) if direction == "Leaving" else (255, 255, 0) if direction == "Stationary" else (200, 200, 200))
        
            if direction != "Unkonwn":
                # Draw vehicle box and info
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{vid}: {direction}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
            # Draw movement trail
            positions = tracker.tracks[vid]['positions']
            for i in range(1, len(positions)):
                pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                pt2 = (int(positions[i][0]), int(positions[i][1]))
                cv2.line(frame, pt2, pt1, (255, 0, 0), 2)
            
    # Update results and counters
    for vid, data in tracker.tracks.items():
        results[frame_nmr][vid] = {'vehicle_type': next((v['class'] for v in current_vehicles if v['id'] == vid), None),'vehicle_score': next((v['score'] for v in current_vehicles if v['id'] == vid), None),'car_bbox': next((v['bbox'] for v in current_vehicles if v['id'] == vid), None),'timestamp': timestamp,'direction': tracker.get_direction(vid),'license_plate': data['lp_info']}

    if OUTPUT_VIDEO == True:
        out.write(frame)



write_csv(results, './sample.csv')

out.release()
cap.release()
cv2.destroyAllWindows()



# Load the CSV file
file_path = "./sample.csv"
df = pd.read_csv(file_path)

# Drop rows with missing values in the first 7 columns
df_cleaned = df.dropna(subset=df.columns[:7])
# Process data for each unique car_id without limiting to the first 50 rows
final_data = []

for car_id, group in df_cleaned.groupby("car_id"):
    # Determine vehicle_type by selecting the row with the highest vehicle_score
    best_vehicle_row = group.loc[group["vehicle_score"].idxmax()]
    vehicle_type = best_vehicle_row["vehicle_type"]

    # Extract license_plate_number: first non-"None" value or "None"
    license_number = next((ln for ln in group["license_number"] if ln != "None"), "None")

    # Determine whether "Entering" or "Leaving" is more frequent in all rows
    direction_counts = group["direction"].value_counts()
    most_frequent_direction = direction_counts.idxmax() if not direction_counts.empty else None

    # Extract entering_time and leaving_time based on correct capitalization
    entering_time = group.loc[group["direction"] == "Entering", "timestamp"].min() if most_frequent_direction == "Entering" else None
    leaving_time = group.loc[group["direction"] == "Leaving", "timestamp"].min() if most_frequent_direction == "Leaving" else None

    # Get the earliest timestamp as time_detected
    time_detected = group["timestamp"].min()

    # Append extracted information to the final data list
    final_data.append([car_id, vehicle_type, entering_time, leaving_time, time_detected, license_number])

# Create DataFrame and save to CSV
final_df = pd.DataFrame(final_data, columns=["car_id", "vehicle_type", "entering_time", "leaving_time", "time_detected", "license_plate_number"])
final_csv_path = "final.csv"
final_df.to_csv(final_csv_path, index=False)

# Return the final file path
final_csv_path
