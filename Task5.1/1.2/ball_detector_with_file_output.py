import cv2
import numpy as np
import os

# We pass the output_folder so the function knows where to save the .txt files
def detect_colored_balls(image_path, output_folder):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}")
        return

    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    
    img_h, img_w = img.shape[:2]
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Isolate Blue
    mask_blue = cv2.inRange(hsv, np.array([100, 100, 50]), np.array([140, 255, 255]))
    
    # Isolate Red
    mask_red1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    detected_labels = []

    def find_and_format(mask, class_id):
        
        clean_mask = np.zeros_like(mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / float(h)
                
                if 0.5 < aspect_ratio < 2.0:
                    
                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(clean_mask, [hull], -1, 255, thickness=cv2.FILLED)
        
        # Now we only need a TINY blur just to soften the edges. 
        blurred_mask = cv2.GaussianBlur(clean_mask, (7, 7), 0)
        
        # Run HoughCircles on the beautifully clean mask
        circles = cv2.HoughCircles(
            blurred_mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=50, 
            param1=50, 
            param2=20,    
            minRadius=10, 
            maxRadius=200
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x, y, radius = i[0], i[1], i[2]
                
                width = radius * 2
                height = radius * 2
                
                # Normalize the coordinates (divide by image width/height)
                x_norm = x / img_w
                y_norm = y / img_h
                w_norm = width / img_w
                h_norm = height / img_h
                
                detected_labels.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")


    find_and_format(mask_blue, 0)
    find_and_format(mask_red, 1)

    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    txt_path = os.path.join(output_folder, f"{base_name}.txt")
    
    with open(txt_path, 'w') as f:
        for label in detected_labels:
            f.write(label + "\n")
            
    print(f"Saved results for {filename} to {txt_path}")



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(SCRIPT_DIR, "balls")
output_folder = os.path.join(SCRIPT_DIR, "labels") 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        full_image_path = os.path.join(image_folder, filename)
        detect_colored_balls(full_image_path, output_folder)

print("All images processed successfully!")