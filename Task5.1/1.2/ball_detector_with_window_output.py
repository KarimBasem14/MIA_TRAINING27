import cv2
import numpy as np
import os

def detect_colored_balls(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}")
        return

    # Keep a bit more of the original size so tiny balls don't vanish
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    
    # Convert to HSV for easy color isolation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Isolate Blue
    mask_blue = cv2.inRange(hsv, np.array([100, 100, 50]), np.array([140, 255, 255]))
    
    # Isolate Red
    mask_red1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    def find_and_draw(mask, color_name, draw_color):
        
        clean_mask = np.zeros_like(mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 50:
                
                # --- NEW ASPECT RATIO CHECK TO STOP FALSE POSITIVES ---
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / float(h)
                
                if 0.5 < aspect_ratio < 2.0:
                    hull = cv2.convexHull(cnt)

                    cv2.drawContours(clean_mask, [hull], -1, 255, thickness=cv2.FILLED)
        
        blurred_mask = cv2.GaussianBlur(clean_mask, (7, 7), 0)
        
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
                
                print(f"{color_name} at X:{x} Y:{y} | Width: {width}, Height: {height}")
                
                cv2.circle(img, (x, y), radius, draw_color, 2)
                cv2.circle(img, (x, y), 2, (255, 255, 255), 3) # Center dot
                cv2.putText(img, color_name, (x - radius, y - radius - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

    # Process both colors
    find_and_draw(mask_blue, "Blue Ball", (255, 0, 0))
    find_and_draw(mask_red, "Red Ball", (0, 0, 255))

    cv2.imshow("Detected Balls", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
test_image = os.path.join(SCRIPT_DIR, "balls", f"ball_16.jpg") 
detect_colored_balls(test_image)