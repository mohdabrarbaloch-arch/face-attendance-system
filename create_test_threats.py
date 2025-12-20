import cv2
import os
import time

# Create screenshots folder if it doesn't exist
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully!")
print("Press 's' to save a test threat image")
print("Press 'q' to quit")

saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Display the frame
    cv2.imshow("Test Threat Capture - Press 's' to save, 'q' to quit", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        # Save the frame as a threat image
        timestamp = int(time.time())
        filename = f"screenshots/threat_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        saved_count += 1
        print(f"Saved: {filename} (Total: {saved_count})")
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal test threats saved: {saved_count}")
