import cv2
import numpy as np 

cap = cv2.VideoCapture('largedotlit.mp4')

# goodFeaturesToTrack params
feature_params = dict(maxCorners=100, 
                      qualityLevel=0.3, 
                      minDistance=4, 
                      blockSize=7)

# lucas kanade params (untuned) 
lk_params = dict(winSize=(25, 25), 
                 maxLevel=3, 
                 minEigThreshold=0.05,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply edge detection
    edges = cv2.Canny(frame, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # magico
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None , **lk_params)

    if p1 is not None: #selecting good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        print("Optical flow failed.")
        frame = cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), 10, (255, 0, 0), -1)


    # Draw directly on the frame
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(80) & 0xff
    if k == 27:
        break

    # Update the prev frame/points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


cap.release()
cv2.destroyAllWindows()
