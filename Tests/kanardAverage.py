import cv2
import numpy as np 

cap = cv2.VideoCapture('shapeslit.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
refreshrate = 3 #smoothing

dx_stack = []
dy_stack = []

currentposx, currentposy = 0,0

# goodFeaturesToTrack params
feature_params = dict(maxCorners=100, 
                      qualityLevel=0.6,  
                      minDistance=0.0001, 
                      blockSize=7)

# lucas kanade params 
lk_params = dict(winSize=(25, 25), 
                 maxLevel = 10,
                 minEigThreshold=0.0001, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# to write to file
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output.mp4', fourcc, fps, (old_frame.shape[1], old_frame.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply edge detection
    edges = cv2.Canny(frame, 50, 150) #frame lower upper thres
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None , **lk_params)
    except:
        print("Nothing on frame")
    if p1 is not None: #selecting good points
        try:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        except:
            print("idk wtf goin on here when doin wood")
    else:
        print("Optical flow calculation failed.")
        frame = cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), 10, (255, 0, 0), -1)


    # Draw the tracks directly on the frame
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        dy = d-b
        dx = c-a 
        if len(dx_stack) > int(fps*refreshrate):
            dx_stack.pop(0)
            dy_stack.pop(0)
        dx_stack.append(dx)
        dy_stack.append(dy)
        frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    if dx_stack:
        dx_avg = np.mean(dx_stack)
    else:
        dx_avg = 0  # or some other default value

    if dy_stack:
        dy_avg = np.mean(dy_stack)
    else:
        dy_avg = 0  # or some other default value

    magnitude_avg = np.hypot(dx_avg, dy_avg)
    direction_avg = np.arctan2(dy_avg, dx_avg) * (180 / np.pi)  # Convert to degrees

    if not np.isnan(dx_avg) and not np.isnan(dy_avg):
        # Draw average flow
        cv2.arrowedLine(frame, 
                (int(frame.shape[1]/2), int(frame.shape[0]/2)), 
                (int(frame.shape[1]/2 + dx_avg * 3), int(frame.shape[0]/2 + dy_avg *3)), 
                color = (255, 0, 0),
                thickness= 5)
        cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), 10, (255, 0, 0), -1)
        cv2.putText(frame, "Avg Magnitude: " + str(magnitude_avg), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.putText(frame, "Avg Direction: " + str(direction_avg), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    
    # write to file 
    out.write(frame)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(80) & 0xff
    if k == 27:
        break

    # Update the prev frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

cap.release()
out.release

cv2.destroyAllWindows()
