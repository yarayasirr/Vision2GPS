import cv2
import numpy as np

frame = cv2.imread("frame_for_calibration.jpg")

points = []

def click_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(frame, str(len(points)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Pick 4 road points", frame)
        print(f"Point {len(points)}: pixel ({x}, {y})")
        if len(points) == 4:
            print("\nCopy these into your homography script:")
            print("pixel_points = np.float32([")
            for p in points:
                print(f"    [{p[0]}, {p[1]}],")
            print("])")

cv2.imshow("Pick 4 road points", frame)
cv2.setMouseCallback("Pick 4 road points", click_point)
cv2.waitKey(0)
cv2.destroyAllWindows()