import cv2
import numpy as np

def straighten_document(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image from {image_path}.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        print("No lines detected in the image.")
        return

    # Find the average angle of the detected lines
    angles = []
    for line in lines:
        for rho, theta in line:
            angle = theta * 180 / np.pi
            angles.append(angle)
    avg_angle = np.mean(angles)

    rotation_angle= avg_angle - 99

    # Rotate the image to straighten the document
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Display the straightened document
    cv2.imshow('Straightened Document', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the path to your tilted document image
image_path = 'Citizenship7.png'

# Call the function to straighten the document
straighten_document(image_path)
