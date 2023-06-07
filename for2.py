import cv2
import numpy as np

def crop_document(image_path, offset, onset):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny with adjusted parameters
    edges = cv2.Canny(blurred, 30, 100)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours based on their area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Find the document contour by iterating over all contours
    document_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.041 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            document_contour = approx
            break
    
    # Ensure the document contour is found
    if document_contour is None:
        raise ValueError("Unable to detect the document edges.")
    
    # Find the bounding box of the document contour
    x, y, w, h = cv2.boundingRect(document_contour)
    
    # Calculate the new cropping coordinates
    new_x = max(0, x - offset)
    new_y = max(0, y - onset)
    new_w = w + offset + min(x - offset, 0) + min(image.shape[1] - (x + w + offset), 0) + 750
    new_h = h + onset
    
    # Crop the document from the original image with adjusted cropping coordinates
    cropped = image[new_y : new_y + new_h, new_x : new_x + new_w]
    
    # Display the cropped image
    cv2.imshow('Cropped Document', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the path to your citizenship document image
image_path = 'D:/op/Citizenship2.jpg'

# Specify the desired offset and onset values for cropping
offset = 900
onset = 1200

# Call the function to crop the document with the specified parameters
crop_document(image_path, offset, onset)
