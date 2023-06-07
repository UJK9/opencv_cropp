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
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    document_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the bounding rectangle of the document contour
    x, y, w, h = cv2.boundingRect(document_contour)

    
    
    # Adjust the bounding rectangle with the specified offset and onset
    x -= offset
    y -= onset
    w += 2 * offset
    h = h+ 2* onset

    print(x, y, w, h)
    
    # Ensure the cropped region has a valid size
    if w > 0 and h > 0:
        # Crop the document from the original image with the adjusted bounding rectangle
        cropped = image[y : y + h, x : x + w]
        
        # Display the cropped image
        cv2.imshow('Cropped Document', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid cropping region. Please adjust the offset and onset values.")

# Specify the path to your document image
image_path = 'Citizenship5.jpg'

# Specify the desired offset and onset values for cropping
offset = 660
onset = 580

# Call the function to crop the document with the specified parameters
crop_document(image_path, offset, onset)
