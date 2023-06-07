import cv2

def crop_document(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to create a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours based on their area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Extract the largest contour (document outline)
    document_contour = contours[0]
    
    # Approximate the contour to a polygon
    epsilon = 0.03 * cv2.arcLength(document_contour, True)
    approx_polygon = cv2.approxPolyDP(document_contour, epsilon, True)
    
    # Ensure the polygon has 4 vertices
    if len(approx_polygon) != 4:
        raise ValueError("Unable to detect the document edges.")
    
    # Reorder the vertices of the polygon
    rect = cv2.boundingRect(approx_polygon)
    x, y, w, h = rect
    offset= 240
    cropped = image[y-offset:y+h, x:x+w]
    
    # Display the cropped image
    cv2.imshow('Cropped Document', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the path to your citizenship document image
image_path = 'D:/op/Citizenship1.jpg'

# Call the function to crop the document
crop_document(image_path)
