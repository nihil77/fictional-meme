import cv2

# Load an image from a file
image = cv2.imread('annotated_horse_stance.jpg')

# Define the new width and height
new_width = 400
new_height = 391

# Resize the image to the new dimensions
resized_image = cv2.resize(image, (new_width, new_height))

# Display the resized image in a window
cv2.imshow('Resized Image', resized_image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
