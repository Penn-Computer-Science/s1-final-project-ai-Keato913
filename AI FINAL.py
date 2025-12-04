import cv2

image = cv2.imread("C:\\Users\\kwatkins799\\Documents\\GitHub\\s1-final-project-ai-Keato913\\shape.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Takes our gray image then takes the 220 value and for each pixel in our image if the pixel's gray value is equal to or below this number the function turns the pixel black, each other pixel is white.
_, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

#Contours is a list of contours and hierarchy stores the relationship that each of the contours have with eachother
contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Iterating through each contour
for i, contour in enumerate(contours):
    if i == 0:
        continue

#Approximates the shape, Makes it to where the A.I. recognises a shape even if the shape has imperfections
    epsilon = 0.01*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    cv2.drawContours(image, contour, 0, (0, 0, 0), 4)