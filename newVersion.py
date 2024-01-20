import tkinter as tk
from PIL import Image, ImageTk
import cv2

# Function to perform prediction
def predict_sign_language():
    global label_result
    img = cv2.imread('path_to_your_test_image', cv2.IMREAD_GRAYSCALE)  # Replace 'path_to_your_test_image' with test image path
    img = cv2.resize(img, (width, height))  # Resize image to match training data
    img = img.flatten().reshape(1, -1)  # Flatten and reshape image for prediction
    prediction = rf_classifier.predict(img)
    label_result.config(text=f"Predicted sign: {prediction[0]}")

# Create GUI window
root = tk.Tk()
root.title("Sign Language Detection")

# Replace 'path_to_your_test_image' with a path to your test image
test_image = Image.open('path_to_your_test_image')  # Load test image
width, height = test_image.size
test_image = test_image.resize((300, 300))  # Resize for display
test_image = ImageTk.PhotoImage(test_image)

# Display test image in GUI
panel = tk.Label(root, image=test_image)
panel.pack(side="top", padx=10, pady=10)

# Button to perform prediction
button_predict = tk.Button(root, text="Predict", command=predict_sign_language)
button_predict.pack(pady=10)

# Label to display prediction result
label_result = tk.Label(root, text="")
label_result.pack(pady=10)

root.mainloop()
