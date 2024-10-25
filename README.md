# task-2
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained MobileNet model
model = MobileNet(weights="imagenet")

# Load and preprocess the image
image_path = 'sample_image.jpg'
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (224, 224))  # Resize to model's input size
image_array = img_to_array(image_resized)
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
image_array = preprocess_input(image_array)  # Preprocess for MobileNet

# Predict the image using the MobileNet model
predictions = model.predict(image_array)
decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decode top-3 predictions

# Display predictions
print("Predicted objects in the image:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i+1}: {label} (confidence: {score * 100:.2f}%)")

# Show the image with OpenCV
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Output
Predicted objects in the image:
1: tabby (confidence: 85.43%)
2: tiger_cat (confidence: 12.32%)
3: Egyptian_cat (confidence: 1.08%)
