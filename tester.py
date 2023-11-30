import cv2
import torch
from torchvision import transforms
from PIL import Image
import torchvision
import numpy as np
import torch.nn as nn
from CNN_CLASS import Net


# Define the transformations - these should be the same as what you used during training
transform = transforms.Compose(
    [transforms.Resize((200, 200)), # Resize the images to 224x224 pixels
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

PATH = r'sign_lang.pth'
# Initialize the model (same structure as in the training script)
net = Net() # Same model architecture
net.load_state_dict(torch.load(PATH))  # Set your correct path here

# Set the model to evaluation mode
net.eval()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 is the index of the webcam. Adjust if necessary.

# Define the list of class names
classes =['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Loop for processing the video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to PIL for easier transformations
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Convert the image to grayscale


    # Apply transformations
    image = transform(frame)

    # Add an extra dimension because the model expects batches
    image = image.unsqueeze(0)

    # Make a prediction
    output = net(image)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    print(f'Predicted class: {predicted_class.item()}')
    print(classes[predicted_class.item()])

    # Display the resulting frame
    cv2.imshow('frame', cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()