import os
import pickle
import cv2
import mediapipe as mp
import shutil

mp_hands = mp.solutions.hands

# Initialize hands module with a lower detection confidence threshold
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1)

data = []
labels = []
processed_images_count = 0  # Initialize counter

# Create a dictionary to map directories to lists of replacement candidates
replacement_mapping = {
    'path_to_directory_1': 'path_to_directory_with_detected_hands_1',
    'path_to_directory_2': 'path_to_directory_with_detected_hands_2',
    # Add more mappings as needed
}

# Iterate through directories in the data folder
for directory in os.listdir('./data'):
    for img_path in os.listdir(os.path.join('./data', directory)):
        data_aux = []
        img_filepath = os.path.join('./data', directory, img_path)

        try:
            # Read and process the original image
            original_img = cv2.imread(img_filepath)

            if original_img is None:
                raise Exception(f"Error: Unable to read image {img_filepath}")

            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Process hand landmarks using MediaPipe
            original_results = hands.process(original_img_rgb)

            # Check if at least one hand is detected with sufficient confidence in the original image
            if original_results.multi_hand_landmarks and original_results.multi_hand_landmarks[0].landmark:
                for hand_landmarks in original_results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        data_aux.extend([landmark.x, landmark.y])

                # Append data and label
                data.append(data_aux)
                labels.append(directory)

                # Increment the processed images count
                processed_images_count += 1
            else:
                # If no hands are detected, try to replace with an image with detected hands
                if directory in replacement_mapping and replacement_mapping[directory]:
                    replacement_directory = replacement_mapping[directory]

                    # Find an image with detected hands in the replacement directory
                    for candidate_img_name in os.listdir(os.path.join('./data', replacement_directory)):
                        candidate_img_path = os.path.join('./data', replacement_directory, candidate_img_name)
                        candidate_img = cv2.imread(candidate_img_path)
                        candidate_img_rgb = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB)

                        # Process hand landmarks using MediaPipe for the candidate image
                        candidate_results = hands.process(candidate_img_rgb)

                        # Check if at least one hand is detected with sufficient confidence in the candidate image
                        if candidate_results.multi_hand_landmarks and candidate_results.multi_hand_landmarks[0].landmark:
                            # Replace the original image with the candidate image
                            shutil.copy(candidate_img_path, img_filepath)
                            print(f"Replaced image {img_path} in {directory} with a candidate image with detected hands")
                            break  # Break out of the loop since a replacement has been found

        except Exception as e:
            print(f"Error processing {img_filepath}: {e}")

# Display the total number of processed images
print(f"Total number of processed images: {processed_images_count}")

# Save data to a pickle file
output_filepath = 'data.pickle'
with open(output_filepath, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f, protocol=pickle.HIGHEST_PROTOCOL)

# Release resources
hands.close()
