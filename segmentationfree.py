# 231116 Segmentation Free App.
# jpg와 json 파일만 넣고, 컬럼을 선택하면 Segmentation 된 모습을 볼 수 있음

import streamlit as st
import json
import pandas as pd
import cv2
import numpy as np
from PIL import Image

# Define the overlay_segmentation function
def overlay_segmentation(image_array, annotations_df, selected_class_ids):
    # Create a color map for each selected Class_ID
    color_map = {class_id: np.random.randint(0, 256, 3).tolist() for class_id in selected_class_ids}

    # Create a blank image with the same dimensions as the original image for overlay
    overlay_image = np.zeros(image_array.shape, dtype=np.uint8)

    # Iterate over each annotation and apply the segmentation on the overlay
    for _, row in annotations_df.iterrows():
        if row['Class_ID'] in selected_class_ids:
            # Get the segmentation points for the current annotation
            segmentation_points = np.array(row['Type_value']).reshape((-1, 2))
            # Get the color for the current class
            color = color_map[row['Class_ID']]
            # Draw the segmentation on the blank image
            cv2.fillPoly(overlay_image, [segmentation_points], color=color)

    # Overlay the segmentation on the original image
    alpha = 0.5  # Transparency factor.
    # The addWeighted function calculates the weighted sum of two arrays
    overlayed_image = cv2.addWeighted(image_array, 1 - alpha, overlay_image, alpha, 0)

    return overlayed_image

# Streamlit app title
st.title(':blue_book: Segmentation Free :sunglasses:')

# File uploader allows user to add any file
uploaded_files = st.file_uploader("Upload jpg/png and json files", accept_multiple_files=True, type=['jpg', 'png', 'json'])

# Initialize variables
image = None
json_data = None
selected_class_ids = []

# Display the file names and type
if uploaded_files:
    file_info = {file.name: file.type for file in uploaded_files}
    st.json(file_info)

    # Process the uploaded files
    for uploaded_file in uploaded_files:
        # Check if the file is an image
        if uploaded_file.type in ["image/jpeg", "image/png"]:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        # Check if the file is a JSON
        elif uploaded_file.type == "application/json":
            json_data = json.load(uploaded_file)
            df = pd.json_normalize(json_data['Learning_Data_Info.']['Annotations'])
            st.dataframe(df)
            # Allow the user to select which Class_ID to segment
            class_ids = df['Class_ID'].unique()
            selected_class_ids = st.multiselect("Select Class_IDs to segment", class_ids)

# Go detect button
if st.button("GO Segmentation!"):
    # Perform detection and segmentation
    if image and json_data and selected_class_ids:
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        # Perform segmentation
        segmented_image_array = overlay_segmentation(image_array, df, selected_class_ids)
        # Convert numpy array to PIL Image for display
        segmented_image = Image.fromarray(segmented_image_array)
        # Display the segmented image
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)
    else:
        st.write("Please upload both image and JSON files with matching names and select Class_IDs to segment.")

# Reset button (currently just a placeholder)
if st.button("Reset", key='reset'):
    # This will print to the server's console, not the web app.
    print("Reset the app state here")