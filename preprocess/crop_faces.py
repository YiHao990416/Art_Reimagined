import cv2
import dlib
import os
import argparse as args
import os

from tqdm import tqdm

# rename the png file number
def rename_image(filepath):
    filelist = os.listdir(filepath)
    for i, filename in enumerate(filelist):
        os.rename(os.path.join(filepath, filename), os.path.join(filepath, f"{i}.png"))

# Function to detect faces and crop/resize them from an image
def crop_faces(image_path, output_dir, target_size=(256, 256), expansion_factor=2.6, img_name='face'):

    # Load the pre-trained face detector model from dlib (HOG-based)
    detector = dlib.get_frontal_face_detector()

    # Load image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale (dlib works with grayscale images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector(gray)
    
    # Iterate through detected faces
    for i, face in enumerate(faces):
        # Get the coordinates of the face region
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        
        # Calculate the dimensions of the bounding box
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Calculate the expansion area around the face
        expand_x = int(face_width * (expansion_factor - 1) / 2)
        expand_y = int(face_height * (expansion_factor - 1) / 2)
        
        # Adjust the bounding box to include more area around the face
        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(img.shape[1], x2 + expand_x)
        y2 = min(img.shape[0], y2 + expand_y)
        
        # Crop the expanded face region from the image
        cropped_face = img[y1:y2, x1:x2]
        
        # Resize the cropped face to the target size (256x256 pixels)
        resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
        
        # Save the resized face as a PNG file to the output directory
        output_path = os.path.join(output_dir, f"face_{img_name}_{i+1}.png")
        cv2.imwrite(output_path, resized_face)
        
if __name__ == "__main__":
    args = args.ArgumentParser()
    args.add_argument('--input_dir', type=str, required= True, help='Directory containing raw images')
    args.add_argument('--output_dir', type=str, required = True, help='Directory to save cropped images')
    args.add_argument('--image_size', type=int, default=256, help='Size of the output cropped face images')
    args.add_argument('--expansion_factor', type=float, default=2.6, help='Expansion factor for the bounding box')

    args = args.parse_args()

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    target_size = (args.image_size, args.image_size)  # Fixed size of the output cropped face images

    raw_photo_list = os.listdir(args.input_dir)
    for j,photos in tqdm(enumerate(raw_photo_list), total=len(raw_photo_list), desc="Cropping Faces"):
        crop_faces((args.input_dir + "/" + photos), args.output_dir, target_size, args.expansion_factor, str(j))

    # Rename the cropped face images to have sequential filenames
    rename_image(args.output_dir)


