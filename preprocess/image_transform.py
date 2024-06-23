import os
import torch
import argparse  as args

from PIL import Image
from tqdm import tqdm
from torchvision import transforms

if __name__ == '__main__':
    args = args.ArgumentParser()
    args.add_argument('--input_dir', type=str, default='input', help='Directory containing input images')
    args.add_argument('--output_dir', type=str, default='data/portrait2photo', help='Directory to save the dataset')
    args.add_argument('--use_original', type=str, default='True', help='Use original images as well')
    args.add_argument('--transform_flip', type=str, default='False', help='Apply horizontal flip transformation')
    args.add_argument('--transform_crop', type=str, default='False', help='Apply random crop transformation')
    args.add_argument('--transform_jitter', type=str, default='False', help='Apply color jitter transformation')
    args.add_argument('--transform_rotate', type=str, default='False', help='Apply random rotation transformation')

    args = args.parse_args()

    # Define the transformations
    transform_flip = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor()
    ])

    transform_crop = transforms.Compose([
        transforms.RandomCrop((200, 200)),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])  

    transform_original = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    transform_jitter = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    transform_rotate = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    # Function to apply transformations and save images
    def apply_transformations_and_save(input_dir, output_dir, transformations, transform_name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for filename in tqdm(os.listdir(input_dir)):

            img_path = os.path.join(input_dir, filename)
            image = Image.open(img_path)
            
            transformed_image = transformations(image)
            transformed_image = transforms.ToPILImage()(transformed_image)
            
            save_path = os.path.join(output_dir, f"{transform_name}_{filename}")
            transformed_image.save(save_path)

    # input directories
    input_directory_A = args.input_dir+'/train_portrait_input'
    input_directory_B = args.input_dir+'/train_photo_input'

    #output directories
    output_directory_A = args.output_dir+'/train_A'
    output_directory_B = args.output_dir+'/train_B'

    # test dataset directories
    test_directory_A = args.output_dir+'/test_A'
    test_directory_B = args.output_dir+'/test_B'

    directories = [input_directory_A, input_directory_B, output_directory_A, output_directory_B, test_directory_A, test_directory_B]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


    # Apply transformations to the input images
    if args.use_original == 'True':
        print("Copying original images to the output directory")
        apply_transformations_and_save(input_directory_A, output_directory_A, transform_original, 'original')
        print("Copying original images to the output directory")
        apply_transformations_and_save(input_directory_B, output_directory_B, transform_original, 'original')
    
    if args.transform_flip == 'True':
        print("Applying horizontal flip transformation to dataset A")
        apply_transformations_and_save(input_directory_A, output_directory_A, transform_flip, 'flip')
        print("Applying horizontal flip transformation to dataset B")
        apply_transformations_and_save(input_directory_B, output_directory_B, transform_flip, 'flip')

    if args.transform_crop == 'True':
        print("Applying random crop transformation to dataset A")
        apply_transformations_and_save(input_directory_A, output_directory_A, transform_crop, 'crop')
        print("Applying random crop transformation to dataset B")
        apply_transformations_and_save(input_directory_B, output_directory_B, transform_crop, 'crop')

    if args.transform_jitter == 'True':
        print("Applying color jitter transformation to dataset A")
        apply_transformations_and_save(input_directory_A, output_directory_A, transform_jitter, 'jitter')
        print("Applying color jitter transformation to dataset B")
        apply_transformations_and_save(input_directory_B, output_directory_B, transform_jitter, 'jitter')

    if args.transform_rotate == 'True':
        print("Applying random rotation transformation to dataset A")
        apply_transformations_and_save(input_directory_A, output_directory_A, transform_rotate, 'rotate')
        print("Applying random rotation transformation to dataset B")
        apply_transformations_and_save(input_directory_B, output_directory_B, transform_rotate, 'rotate')

    # Copy the first 100 images to the test directory
    for filename in os.listdir(output_directory_A)[:100]:
        img_path = os.path.join(output_directory_A, filename)
        save_path = os.path.join(test_directory_A, filename)
        os.rename(img_path, save_path)

    for filename in os.listdir(output_directory_B)[:100]:
        img_path = os.path.join(output_directory_B, filename)
        save_path = os.path.join(test_directory_B, filename)
        os.rename(img_path, save_path)


    print("----------Data preparation completed----------")
