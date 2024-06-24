# Art_Reimagined: Bringing Masterpieces to Life
This is a simple Cycle-GAN project trained with custom dataset collected from the internet. The goal of Art_Reimagined is to transform traditional Chinese portrait into real-life images. 

## Introduction
This is a simple Cycle-GAN project which trained with custom dataset collected from the Internet. The idea of this project is to transform traditional chinese portrait painting into real-life photos by utilizing the power of Cycle-GAN.Traditional Chinese paintings are revered for their unique artistic styles and rich cultural heritage. However, translating these traditional artworks into modern presents a significant challenge. In this project the datase are collected specifically from ancient Chinese TV-show to allow the model to capture details of the costumes, hairstyle and feature of faces. 

## Dataset 
The dataset contain Train_A domain and Train_B where the domain A is the Traditional Chinese Portrait Painting collected from Pinterest and the domain B is the photos collected from Chinese TV-shows. We filtered the collected images, croppped out the faces from the images and convert all the images to 256x256 pixels images. The dataset can be found in the input folder.
```
input
    ├── train_photo_input.zip
    └── train_portrait_input.zip


