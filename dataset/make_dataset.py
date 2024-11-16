import argparse
import os
import random
import numpy as np
import cv2
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Constants
CANVAS_SIZE = 100
NUM_DISTRACTORS_RANGE = (0, 20)
DIGIT_SIZE_RANGE = (0.33, 3.0)

def preprocess_mnist(mnist_dataset):
    digit_dict = {i: [] for i in range(10)}
    for img, label in mnist_dataset:
        digit_dict[label].append(img.numpy().squeeze() * 255)  # Convert to NumPy and scale
    return digit_dict

def get_rand_distractor_fragment(digit_dict, target_label):
    '''Gets a random distractor NOT a part of the target_label class'''
    while True:
        distractor_label = random.randint(0, 9)  # Random label
        if distractor_label != target_label:
            break

    distractor_digit = random.choice(digit_dict[distractor_label])

    h, w = distractor_digit.shape
    
    max_crop_width = int(w/3) #TODO: maybe change this? 
    max_crop_height = int(h/3) #TODO: maybe change this? 
     
    crop_w = random.randint(1, max_crop_width)
    crop_h = random.randint(1, max_crop_height)
    
    crop_offset_x = random.randint(0, w-crop_w)
    crop_offset_y = random.randint(0, h-crop_h)
    
    distractor_fragment = distractor_digit[crop_offset_y:crop_offset_y+crop_h, crop_offset_x:crop_offset_x+crop_h]
    
    return distractor_fragment

def add_distractors_to_canvas(num_distractors, canvas, digit_dict, label):
    for _ in range(num_distractors):
        distractor_fragment = get_rand_distractor_fragment(digit_dict, label)

        # Place the distractor
        h, w = distractor_fragment.shape
        x_offset = random.randint(0, CANVAS_SIZE - w)
        y_offset = random.randint(0, CANVAS_SIZE - h)
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = distractor_fragment
        
    return canvas

def add_target_digit_to_canvas(canvas, digit_dict, label, scaled):
    target_digit = random.choice(digit_dict[label])

    if scaled:
        scale_factor = random.uniform(*DIGIT_SIZE_RANGE)
        target_digit = cv2.resize(
            target_digit,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR,
        )

    # Place the target digit randomly on the canvas
    h, w = target_digit.shape
    x_offset = random.randint(0, CANVAS_SIZE - w)
    y_offset = random.randint(0, CANVAS_SIZE - h)
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = target_digit
    
    return canvas

def create_cluttered_image(label, digit_dict, scaled=False):
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)  # Blank canvas

    # Add distractors
    num_distractors = random.randint(*NUM_DISTRACTORS_RANGE)
    canvas = add_distractors_to_canvas(num_distractors, canvas, digit_dict, label)
    
    # Add target digit
    canvas = add_target_digit_to_canvas(canvas, digit_dict, label, scaled)

    return canvas

def main(args):
    # Load MNIST dataset
    mnist_dataset = MNIST(root="./raw_mnist", train=True, download=True, transform=ToTensor())
    digit_dict = preprocess_mnist(mnist_dataset)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(10):
        os.makedirs(os.path.join(args.output_dir, str(i)), exist_ok=True)

    # Generate images
    for label in range(10):
        for i in range(args.num_images_per_class):
            canvas = create_cluttered_image(label, digit_dict, scaled=args.scaled)
            filename = os.path.join(args.output_dir, str(label), f"{i}.png")
            cv2.imwrite(filename, canvas)
            print(f"Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="cluttered_mnist", help="Output directory for cluttered MNIST images")
    parser.add_argument("--scaled", action="store_true", help="Whether to create the scaled version of the dataset")
    parser.add_argument("--num_images_per_class", type=int, default=100, help="Number of images to create per class")
    args = parser.parse_args()
    main(args)
