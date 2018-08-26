## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg 

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

These are the functions that I used to identify navigable terrain and rocks.
For navigable terrain I used the threshold function as provided. For rock selection, I defined a new function that allows selecting a range for each RGB channel.

```python
def color_thresh(img, rgb_thresh=(180, 160, 150)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


# Pick colors which RGB values are all between their given ranges
def color_range(img, rgb_range=((150, 170), (150, 170), (150, 170))):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    
    in_range = (img[:,:,0] >= rgb_range[0][0]) & (img[:,:,0] <= rgb_range[0][1])\
               & (img[:,:,1] >= rgb_range[1][0]) & (img[:,:,1] <= rgb_range[1][1])\
               & (img[:,:,2] >= rgb_range[2][0]) & (img[:,:,2] <= rgb_range[2][1])
    # Index the array of zeros with the boolean array and set to 1
    color_select[in_range] = 1
    # Return the binary image
    return color_select


threshed = color_thresh(warped)
rock_image = mpimg.imread(example_rock)
ranged = color_range(rock_image, rgb_range=((130, 250), (90, 200), (0, 40)))
```
![alt text][w_imgs/rock_threshed.png]

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

```python
def rover_cam_to_map_view_coords(dst_img):
    # Get source and destination squares to transform image from rover camera
    # perspective, to a map view perspective.
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([
            [dst_img.shape[1]/2 - dst_size, dst_img.shape[0] - bottom_offset],
            [dst_img.shape[1]/2 + dst_size, dst_img.shape[0] - bottom_offset],
            [dst_img.shape[1]/2 + dst_size, dst_img.shape[0] - 2*dst_size - bottom_offset],
            [dst_img.shape[1]/2 - dst_size, dst_img.shape[0] - 2*dst_size - bottom_offset],
            ])
    return source, destination

def process_image(img):
    # From each img, generate a mosaic with 3 sub-images:
    #  - copy of input img
    #  - warped img converted from rover-POV to map-view
    #  - ground truth map overlayed with navigable map and yellow rocks found

        # Warp camera to map-view
    source, destination = rover_cam_to_map_view_coords(img)
    warped = perspect_transform(img, source, destination)
        # Apply thresholds to detect navigable map and rocks first
    nav_thres = color_thresh(warped, rgb_thresh=(180, 160, 150))
    rock_range = color_range(warped, rgb_range=((130, 250), (90, 200), (0, 40)))

        # Calculate navigable pixel values in rover-centric coords
    xpix_nav, ypix_nav = rover_coords(nav_thres)
        # world map is 1pix = 1m, our perspect_transform() produces 10pix = 1m
    scale = 10
        # Convert from rover-centric to worldmap coords
    xpix_nav, ypix_nav = pix_to_world(xpix_nav, ypix_nav,
                                      data.xpos[data.count], data.ypos[data.count],
                                      data.yaw[data.count],
                                      data.worldmap.shape[0],
                                      scale)
    data.worldmap[ypix_nav, xpix_nav, :] = 50  # paint grey

        # Repeat the procedure to show rocks on the map
    xpix_rock, ypix_rock = rover_coords(rock_range)
    xpix_rock, ypix_rock = pix_to_world(xpix_rock, ypix_rock,
                                        data.xpos[data.count], data.ypos[data.count],
                                        data.yaw[data.count],
                                        data.worldmap.shape[0],
                                        scale)
    data.worldmap[ypix_rock, xpix_rock] = (255, 0, 0)  # paint red

        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
    
        # Top left image: original
    output_image[0:img.shape[0], 0:img.shape[1]] = img
    
        # Top right image: Overlay warped with detected thresholds and rocks
    # warped_percept = np.zeros_like(warped)
    # warped_percept[nav_thres] = (0, 255, 255)
    # warped_percept[rock_range] = (255, 0, 0)
    # warped = cv2.addWeighted(warped, 1, warped_percept, 1, 0)  # comment this line to avoid overlay
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Bottom left: Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.2, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:map_add.shape[1]] = np.flipud(map_add)
    
        # Bottom right: Unused for now

    data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
```

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]


