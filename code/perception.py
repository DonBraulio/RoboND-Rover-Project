import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
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

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, M):
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

# Store camera perspective matrix and view_mask to avoid recalculation on every step
class RoverCamera():
    def __init__(self):
        # Init vars with default shape
        self.init_img_shape()

    @property
    def width(self):
        return self.columns

    @property
    def height(self):
        return self.rows

    def init_img_shape(self, rows=160, cols=320):
        self.rows = rows
        self.columns = cols
        src, dst = RoverCamera.rover_cam_to_map_coords(self.width, self.height)
        self.perspective_M = cv2.getPerspectiveTransform(src, dst)
        # filters by distance to camera and angle
        self.x_camera = self.width/2
        self.y_camera = self.height
        self.view_mask = cv2.warpPerspective(np.ones((self.rows, self.columns)),
                                             self.perspective_M,
                                             (self.width, self.height))
        self.calculate_view_mask_histogram()


    def set_img_shape(self, rows, cols):
        if rows != self.rows or cols != self.columns:
            self.init_img_shape(rows, cols)

    def get_target_direction(self, target_angle):
        if target_angle <= self.angle_ranges[1]:  # negative angles (right)
            return 'right' if target_angle > self.angle_ranges[0] else 'out_right'
        elif target_angle >= self.angle_ranges[2]:
            return 'left' if target_angle < self.angle_ranges[3] else 'out_left'
        else:
            return 'center'

    def calculate_view_mask_histogram(self):
        mask_xpix, mask_ypix = rover_coords(self.view_mask)
        mask_dist, mask_angles = to_polar_coords(mask_xpix, mask_ypix)
        self.dist_ranges = [0, 10, 20, 30]
        mask_angles_deg = mask_angles*180/np.pi
        self.angle_ranges = [min(mask_angles_deg), -5, 5, max(mask_angles_deg)]
        self.view_mask_H = np.histogram2d(mask_dist, mask_angles_deg,
                                          bins=(self.dist_ranges, self.angle_ranges))[0]

    def get_vision_indexes(self, distances, angles_rad):
        angles_deg = angles_rad*180/np.pi
        H = np.histogram2d(distances, angles_deg, bins=(self.dist_ranges, self.angle_ranges))[0]
        H_norm = H / self.view_mask_H
        H_norm += 1e-6  # ensure that no index is 0 to avoid zero division problems
        result = {'near_center': H_norm[0, 1], 'near_left': H_norm[0, 2], 'near_right': H_norm[0, 0],
                  'mid_center': H_norm[1, 1], 'mid_left': H_norm[1, 2], 'mid_right': H_norm[1, 0],
                  'far_center': H_norm[2, 1], 'far_left': H_norm[2, 2], 'far_right': H_norm[2, 0]}
        print(result)
        # result = {'near_center': 0.1, 'near_left': 0.1, 'near_right': 0.1,
        #          'mid_center': 0.1, 'mid_left': 0.1, 'mid_right': 0.1,
        #          'far_center': 0.1, 'far_left': 0.1, 'far_right': 0.1}
        return result

    @staticmethod
    def rover_cam_to_map_coords(dst_width, dst_height):
        # Get source and destination squares to transform image from rover camera
        # perspective, to a map view perspective.
        dst_size = 5
        bottom_offset = 6
        source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
        destination = np.float32([
                [dst_width/2 - dst_size, dst_height - bottom_offset],
                [dst_width/2 + dst_size, dst_height - bottom_offset],
                [dst_width/2 + dst_size, dst_height - 2*dst_size - bottom_offset],
                [dst_width/2 - dst_size, dst_height - 2*dst_size - bottom_offset],
                ])
        return source, destination


RoverCam = RoverCamera()

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
        # Set camera size (will update transform params only on change)
    RoverCam.set_img_shape(Rover.img.shape[0], Rover.img.shape[1])
        # Warp camera to map-view
    warped = perspect_transform(Rover.img, RoverCam.perspective_M)
        # Apply thresholds to detect navigable map and rocks first
    nav_thres = color_thresh(warped, rgb_thresh=(180, 160, 150))
    rock_range = color_range(warped, rgb_range=((130, 250), (90, 200), (0, 40)))
    obstacles = np.abs(1 - nav_thres)*RoverCam.view_mask  # opossite to navigable terrain

    Rover.vision_image[:,:,0] = 255*obstacles
    Rover.vision_image[:,:,2] = 255*nav_thres
    Rover.vision_image[:,:,1] = 255*rock_range

        # Calculate navigable pixel values in rover-centric coords
    xpix_rov, ypix_rov = rover_coords(nav_thres)
        # world map is 1pix = 1m, our perspect_transform() produces 10pix = 1m
    scale = 10
        # Convert from rover-centric to worldmap coords
    xpix_nav, ypix_nav = pix_to_world(xpix_rov, ypix_rov,
                                      Rover.pos[0], Rover.pos[1],
                                      Rover.yaw,
                                      Rover.worldmap.shape[0],
                                      scale)
    Rover.worldmap[ypix_nav, xpix_nav, 2] += 1

        # Repeat the transformation to show obstacles on the map
    xpix_obs_rov, ypix_obs_rov = rover_coords(obstacles)
    xpix_obs, ypix_obs = pix_to_world(xpix_obs_rov, ypix_obs_rov,
					Rover.pos[0], Rover.pos[1],
					Rover.yaw,
					Rover.worldmap.shape[0],
                                        scale)
    Rover.worldmap[ypix_obs, xpix_obs, 0] += 1

        # Repeat the procedure to show rocks on the map
    xpix_rock_rov, ypix_rock_rov = rover_coords(rock_range)
    xpix_rock, ypix_rock = pix_to_world(xpix_rock_rov, ypix_rock_rov,
					Rover.pos[0], Rover.pos[1],
					Rover.yaw,
					Rover.worldmap.shape[0],
                                        scale)
    Rover.worldmap[ypix_rock, xpix_rock, 1] += 1

    Rover.worldmap = np.clip(Rover.worldmap, 0, 255)

    dist, angles = to_polar_coords(xpix_rov, ypix_rov)
    Rover.nav_dists = dist
    Rover.nav_angles = angles

    if len(xpix_rock_rov):
        dist, angles = to_polar_coords(xpix_rock_rov, ypix_rock_rov)
        Rover.seeing_rock = True
        Rover.rock_dists = dist
        Rover.rock_angles = angles
    else:
        Rover.seeing_rock = False

    # mean_dir = np.mean(angles)
    # mean_dist = np.mean(dist)
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
 
    
    
    return Rover
