import numpy as np
import cv2

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
        self.max_view_distance = rows/2
        self.view_mask = self.calculate_view_mask()

    def set_img_shape(self, rows, cols):
        if rows != self.rows or cols != self.columns:
            self.init_img_shape(rows, cols)

    # Get a mask of the valid image range, filtering by vision angle and distance to camera
    def calculate_view_mask(self):
        # transform the whole image to get valid angle of vision
        view_mask = cv2.warpPerspective(np.ones((self.rows, self.columns)),
                                        self.perspective_M,
                                        (self.width, self.height))
        # mask by distance to the camera point
        xview, yview = view_mask.nonzero()
        distances_to_camera = np.sqrt((xview - self.x_camera)**2 + (yview - self.y_camera)**2)
        away_mask = distances_to_camera > self.max_view_distance
        # filter by distance mask
        view_mask[xview[away_mask], yview[away_mask]] = 0
        return view_mask

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
    angles = (180 / np.pi) * np.arctan2(y_pixel, x_pixel)
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

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
        # Set camera size (will update transform params only on change)
    RoverCam.set_img_shape(Rover.img.shape[0], Rover.img.shape[1])
        # Warp camera to map-view
    warped = perspect_transform(Rover.img, RoverCam.perspective_M)
        # Apply thresholds to detect navigable map and rocks first
    nav_thres = color_thresh(warped, rgb_thresh=(180, 160, 150)) * RoverCam.view_mask
    rock_range = color_range(warped, rgb_range=((130, 250), (90, 200), (0, 40)))
    obstacles = np.abs(1 - nav_thres) * RoverCam.view_mask  # opossite to navigable terrain

        # Calculate navigable pixel values in rover-centric coords
    xpix_rov, ypix_rov = rover_coords(nav_thres)
    xpix_obs_rov, ypix_obs_rov = rover_coords(obstacles)
    xpix_rock_rov, ypix_rock_rov = rover_coords(rock_range)

    dist, angles = to_polar_coords(xpix_rov, ypix_rov)
    Rover.nav_dists = dist
    Rover.nav_angles = angles

    roll_err = np.abs(Rover.roll if Rover.roll < 180 else Rover.roll - 360)
    pitch_err = np.abs(Rover.pitch if Rover.pitch < 180 else Rover.pitch - 360)

        # world map is 1pix = 1m, our perspect_transform() produces 10pix = 1m
    scale = 10
        # Convert from rover-centric to worldmap coords
    xpix_nav, ypix_nav = pix_to_world(xpix_rov, ypix_rov,
                                      Rover.pos[0], Rover.pos[1],
                                      Rover.yaw,
                                      Rover.worldmap.shape[0],
                                      scale)

        # Visited places have lesser values [0, 1], this is used to ponderate mean angle
    Rover.visited_ponderators = (280 - Rover.worldmap[ypix_nav, xpix_nav, 2]) / 280

    # only add points to worldmap when we've small pitch and roll
    if roll_err < 1.5 and pitch_err < 2.0:
        sure_mask = dist < 30
        ypix_nav_sure = ypix_nav[sure_mask]
        xpix_nav_sure = xpix_nav[sure_mask]
        Rover.worldmap[ypix_nav_sure, xpix_nav_sure, 2] += 10

            # Repeat the transformation to show obstacles on the map
        xpix_obs, ypix_obs = pix_to_world(xpix_obs_rov, ypix_obs_rov,
                                            Rover.pos[0], Rover.pos[1],
                                            Rover.yaw,
                                            Rover.worldmap.shape[0],
                                            scale)
        Rover.worldmap[ypix_obs, xpix_obs, 0] += 1
            # Remove objects from any navigable zones
        Rover.worldmap[Rover.worldmap[:, :, 2].nonzero(), 0] = 0

            # Repeat the procedure to show rocks on the map
        xpix_rock, ypix_rock = pix_to_world(xpix_rock_rov, ypix_rock_rov,
                                            Rover.pos[0], Rover.pos[1],
                                            Rover.yaw,
                                            Rover.worldmap.shape[0],
                                            scale)
        Rover.worldmap[ypix_rock, xpix_rock, 1] += 1

        Rover.worldmap = np.clip(Rover.worldmap, 0, 255)

    Rover.vision_image[:,:,0] = 255 * obstacles
    # Rover.vision_image[:,:,2] = 255*nav_thres
    Rover.vision_image[:,:,1] = 255 * rock_range

    # Show the ponderators in the image
    ypix_img, xpix_img = nav_thres.nonzero()
    Rover.vision_image[ypix_img, xpix_img, 2] = 255 * Rover.visited_ponderators

    # # Obstacle avoid: if we see an obstacle ahead, only see to the left
    if len(xpix_obs_rov):
        dist, angles = to_polar_coords(xpix_obs_rov, ypix_obs_rov)
        Rover.obs_dists = dist
        Rover.obs_angles = angles

    if len(xpix_rock_rov):
        Rover.seeing_rock = True
        dist, angles = to_polar_coords(xpix_rock_rov, ypix_rock_rov)
        Rover.rock_dists = dist
        Rover.rock_angles = angles
    else:
        Rover.seeing_rock = False
        
    return Rover
