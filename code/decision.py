import time
import numpy as np

from numpy.linalg import norm


# Get the N minimum values from array
def get_n_min_values(arr, N):
    return arr[arr.argsort()[:min(N, len(arr))]]


np.seterr(all='raise')
# Detect nearest obstacles in the target angle, with the given angle span
def get_nearest_object(Rover, target_angle, span=5):
    if not len(Rover.obs_angles):
        return Rover.max_view_distance

    # Obstacles with angles near target_angle
    target_mask = np.abs(Rover.obs_angles - target_angle) < span
    obstacles_ahead = Rover.obs_dists[target_mask]
    if len(obstacles_ahead) < 20:  # avoid false obstacles (e.g: wheel marks)
        return Rover.max_view_distance
    return np.mean(get_n_min_values(obstacles_ahead, 30)) - Rover.min_view_distance


# Add two components of offsets to the target angle:
#  - One to run from obstacles dangerously near
#  - Another to avoid going towards obstacles ahead
def add_obstacle_avoiding_offset(Rover, target_angle, margin=40):
    target_angle = np.clip(target_angle, -15, 15)

    # Short range crash avoid: offset increases with inverse distance to near objects
    nearest_object_left = get_nearest_object(Rover, 20, 20)
    nearest_object_right = get_nearest_object(Rover, -20, 20)
    Rover.debug_txt += "L: {:.0f} R: {:.0f}".format(nearest_object_left, nearest_object_right)
    offset = 0
    if nearest_object_left < margin:
        offset = -3 * (margin / nearest_object_left) ** 2 # I'm very afraid of rocks!
    if nearest_object_right < margin:
        offset += 3 * (margin / nearest_object_right) ** 2
    if offset:
        if min(nearest_object_left, nearest_object_right) < margin/2:
            target_angle = offset
        else:
            target_angle += offset
        Rover.debug_txt += " {} {:.0f} | ".format('<<' if offset > 0 else '>>', offset)

    return np.clip(target_angle, -15, 15)


# Given a target yaw to reach a destination, get the steering in
# between -15 and 15. Avoid switching at 180 cause it gets unstable
def get_steering_to(Rover, target_direction):
    err = Rover.yaw - target_direction
    while err > 180:
        err -= 360
    while err < -180:
        err += 360
    if np.abs(err) > 170:  # hysteresis
        err = 15
    return np.clip(-err, -15, 15)  # this is our target angle


# Rocks are very near walls, we've to navigate differently
def go_towards_rock(Rover):
    rock_dist = np.min(Rover.rock_dists)
    deviation = np.abs(Rover.last_seen_rock)
    if (not Rover.recovering_rock and deviation < 40) or rock_dist > 30:
        rock_dist_factor = rock_dist / (Rover.max_view_distance / 2)
        rock_dist_factor = np.clip(rock_dist_factor,  0.2, 1)
        target_angle = add_obstacle_avoiding_offset(Rover, Rover.last_seen_rock, 40)
        target_speed = 1 * rock_dist_factor + 1
    else:  # steer to center rock again
        Rover.recovering_rock = deviation > 5
        target_angle = Rover.last_seen_rock
        target_speed = 0
    return target_angle, target_speed


# Avoid obstacles and calculate speed (including STOP condition)
def go_towards_direction(Rover, preferred_direction):
    nearest_object_ahead = get_nearest_object(Rover, 0, 10)  # obstacle in narrow span ahead
    Rover.pos_txt += "({:.0f})".format(nearest_object_ahead)

    target_speed = 0
    target_angle = add_obstacle_avoiding_offset(Rover, preferred_direction, 40)
    deviation = np.abs(target_angle)
    if not closed_boundary(Rover) and not Rover.steering and nearest_object_ahead > 15:
        if deviation >= 14:  # hard curve, fix to low speed
            target_speed = 1
        else:
            target_speed = 4 * min(nearest_object_ahead / 40, 15 / deviation)
            target_speed = 0 if target_speed < 0.2 else np.clip(target_speed, 0.4, 5)
        if Rover.speed < 0.2 and target_speed > 0:  # avoid steering when we're starting throttle
            target_angle = 0

    # Steer on null speed
    if not target_speed:
        if Rover.steering_dir is None:
            print("Started steering...")
            # steer back to navigate dir 3/4 of the times
            Rover.steering_dir = get_steering_to(Rover, Rover.last_nav_yaw)\
                                 * np.random.choice([1, 1, 1, -1])
        target_angle = (-10 if Rover.steering_dir < 0 else 10)
        Rover.steering = nearest_object_ahead < 25
    else:
        Rover.last_nav_yaw = Rover.yaw
        Rover.steering_dir = None
    return target_angle, target_speed


def set_rover_to(Rover, target_angle, target_speed):
    Rover.steer = np.clip(target_angle, -15, 15)
    if Rover.speed > 0.2 and Rover.speed > 2.5 * target_speed:  # includes braking until 0
        Rover.brake = Rover.brake_set
        Rover.throttle = 0
    elif target_speed == 0:  # Steer around
        Rover.brake = 0
        Rover.throttle = 0
    else:
        Rover.throttle = Rover.throttle_set * ((target_speed - Rover.vel) / target_speed)
        Rover.brake = 0


# Detect when there's no visible way out
def closed_boundary(Rover):
    return len(Rover.obs_dists) and len(Rover.nav_dists)\
           and np.max(Rover.nav_dists) < (np.max(Rover.obs_dists) * 0.5)


def unlock_mechanism(Rover):
    # Lock watchdog increment
    if (Rover.speed < 0.2 and not Rover.picking_up and not Rover.steering):  # count to detect lock
        Rover.locked_counter += 1
        Rover.locked_pos = np.array(Rover.pos)
    elif Rover.locked_counter and norm(Rover.locked_pos - Rover.pos) > 2:  # exit lock mode
        Rover.locked_counter = 0
    elif Rover.picking_up:
        Rover.locked_counter = 0

    # Count reached. Activate unlocking
    if Rover.locked_counter > 150:  # 1 sec ~ 30 secs
        Rover.brake = 0
        Rover.throttle = -Rover.throttle_set  # Phase 1: reverse
        Rover.steer = 0
        if Rover.locked_counter > 210:  # Phase 2: rotate
            Rover.brake = 0
            Rover.throttle = 0
            Rover.steer = 90
        if Rover.locked_counter > 240:  # Phase 3: try yielding control to visual navigation again
            Rover.locked_counter = 0
        Rover.debug_txt += " UNLOCK!"
        return True
    return False


def finished_mission(Rover):
    Rover.dist_to_orig = norm(Rover.initial_pos - Rover.pos)
    if Rover.samples_collected == Rover.samples_to_find\
            and Rover.dist_to_orig < 3:
        Rover.brake = Rover.brake_set
        Rover.throttle = 0
        Rover.debug_txt = 'THE END'
        Rover.mode_txt = 'Home sweet home'
        return True
    return False


def get_point_direction(Rover, target_point):
    vector = target_point - np.array(Rover.pos)  # vector pointing to target_direction
    vector[0] += 0 if np.abs(vector[0]) > 1e-6 else 1e-6  # avoid div by zero
    vector_arg = (180 / np.pi) * np.arctan(vector[1]/vector[0])
    vector_arg = vector_arg if vector[0] > 0 else 180 + vector_arg
    return get_steering_to(Rover, vector_arg)


# Handle all navigation modes except rock pursuing and picking
def select_nav_mode(Rover):
    Rover.prev_nav_mode = Rover.nav_mode

    # Restart in NAV_LEFT mode after picking rocks
    if Rover.nav_mode == Rover.NAV_TO_ROCK:
        Rover.nav_mode = Rover.NAV_BIAS_LEFT

    # Switch mode on timeout
    if Rover.nav_mode_counter > 30 * 60:  # aprox 30 frames/second
        if Rover.samples_collected == Rover.samples_to_find\
                and Rover.nav_mode != Rover.NAV_BACK_HOME:
            Rover.nav_mode = Rover.NAV_BACK_HOME  # will enter here at least 1 min after finishing
        elif Rover.nav_mode == Rover.NAV_BIAS_LEFT:
            Rover.nav_mode = Rover.NAV_MEAN
        elif Rover.nav_mode == Rover.NAV_MEAN:
            Rover.nav_mode = Rover.NAV_BIAS_LEFT

    # count how much time we spend in a mode, without exploring new terrain
    if Rover.nav_mode == Rover.prev_nav_mode and Rover.visit_gain < 0.2:
        Rover.nav_mode_counter += 1
    else:
        Rover.nav_mode_counter = 0
    return Rover.nav_mode


def decision_step(Rover):
    Rover.debug_txt = ''
    Rover.pos_txt = '{:.0f} | {:.0f}'.format(Rover.pos[0], Rover.pos[1])
    Rover.speed = np.abs(Rover.vel)

    if Rover.initial_pos is None:
        Rover.initial_pos = np.array(Rover.pos)

    # Check mission and emergency unlocking
    if finished_mission(Rover) or unlock_mechanism(Rover):
        return Rover

    # First priority: if we're near sample, brake and pick it up
    if Rover.near_sample:
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        if Rover.vel == 0:
            if not Rover.picking_up:
                Rover.send_pickup = True
            Rover.rock_seeking_counter = 0
    else:
        # Second priority: seeing rock, go towards it
        if Rover.seeing_rock:
            Rover.nav_mode = Rover.NAV_TO_ROCK
            Rover.rock_seeking_counter = 400
            Rover.last_seen_rock = np.mean(Rover.rock_angles)

        # Third priority: explore map in different modes
        elif not Rover.rock_seeking_counter:
            Rover.nav_mode = select_nav_mode(Rover)

        # Navigate according to the current mode
        target_speed = None
        if Rover.nav_mode == Rover.NAV_TO_ROCK:
            target_angle, target_speed = go_towards_rock(Rover)
            Rover.rock_seeking_counter -= 1
            Rover.mode_txt = "Pick Rock"

        elif Rover.nav_mode == Rover.NAV_MEAN:
            target_angle = np.mean(Rover.nav_angles * Rover.visited_ponderators)
            Rover.mode_txt = "Free Mode"

        elif Rover.nav_mode == Rover.NAV_BIAS_LEFT:
            target_angle = np.mean(Rover.nav_angles) + 9
            Rover.mode_txt = "Left Crawl"

        elif Rover.nav_mode == Rover.NAV_BACK_HOME:
            target_angle = get_point_direction(Rover, Rover.initial_pos)
            Rover.mode_txt = "Go Home {}".format(Rover.dist_to_orig)

        Rover.mode_txt += " {:.0f}".format(100 * Rover.nav_mode_counter / (30*60))

        # We've got a destination, calculate speed if not set by nav_mode
        if target_speed is None:
            target_angle, target_speed = go_towards_direction(Rover, target_angle)

        # Set throttle/brake/steer
        set_rover_to(Rover, target_angle, target_speed)

    return Rover

