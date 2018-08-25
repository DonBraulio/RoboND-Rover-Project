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
    try:
        target_mask = np.abs(Rover.obs_angles - target_angle) < span
        obstacles_ahead = Rover.obs_dists[target_mask]
        if len(obstacles_ahead) < 20:  # avoid false obstacles (e.g: wheel marks)
            return Rover.max_view_distance
        return np.mean(get_n_min_values(obstacles_ahead, 30))
    except RuntimeWarning:
        import pdb; pdb.set_trace()


# Add two components of offsets to the target angle:
#  - One to run from obstacles dangerously near
#  - Another to avoid going towards obstacles ahead
def add_obstacle_avoiding_offset(Rover, target_angle, margin=40):
    target_angle = np.clip(target_angle, -15, 15)

    # Short range crash avoid: offset increases cuadratically with inverse distance to near objects
    nearest_object_left = get_nearest_object(Rover, 20, 15)
    nearest_object_right = get_nearest_object(Rover, -20, 15)
    if nearest_object_left < margin:
        offset = -6 * (margin / nearest_object_left) ** 2  # I'm very afraid of rocks!
        Rover.debug_txt += " >> {:.0f} | ".format(offset)
        target_angle += offset
    if nearest_object_right < margin:
        offset = 6 * (margin / nearest_object_right) ** 2
        Rover.debug_txt += " << {:.0f} | ".format(offset)
        target_angle += offset

    # Avoid obstacles ahead: check if we have better navigability at left (prefer) or right
    nearest_object_target = get_nearest_object(Rover, target_angle, 10)  # never is zero
    if nearest_object_target < margin:
        Rover.debug_txt += "C"
        nearest_object_left = get_nearest_object(Rover, target_angle + 15, 5)
        nearest_object_right = get_nearest_object(Rover, target_angle - 15, 5)
        if nearest_object_left > margin:  # check left before right (preferred dir)
            offset = 5 * (margin / nearest_object_target)
            target_angle = target_angle + offset
            Rover.debug_txt += " L: {:.0f} |".format(offset)
        elif nearest_object_right > margin:
            offset = - 5 * (margin / nearest_object_target)
            target_angle = target_angle + offset
            Rover.debug_txt += " R: {:.0f} |".format(offset)
    return np.clip(target_angle, -15, 15)


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


def go_back_home(Rover):
    vec_to_orig = Rover.initial_pos - np.array(Rover.pos)  # points to origin
    vec_to_orig[0] += 0 if np.abs(vec_to_orig[0]) > 1e-6 else 1e-6
    angle_to_orig = (180 / np.pi) * np.arctan(vec_to_orig[1]/vec_to_orig[0])
    angle_to_orig = angle_to_orig if vec_to_orig[0] > 0 else -angle_to_orig
    Rover.dist_to_orig = norm(vec_to_orig)
    err_to_orig = Rover.yaw - angle_to_orig
    while err_to_orig > 180:
        err_to_orig -= 360
    while err_to_orig < -180:
        err_to_orig += 360
    return -err_to_orig  # this is our target angle


# Avoid obstacles and calculate speed (including STOP condition)
def go_towards_direction(Rover, preferred_direction):
    target_angle = add_obstacle_avoiding_offset(Rover, preferred_direction, 40)

    nearest_object_ahead = get_nearest_object(Rover, 0, 10)  # obstacle in narrow span ahead
    nearest_around = get_nearest_object(Rover, 0, 50)  # obstacle near in the whole visual range
    Rover.sensors_txt += "{:.0f} | {:.0f} "\
            .format(nearest_around, nearest_object_ahead)
    Rover.sensors_txt += "P: {:.0f} | R: {:.0f}"\
            .format(Rover.pitch, Rover.roll)

    target_speed = 0
    if not closed_boundary(Rover) and not Rover.steering and nearest_object_ahead > 15:
        target_speed = 2 * (nearest_object_ahead / 20 - (np.abs(target_angle) / 15))
        target_speed = 0 if target_speed < 0.2 else np.clip(target_speed, 0.4, 5)
        if Rover.speed < 0.2:  # avoid steering when we're starting throttle
            target_angle = 0
        Rover.last_nav_angle = np.mean(Rover.nav_angles)

    # Steer on null speed
    if not target_speed:
        target_angle = -10 if Rover.last_nav_angle < 0 else 10
        disbalance = np.abs(add_obstacle_avoiding_offset(Rover, 0, 40))  # how much should I deviate?
        Rover.steering = disbalance > 5 or nearest_object_ahead < 25
        target_speed = 0
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
    if (Rover.speed < 0.2 and not Rover.picking_up):
        Rover.locked_counter += 1
    elif Rover.speed > 0.5:
        Rover.locked_counter = 0
    # Count reached. Activate unlocking
    if Rover.locked_counter > 500:
        Rover.brake = 0
        Rover.throttle = 0
        Rover.steer = -15
        if Rover.locked_counter > 600:
            Rover.throttle = -5
        if Rover.locked_counter > 700:
            Rover.throttle = 0
            Rover.steer = 15
        if Rover.locked_counter > 800:
            Rover.locked_counter = 0  # try yielding control to visual navigation again
        Rover.debug_txt += " UNLOCK!"
        return True
    return False


def finished_mission(Rover):
    if Rover.dist_to_orig < 5:  # only calculated when all rocks get picked up
        Rover.debug_txt = 'THE END'
        Rover.brake = Rover.brake_set
        Rover.throttle = 0
        return True
    return False


def decision_step(Rover):
    Rover.debug_txt = ''
    Rover.sensors_txt = ''
    Rover.speed = np.abs(Rover.vel)

    if Rover.initial_pos is None:
        Rover.initial_pos = np.array(Rover.pos)

    # Check mission and emergency unlocking
    if finished_mission(Rover) or unlock_mechanism(Rover):
        return Rover

    # Check if we have vision data to make decisions with
    if len(Rover.nav_angles):
        # First priority: if we're near sample, brake and pick it up
        if Rover.near_sample:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
        else:
            if Rover.seeing_rock:
                Rover.rock_seeking_counter = 200
                Rover.last_seen_rock = np.mean(Rover.rock_angles)

            # Pursue a rock we just saw
            if Rover.rock_seeking_counter:  # see the rock two frames
                target_angle, target_speed = go_towards_rock(Rover)
                Rover.rock_seeking_counter -= 1
            else: 
                # Free navigation, prefer left and not visited places
                if Rover.samples_to_find != Rover.samples_collected:
                    target_angle = np.mean(Rover.nav_angles * Rover.visited_ponderators) + 4  # tend to left
                # Found all rocks: RETURN HOME
                else:
                    target_angle = go_back_home(Rover)

                # We've got a destination, find a way and speed
                target_angle, target_speed = go_towards_direction(Rover, target_angle)

            # Set throttle/brake/steer
            set_rover_to(Rover, target_angle, target_speed)
    else:
        # if not nav_angles, just steer
        set_rover_to(Rover, 15, 0)

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        Rover.rock_seeking_counter = 0
    
    return Rover

