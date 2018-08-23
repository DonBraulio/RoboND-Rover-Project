import numpy as np
from numpy.linalg import norm

steering_counter = 0
steering = False
lost_rock_counter = 0
last_seen_rock = 0
# last_steering = 0
rock_seeking_counter = 0
locked_counter = 0
last_nav_angle = 0
recovering_rock = False
initial_pos = None


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


def decision_step(Rover):
    global rock_seeking_counter
    global steering
    global lost_rock_counter
    global last_seen_rock
    # global last_steering
    global locked_counter
    global recovering_rock
    global last_nav_angle
    global initial_pos
    Rover.debug_txt = ''
    Rover.sensors_txt = ''

    if initial_pos is None:
        initial_pos = np.array(Rover.pos)

    current_speed = np.abs(Rover.vel)

    def unlock_mechanism():
        global locked_counter
        # Lock watchdog increment
        if (current_speed < 0.2 and not Rover.picking_up):
            locked_counter += 1
        elif current_speed > 0.5:
            locked_counter = 0
        # Count reached. Activate unlocking
        if locked_counter > 300:
            Rover.brake = 0
            Rover.throttle = -5
            Rover.steer = -15
            if locked_counter > 400:
                Rover.throttle = 0
            if locked_counter > 500:
                Rover.steer = 15
            if locked_counter > 800:
                locked_counter = 300
            Rover.debug_txt += " UNLOCK!"
            return True
        return False

    def add_obstacle_avoiding_offset(target_angle, margin=40):
        nearest_object_target = get_nearest_object(Rover, target_angle, 10)  # never is zero
        nearest_object_left = get_nearest_object(Rover, 20, 15)
        nearest_object_right = get_nearest_object(Rover, -20, 15)
        if nearest_object_left < margin:
            offset = -6 * ((margin) / nearest_object_left) ** 2  # I'm very afraid of rocks!
            Rover.debug_txt += " >> {:.0f} | ".format(offset)
            target_angle += offset
        if nearest_object_right < margin:
            offset = 6 * ((margin) / nearest_object_right) ** 2
            Rover.debug_txt += " << {:.0f} | ".format(offset)
            target_angle += offset

        if nearest_object_target < margin:
            Rover.debug_txt += "C"
            nearest_object_left = get_nearest_object(Rover, target_angle + 15, 5)
            nearest_object_right = get_nearest_object(Rover, target_angle - 15, 5)
            if nearest_object_left > margin:  # check left before right (preferred dir)
                offset = 5 * (margin / nearest_object_target)
                target_angle = target_angle + offset
                Rover.debug_txt += " | L: {:.0f}".format(offset)
            elif nearest_object_right > margin:
                offset = - 5 * (margin / nearest_object_target)
                target_angle = target_angle + offset
                Rover.debug_txt += " | R: {:.0f}".format(offset)

        return np.clip(target_angle, -15, 15)

    if unlock_mechanism():
        return Rover

    # Check if we have vision data to make decisions with
    if len(Rover.nav_angles):
        # First priority: if we're near sample, brake and pick it up
        if Rover.near_sample:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set

        else:
            if Rover.seeing_rock:
                rock_seeking_counter = 200
                last_seen_rock = np.mean(Rover.rock_angles)

            if rock_seeking_counter:  # see the rock two frames
                rock_dist = np.min(Rover.rock_dists)
                deviation = np.abs(last_seen_rock)
                if (not recovering_rock and deviation < 20) or rock_dist > 30:
                    rock_dist_factor = rock_dist / (Rover.max_view_distance / 2)
                    rock_dist_factor = np.clip(rock_dist_factor,  0.2, 1)
                    target_angle = add_obstacle_avoiding_offset(last_seen_rock, 20)
                    target_speed = 1 * rock_dist_factor + 1
                else:
                    recovering_rock = deviation < 5
                    target_angle = last_seen_rock
                    target_speed = 0

                # Ensure that we aren't in this state forever
                rock_seeking_counter -= 1
            else: 

                # Return to initial_pos!!!
                if Rover.samples_to_find == Rover.samples_collected:
                    vec_to_orig = initial_pos - np.array(Rover.pos)  # points to origin
                    vec_to_orig[0] += 0 if np.abs(vec_to_orig[0]) > 1e-6 else 1e-6
                    angle_to_orig = (180 / np.pi) * np.arctan(vec_to_orig[1]/vec_to_orig[0])
                    angle_to_orig = angle_to_orig if vec_to_orig[0] > 0 else -angle_to_orig
                    dist_to_orig = norm(vec_to_orig)
                    err_to_orig = Rover.yaw - angle_to_orig
                    while err_to_orig > 180:
                        err_to_orig -= 360
                    while err_to_orig < -180:
                        err_to_orig += 360

                    target_angle = -err_to_orig
                    print('GO BACK HOME! {:.1f} < {:.0f}'.format(norm(vec_to_orig), err_to_orig))
                    # Reached destination!
                    if dist_to_orig < 5:
                        Rover.brake = Rover.brake_set
                        Rover.throttle = 0
                        return Rover
                else:
                    target_angle = np.mean(Rover.nav_angles * Rover.visited_ponderators) + 2  # tend to left

                closed_boundary = len(Rover.obs_dists) and len(Rover.nav_dists)\
                                  and np.max(Rover.nav_dists) < (np.max(Rover.obs_dists) * 0.5)

                target_angle = np.clip(target_angle, -15, 15)
                target_angle = add_obstacle_avoiding_offset(target_angle, 40)

                nearest_object_ahead = get_nearest_object(Rover, 0, 10)  # obstacle in narrow span ahead
                nearest_around = get_nearest_object(Rover, 0, 50)  # obstacle near in the whole visual range
                Rover.sensors_txt += "{:.0f} | {:.0f} "\
                        .format(nearest_around, nearest_object_ahead)
                Rover.sensors_txt += "P: {:.0f} | R: {:.0f}"\
                        .format(Rover.pitch, Rover.roll)

                if not closed_boundary and not steering and nearest_object_ahead > 20:
                    target_speed = 2 * (nearest_object_ahead / 20 - (np.abs(target_angle) / 15))
                    target_speed = np.clip(target_speed, 0.4, 5)
                    last_nav_angle = np.mean(Rover.nav_angles)
                    if current_speed < 0.2:  # avoid steering when we're starting throttle
                        target_angle = 0
                else:
                    target_angle = -10 if last_nav_angle < 0 else 10
                    disbalance = np.abs(add_obstacle_avoiding_offset(0, 40))  # how much should I deviate?
                    steering = disbalance > 5 or nearest_object_ahead < 25
                    target_speed = 0

            Rover.steer = np.clip(target_angle, -15, 15)
            if current_speed > 0.2 and current_speed > 2.5 * target_speed:  # includes braking until 0
                Rover.brake = Rover.brake_set
                Rover.throttle = 0
            elif target_speed == 0:  # Steer around
                Rover.brake = 0
                Rover.throttle = 0
            else:
                # Set throttle value to throttle setting
                Rover.throttle = Rover.throttle_set * ((target_speed - Rover.vel) / target_speed)
                Rover.brake = 0
    else:
        Rover.steer = 15  # avoid blocking after pickup in a blind spot
        Rover.brake = 0
        Rover.throttle = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        rock_seeking_counter = 0
    
    return Rover

