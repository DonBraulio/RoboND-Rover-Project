import numpy as np

steering_counter = 0
steering = False
seen_rock_flag = False
lost_rock_counter = 0
last_seen_rock = 0
# last_steering = 0
rock_seeking_counter = 0
locked_counter = 0
last_nav_angle = 0
recovering_rock = False


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
    global seen_rock_flag
    global lost_rock_counter
    global last_seen_rock
    # global last_steering
    global locked_counter
    global recovering_rock
    global last_nav_angle
    Rover.debug_txt = ''
    Rover.sensors_txt = ''

    def add_obstacle_avoiding_offset(target_angle, margin=40):
        nearest_object_target = get_nearest_object(Rover, target_angle, 10)  # never is zero
        nearest_object_left = get_nearest_object(Rover, target_angle + 15, 5)
        nearest_object_right = get_nearest_object(Rover, target_angle - 15, 5)
        if nearest_object_left < margin/2:
            offset = -3 * (margin/2) / nearest_object_left
            Rover.debug_txt += " >> {:.0f} | ".format(offset)
            target_angle += offset
        if nearest_object_right < margin/2:
            offset = 3 * (margin/2) / nearest_object_right
            Rover.debug_txt += " << {:.0f} | ".format(offset)
            target_angle += offset

        if nearest_object_target < margin:
            Rover.debug_txt += "C"
            if nearest_object_left > margin:  # check left before right (preferred dir)
                offset = 5 * (margin / nearest_object_target)
                target_angle = target_angle + offset
                Rover.debug_txt += " | L: {:.0f}".format(offset)
            elif nearest_object_right > margin:
                offset = - 5 * (margin / nearest_object_target)
                target_angle = target_angle + offset
                Rover.debug_txt += " | R: {:.0f}".format(offset)

        return np.clip(target_angle, -15, 15)

    # Check if we have vision data to make decisions with
    if len(Rover.nav_angles):
        if (np.abs(Rover.vel) < 0.2 and not Rover.picking_up):
            locked_counter += 1
        elif np.abs(Rover.vel) > 0.5:
            locked_counter = 0

        if locked_counter > 200:
            Rover.brake = 0
            Rover.throttle = -5
            Rover.steer = -15
            if locked_counter > 300:
                Rover.throttle = 0
            if locked_counter > 400:
                Rover.steer = 15
            if locked_counter > 500:
                locked_counter = 80
            Rover.debug_txt += " UNLOCK!"
            return Rover

        # First priority: if we're near sample, brake and pick it up
        if Rover.near_sample:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set

        else:

            if Rover.seeing_rock:
                seen_rock_flag = True
                last_seen_rock = np.mean(Rover.rock_angles)
                rock_seeking_counter = 35

            if seen_rock_flag:  # see the rock two frames
                go_margin = 10
                steer_margin = 5
                rock_dist = np.min(Rover.rock_dists)
                deviation = np.abs(last_seen_rock)
                if (not recovering_rock and deviation < go_margin) or rock_dist > 30:
                    rock_dist_factor = rock_dist / (Rover.max_view_distance / 2)
                    rock_dist_factor = np.clip(rock_dist_factor,  0.2, 1)
                    target_angle = add_obstacle_avoiding_offset(last_seen_rock, 20)
                    target_speed = 1 * rock_dist_factor + 1
                else:
                    recovering_rock = deviation < steer_margin
                    target_angle = last_seen_rock
                    target_speed = 0

                # Ensure that we aren't in this state forever
                if rock_seeking_counter:
                    rock_seeking_counter -= 1
                else:
                    seen_rock_flag = False  # we lost the rock, let it go man
            else:
                nav_angles = Rover.nav_angles
                closed_boundary = len(Rover.obs_dists) and len(Rover.nav_dists)\
                                  and np.max(Rover.nav_dists) < (np.max(Rover.obs_dists) * 0.5)
                if closed_boundary:
                    Rover.debug_txt += " CLOSED"

                target_angle = np.mean(nav_angles) + 10
                target_angle = np.clip(target_angle, -15, 15)
                target_angle = add_obstacle_avoiding_offset(target_angle, 40)

                nearest_object_ahead = get_nearest_object(Rover, 0, 20)  # obstacle in narrow span ahead
                nearest_around = get_nearest_object(Rover, 0, 40)  # obstacle near in the whole visual range
                Rover.sensors_txt += "AH: {:.0f} | AR: {:.0f} | TG: {:.0f}"\
                        .format(nearest_object_ahead, nearest_around, target_angle)

                if not closed_boundary and not steering and nearest_object_ahead > 20 and nearest_around > 15:
                    target_speed = 2 * (nearest_object_ahead / 20 - (np.abs(target_angle) / 15))
                    target_speed = np.clip(target_speed, 0.4, 5)
                    last_nav_angle = target_angle
                else:
                    target_angle = 10 if last_nav_angle < 0 else -10  # opposite to where I came
                    steering = nearest_object_ahead < 25 or nearest_around < 10
                    target_speed = 0

            Rover.steer = np.clip(target_angle, -15, 15)

            current_speed = np.abs(Rover.vel)
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

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

