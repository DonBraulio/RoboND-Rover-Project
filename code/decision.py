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

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
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

    def get_nearest_object(target_angle, span=5):
        obs_angles = Rover.obs_angles * 180 / np.pi
        target_mask = np.abs(obs_angles - target_angle) < span
        obstacles_ahead = Rover.obs_dists[target_mask]
        return np.mean(get_n_min_values(obstacles_ahead, 30))\
                       if len(obstacles_ahead) > 10\
                       else Rover.max_view_distance

    def add_obstacle_avoiding_offset(target_angle, margin=40):
        nearest_object_target = get_nearest_object(target_angle, 10)  # never is zero
        if nearest_object_target < margin:
            nearest_object_left = get_nearest_object(target_angle + 15, 5)
            nearest_object_right = get_nearest_object(target_angle - 15, 5)
            if nearest_object_left > margin:  # check left before right (preferred dir)
                target_angle = target_angle + 5 * (margin / nearest_object_target)
                # print("PREFERRING LEFT: {}".format(target_angle))
            elif nearest_object_right > margin:
                target_angle = target_angle - 5 * (margin / nearest_object_target)
                # print("PREFERRING RIGHT: {}".format(target_angle))

        return np.clip(target_angle, -15, 15)

    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        print("State: {}".format(Rover.mode))
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
            print("UNLOCKING!")
            return Rover

        # First priority: if we're near sample, brake and pick it up
        if Rover.near_sample:
            print("BRAKE! NEAR SAMPLE")
            Rover.throttle = 0
            Rover.brake = Rover.brake_set

        else:

            # visibility_factor = get_visibility_factor()
            # nearby_angles = (180 / np.pi) * Rover.obs_angles[Rover.obs_dists < 25]
            # angles_level_1 = nearby_angles[np.abs(nearby_angles) < 15]
            # angles_level_2 = nearby_angles[np.abs(nearby_angles) >= 15]
            # mean_angles_level_1 = np.mean(angles_level_1) if len(angles_level_1) > 10 else 0
            # mean_angles_level_2 = np.mean(angles_level_2) if len(angles_level_2) > 10 else 0
            # avoid_obstacle_offset = - np.clip((5 * mean_angles_level_1 + mean_angles_level_2), -15, 15)
            # if avoid_obstacle_offset:
            #     print("OFFSET 1: {} | OFFSET 2: {} | TOTAL: {}".format(mean_angles_level_1, mean_angles_level_2, avoid_obstacle_offset))

            if Rover.seeing_rock:
                seen_rock_flag = True
                last_seen_rock = np.mean(Rover.rock_angles) * 180 / np.pi
                rock_seeking_counter = 35

            if seen_rock_flag:  # see the rock two frames
                go_margin = 10
                steer_margin = 5
                rock_dist = np.min(Rover.rock_dists)
                deviation = np.abs(last_seen_rock)
                print("ROCK DISTANCE: {} | ANGLE: {}".format(rock_dist, last_seen_rock))
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
                nav_angles = 180 * Rover.nav_angles / np.pi
                closed_boundary = len(Rover.obs_dists) and len(Rover.nav_dists)\
                                  and np.max(Rover.nav_dists) < (np.max(Rover.obs_dists) * 0.5)

                target_angle = np.mean(nav_angles) + 10
                target_angle = np.clip(target_angle, -15, 15)
                # print("TARGET: {}".format(target_angle))
                target_angle = add_obstacle_avoiding_offset(target_angle, 40)

                nearest_object_ahead = get_nearest_object(0, 20)  # obstacle in narrow span ahead
                nearest_around = get_nearest_object(0, 40)  # obstacle near in the whole visual range
                print("NEAREST AHEAD: {} | AROUND: {}".format(nearest_object_ahead, nearest_around))

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

