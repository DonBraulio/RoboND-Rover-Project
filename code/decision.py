import numpy as np
from perception import RoverCam

def mean_angle(angles):
    return np.clip(np.mean(angles * 180/np.pi), -15, 15)

def decision_step_new(Rover):
    vision = RoverCam.get_vision_indexes(Rover.nav_dists, Rover.nav_angles)
    target_angle = mean_angle(Rover.nav_angles)
    target_dir = RoverCam.get_target_direction(target_angle)  # center, right, left


steering_counter = 0
seen_rock_counter = 15
last_seen_rock_angle = 0
steering_direction = 0
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    global steering_counter
    global seen_rock_counter
    global last_seen_rock_angle
    global steering_direction

    vision = RoverCam.get_vision_indexes(Rover.nav_dists, Rover.nav_angles)

    def get_next_steering(target_angle=0):
        return 15 if target_angle >= 0 else -15

    def steer_to_find_rock():
        Rover.throttle = 0
        if Rover.vel:
            Rover.brake = Rover.brake_set
            Rover.steer = 0  # dont't steer, we might lose the rock
        elif Rover.seeing_rock:
            print("BRAKE: found the rock" )
            Rover.brake = 0
            Rover.steer = mean_angle(Rover.rock_angles)
        else:
            print("Steering to find rock")
            Rover.brake = 0
            Rover.steer = 15 if last_seen_rock_angle >= 0 else -15

    def not_crashing_offset():
        print("FORCING CRASH AVOID")
        L_R_ratio = vision['near_left']/vision['near_right']  # denom is never zero (see perception)
        if min(L_R_ratio, 1.0/L_R_ratio) <= 0.6:
            return 30 if vision['near_left'] > vision['near_right'] else -30
        else:
            return 0

    def obstacle_avoiding_offset():
        L_R_ratio = vision['far_left']/vision['far_right']  # denom is never zero (see perception)
        if min(L_R_ratio, 1.0/L_R_ratio) <= 0.8:
            return 3 if vision['far_left'] > vision['far_right'] else -3
        else:
            return 0

    def get_nav_angle(target_angle):
        best_angle = 0
        best_angle_err = 31
        ranges = {'mid_left': [5, 15], 'mid_right': [-15, -5], 'mid_center': [-5, 5]}
        for zone in ranges:
            if vision[zone] > 0.9:
                new_candidate = np.clip(*ranges[zone], target_angle)
                error = np.abs(target_angle - new_candidate)
                if error < 0.5:
                    return new_candidate
                elif error < best_angle_err:
                    best_angle_err = error
                    best_angle = new_candidate
        return best_angle if best_angle_err != 31 else None


    # def get_nearest_nav_angle(target_angle):
    #     max_angle = 15 if vision['near_left'] > 0.9 else vision['near_left']*0.5
    #     min_angle = -15 if vision['near_right'] > 0.9 else vision['near_right']*(-0.5)
    #     print("ANGLE RANGES: {} to {}".format(min_angle, max_angle))
    #     return np.clip(min_angle, max_angle, target_angle)

    def get_future_place(target_angle):
        if target_angle > 5:
            return 'far_left'
        elif target_angle < -5:
            return 'far_right'
        else: return 'far_center'

    def go_forward_avoiding_obstacles(target_angle, target_distance=np.inf, max_speed=2.5):

        target_speed = 0  # by default, stop
        if target_distance < RoverCam.dist_ranges[-1]:  # lower than our resolution for obstacles
            print("Reaching some object")
            target_angle += obstacle_avoiding_offset() + not_crashing_offset()
            target_speed = max_speed*target_distance/20
        elif vision['near_center'] > 0.9:
            target_angle = get_nav_angle(target_angle)
            if target_angle is not None:
                ahead_clearness = min(vision['near_center'], vision[get_future_place(target_angle)])
                target_speed = max_speed*ahead_clearness
                # The less clear we see ahead, the more offset to avoid obstacles we add
                obstacle_offset = obstacle_avoiding_offset()
                offset_scale = 0.9/ahead_clearness
                target_angle += obstacle_offset*offset_scale
                target_angle += not_crashing_offset()

        is_navigable = target_speed > 0 and target_angle is not None
        if is_navigable:
            Rover.steer = target_angle
            if Rover.vel > 1.5*target_speed:
                Rover.brake = Rover.brake_set
                Rover.throttle = 0
            elif Rover.vel < target_speed:
                Rover.brake = 0
                Rover.throttle = Rover.throttle_set
            else:  # Go loosely
                Rover.brake = 0
                Rover.throttle = 0
        return is_navigable

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        print("State: {}".format(Rover.mode))
        # First priority: if we're near sample, brake and pick it up
        if Rover.near_sample:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            # Define state for after picking up
            if Rover.vel:
                steering_counter = 0
                Rover.mode = Rover.S_SAW_ROCK  # Still moving, we might lose the rock
            else:
                steering_direction = get_next_steering(np.mean(Rover.nav_angles))
                Rover.mode = Rover.S_STOP  # Not moving, we'll pick rock and continue as S_STOP

        # Check for Rover.mode status
        elif Rover.mode == Rover.S_FORWARD:
            if Rover.seeing_rock:
                seen_rock_counter = 15
                Rover.mode = Rover.S_APPROACH_ROCK

            # Try to go forward or stop
            elif not go_forward_avoiding_obstacles(-10):
                steering_direction = get_next_steering(np.mean(Rover.nav_angles))
                Rover.mode = Rover.S_STOP

        elif Rover.mode == Rover.S_STOP:
            Rover.throttle = 0
            if Rover.vel > 0.2:  # Brake if moving!
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            elif min(vision['near_center'], vision['far_center']) < 0.9 \
                 or min(vision['near_left'], vision['near_right']) < 0.5:
                Rover.brake = 0
                Rover.steer = steering_direction
            else:
                Rover.mode = Rover.S_FORWARD

        elif Rover.mode == Rover.S_APPROACH_ROCK:

            print("Found the ROCK! Approaching...")
            # We lost the rock for some consecutive frames, stop and find it
            if not Rover.seeing_rock:
                seen_rock_counter -= 1

                # If we reach counter limit or can't go forward, go to rock recovery mode
                if seen_rock_counter <= 0 \
                   or not go_forward_avoiding_obstacles(last_seen_rock_angle):
                    steering_counter = 0
                    Rover.mode = Rover.S_SAW_ROCK

            # Seeing rock: head towards the rock with speed inversely proportional to distance
            else:
                seen_rock_counter = 15  # reset this counter

                target_angle = mean_angle(Rover.rock_angles)
                target_distance = np.min(Rover.rock_dists)
                last_seen_rock_angle = target_angle  # keep in case we lose the rock for a moment

                # Try to go forward
                if not go_forward_avoiding_obstacles(target_angle, target_distance, 1.5):
                    steering_direction = get_next_steering(target_angle)

        elif Rover.mode == Rover.S_SAW_ROCK:  # Reset steering_counter before entering here
            print("We lost the rock, find it now!!!")
            steer_to_find_rock()
            steering_counter += 1
            if Rover.seeing_rock:
                seen_rock_counter = 15
                Rover.mode = Rover.S_APPROACH_ROCK
            elif steering_counter > 150:  # steering limit (about 5 secs turning)
                steering_direction = get_next_steering(last_seen_rock_angle)
                Rover.mode = Rover.S_STOP
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

