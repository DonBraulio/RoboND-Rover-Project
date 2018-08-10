import numpy as np

from numpy.linalg import norm
from perception import RoverCam

def mean_angle(angles):
    return np.clip(np.mean(angles * 180/np.pi), -15, 15)


initial_pos = None
steering_counter = 0
seen_rock_counter = 15
last_seen_rock_angle = 0
steering_direction = 0
lock_counter = 0
initial_yaw = 0

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    global initial_pos
    global steering_counter
    global seen_rock_counter
    global last_seen_rock_angle
    global steering_direction
    global lock_counter
    global initial_yaw

    vision = RoverCam.get_vision_indexes(Rover.nav_dists, Rover.nav_angles)
    if initial_pos is None:
        initial_pos = np.array(Rover.pos)

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

    def balance_offset(right_index, left_index, min_ratio, epsilon=1e-6):
        right_index += epsilon  # add minutiae value to avoid division by zero
        left_index += epsilon
        # check that L/R and R/L aren't below min_ratio
        if min(right_index/left_index, left_index/right_index) <= min_ratio:
            return 1 if left_index > right_index else -1
        return 0

    def avoid_crash_steering():
        return 30 * balance_offset(vision['near_right'], vision['near_left'], 0.6)

    def avoid_far_obstacle_steering(steer_val, target_angle):
        if target_angle > 0:  # prefer left
            return steer_val * balance_offset(vision['far_center'], vision['far_left'], 0.9)
        else:  # prefer right
            return steer_val * balance_offset(vision['far_right'], vision['far_center'], 0.9)

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

    # def get_future_place(target_angle):
    #     if target_angle > 5:
    #         return 'far_left'
    #     elif target_angle < -5:
    #         return 'far_right'
    #     else: return 'far_center'

    def go_forward_avoiding_obstacles(target_angle, target_distance=np.inf, max_speed=4):

        target_speed = 0  # by default, stop
        if target_distance < RoverCam.dist_ranges[-1]:  # lower than our resolution for obstacles
            target_angle += avoid_crash_steering()
            target_speed = 0.1 * target_distance if target_distance < 10 else 1.5
        elif vision['near_center'] > 0.9:
            target_angle = get_nav_angle(target_angle)
            if target_angle is not None:
                ahead_clearness = min(vision['near_center'], vision['mid_center'])
                if ahead_clearness > 0.9 and vision['far_center'] > 0.6:
                    target_speed = max_speed * (vision['far_center'] - 0.5) / 0.5
                else:
                    target_speed = 1.0  # obstacle ahead
                # The less clear we see ahead, the more offset to avoid obstacles we add
                target_angle += avoid_far_obstacle_steering(8 / ahead_clearness, target_angle)
                target_angle += avoid_crash_steering()  # avoid short range crash

        is_navigable = target_speed > 0 and target_angle is not None
        if is_navigable:
            Rover.steer = target_angle
            if Rover.vel > 1.5*target_speed:
                Rover.brake = Rover.brake_set
                Rover.throttle = 0
            elif Rover.vel < target_speed:
                Rover.brake = 0
                Rover.throttle = Rover.throttle_set * (target_speed - Rover.vel)
            else:  # Go loosely
                Rover.brake = 0
                Rover.throttle = 0
        return is_navigable

    if Rover.nav_angles is not None:
        print("State: {}".format(Rover.mode))

        # Watchdog to ensure that we're not blocked
        if np.abs(Rover.vel) < 0.1 and not Rover.picking_up:
            lock_counter += 1
            if lock_counter > 100:
                lock_counter = 0
                Rover.mode = Rover.S_UNLOCK_ROVER
        else:
            lock_counter = 0

        # First priority: if we're near sample, brake and pick it up
        if Rover.near_sample:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            # Define state for after picking up
            if Rover.vel:
                steering_counter = 0
                Rover.mode = Rover.S_RECOVER_ROCK  # Still moving, we might lose the rock
            else:
                steering_direction = get_next_steering(np.mean(Rover.nav_angles))
                Rover.mode = Rover.S_STOP  # Not moving, we'll pick rock and continue as S_STOP

        # Check for Rover.mode status
        elif Rover.mode == Rover.S_FORWARD:

            if Rover.seeing_rock:
                seen_rock_counter = 15
                Rover.mode = Rover.S_APPROACH_ROCK

            # Try to go forward or stop

            elif not go_forward_avoiding_obstacles(10):
                steering_direction = get_next_steering(np.mean(Rover.nav_angles))
                Rover.mode = Rover.S_STOP

        elif Rover.mode == Rover.S_STOP:
            Rover.throttle = 0
            if np.abs(Rover.vel) > 0.2:  # Brake if moving!
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            elif min(vision['near_center'], vision['mid_center']) < 0.9 \
                    or balance_offset(vision['far_center'], vision['far_left'], 0.8) != 0:
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
                   or not go_forward_avoiding_obstacles(last_seen_rock_angle, 2.5):
                    steering_counter = 0
                    Rover.mode = Rover.S_RECOVER_ROCK

            # Seeing rock: head towards the rock with speed inversely proportional to distance
            else:
                seen_rock_counter = 15  # reset this counter

                target_angle = mean_angle(Rover.rock_angles)
                target_distance = np.min(Rover.rock_dists)
                last_seen_rock_angle = target_angle  # keep in case we lose the rock for a moment
                # Try to go forward
                if not go_forward_avoiding_obstacles(target_angle, target_distance, 2.5):
                    steering_direction = get_next_steering(target_angle)

        elif Rover.mode == Rover.S_RECOVER_ROCK:
            print("We lost the rock, find it now!!!")
            steer_to_find_rock()
            steering_counter += 1  # reset this counter before entering the state
            if Rover.seeing_rock:
                if Rover.steer < 2:  # keep steering until the rock is in the center (< 2 deg)
                    seen_rock_counter = 15
                    Rover.mode = Rover.S_APPROACH_ROCK
            elif steering_counter > 150:  # steering limit (about 5 secs turning)
                steering_direction = get_next_steering(last_seen_rock_angle)
                Rover.mode = Rover.S_STOP

        elif Rover.mode == Rover.S_UNLOCK_ROVER:
            lock_counter += 1
            if lock_counter < 100:  # Try reverse to unlock
                initial_yaw = Rover.yaw
                Rover.throttle = -1
                if np.abs(Rover.vel) > 0.1:
                    Rover.steer = -get_next_steering(np.mean(Rover.nav_angles))
                    if np.abs(Rover.vel) > 0.3:
                        Rover.mode = Rover.S_STOP
            elif lock_counter < 400:  # then try steering 90 deg
                diff = Rover.yaw - initial_yaw
                if diff < 0:
                    diff += 360
                if diff < 90:
                    Rover.steer = 15
                else:
                    Rover.mode = Rover.S_STOP
            else:
                lock_counter = 0  # retry reverse

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

