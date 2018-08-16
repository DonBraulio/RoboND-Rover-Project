import numpy as np

steering_counter = 0
steering_stopped = False
seen_rock_counter = 0
lost_rock_counter = 0
last_seen_rock = 0
last_steering = 0
rock_seeking_counter = 0
locked_counter = 0
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    global rock_seeking_counter
    global steering_stopped
    global seen_rock_counter
    global lost_rock_counter
    global last_seen_rock
    global last_steering
    global locked_counter

    def mean_angle(angles):
        return np.clip(np.mean(angles * 180/np.pi), -15, 15)

    def get_next_steering():
        return 15 if mean_angle(Rover.nav_angles) >= 0 else -15

    def steer_to_find_rock():
        Rover.throttle = 0
        if Rover.vel:
            Rover.brake = Rover.brake_set
            Rover.steer = 0  # dont't steer, we might lose the rock
        elif Rover.seeing_rock:
            print("BRAKE: found the rock" )
            Rover.brake = 0
            Rover.steer = mean_angle(Rover.nav_angles)
        else:
            print("Steering to find rock")
            Rover.brake = 0
            Rover.steer = 15 if last_seen_rock >= 0 else -15

    def get_visibility_factor():
        angle_limit = 5*np.pi/180
        angles_mask = np.abs(Rover.nav_angles) < angle_limit
        view_dist = np.mean(Rover.nav_dists[angles_mask]) 
        visibility_factor = 0 if np.isnan(view_dist) else view_dist / (Rover.max_view_distance / 2)
        print("VISIBILITY: {}".format(visibility_factor))
        return visibility_factor

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        print("State: {}".format(Rover.mode))
        if (np.abs(Rover.vel) < 0.2 and not Rover.picking_up):
            locked_counter += 1
        elif np.abs(Rover.vel) > 0.5:
            locked_counter = 0

        if locked_counter > 40:
            Rover.brake = 0
            if not locked_counter % 10:
                Rover.steer = -15 if Rover.steer < 0 else 15
            if not locked_counter % 5:
                Rover.throttle = -5 if Rover.throttle < 0 else 5
            print("UNLOCKING!")
            return Rover

        # First priority: if we're near sample, brake and pick it up
        if Rover.near_sample:
            print("BRAKE! NEAR SAMPLE")
            Rover.throttle = 0
            Rover.brake = Rover.brake_set

        else:

            visibility_factor = get_visibility_factor()

            if Rover.seeing_rock:
                seen_rock_counter += 1
                last_seen_rock = np.mean(Rover.rock_angles)
                rock_seeking_counter = 15

            if seen_rock_counter > 3:
                go_straight_margin = (5 * np.pi / 180)
                rock_dist = np.min(Rover.rock_dists)
                print("ROCK DISTANCE: {}".format(rock_dist))
                if np.abs(last_seen_rock) < go_straight_margin or rock_dist > 20:
                    rock_dist_factor = rock_dist / (Rover.max_view_distance / 2)
                    rock_dist_factor = np.clip( 0.2, 1, rock_dist_factor )
                    target_angle = last_seen_rock
                    target_speed = 1.5 * rock_dist_factor + 1
                else:
                    target_angle = 15 if last_seen_rock > 0 else -15
                    target_speed = 0

                # Ensure that we aren't in this state forever
                if rock_seeking_counter:
                    rock_seeking_counter -= 1
                else:
                    seen_rock_counter = 0  # we lost the rock, let it go man
            else:
                nav_angles = Rover.nav_angles
                closed_boundary = len(Rover.obs_dists) and len(Rover.nav_dists)\
                                  and np.max(Rover.nav_dists) < (np.max(Rover.obs_dists) * 0.6)
                if not closed_boundary and not steering_stopped and len(nav_angles) >= Rover.stop_forward:
                    target_speed = visibility_factor * 2
                    target_angle = mean_angle(nav_angles) + 5  # left wall follower
                    last_steering = 1 if target_angle >= 0 else -1
                else:
                    print("STEERING STOPPED")
                    steering_stopped = len(Rover.nav_angles) < Rover.go_forward
                    target_angle = -15  # steer right
                    target_speed = 0

            Rover.steer = target_angle

            current_speed = np.abs(Rover.vel)
            if current_speed > 0.2 and current_speed > 2 * target_speed:
                Rover.brake = Rover.brake_set
                Rover.throttle = 0
            elif target_speed == 0:  # Steer around
                Rover.brake = 0
                Rover.throttle = 0
            else:
                # Set throttle value to throttle setting
                Rover.throttle = Rover.throttle_set * (target_speed - Rover.vel) / target_speed
                Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

