import numpy as np

steering_counter = 0
seen_rock_counter = 15
last_seen_rock = 0
steering_direction = 0
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    global steering_counter
    global seen_rock_counter
    global last_seen_rock
    global steering_direction

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
        visibility_factor = np.mean(Rover.nav_dists[angles_mask])/(Rover.img.shape[0]/4)
        print("Visibility: {}".format(visibility_factor))
        return visibility_factor

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        print("State: {}".format(Rover.mode))
        # First priority: if we're near sample, brake and pick it up
        if Rover.near_sample:
            print("BRAKE! NEAR SAMPLE")
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            # Define state for after picking up
            if Rover.vel:
                steering_counter = 0
                Rover.mode = Rover.S_SAW_ROCK  # Still moving, we might lose the rock
            else:
                steering_direction = get_next_steering()
                Rover.mode = Rover.S_STOP  # Not moving, we'll pick rock and continue as S_STOP
        # Otherwise, process Rover.mode state
        elif Rover.mode == Rover.S_APPROACH_ROCK:

            print("Found the ROCK! Approaching...")
            # We lost the rock for 5 consecutive frames, stop and find it
            if not Rover.seeing_rock:
                seen_rock_counter -= 1
                if seen_rock_counter <= 0:
                    steering_counter = 0
                    Rover.mode = Rover.S_SAW_ROCK

            # Seeing rock: head towards the rock with speed inversely proportional to distance
            else:
                max_approaching_speed = 1.5
                visibility_factor = get_visibility_factor()
                target_speed = max_approaching_speed*visibility_factor
                if Rover.vel > 1.2*target_speed:
                    Rover.brake = Rover.brake_set
                    Rover.throttle = 0
                elif Rover.vel < target_speed:
                    Rover.brake = 0
                    Rover.throttle = Rover.throttle_set
                else:  # Go loosely
                    Rover.brake = 0
                    Rover.throttle = 0
                last_seen_rock = mean_angle(Rover.nav_angles)
                seen_rock_counter = 15
                Rover.steer = last_seen_rock
        elif Rover.mode == Rover.S_SAW_ROCK:  # Reset steering_counter before entering here
            print("We lost the rock, find it now!!!")
            steer_to_find_rock()
            steering_counter += 1
            if Rover.seeing_rock:
                Rover.mode = Rover.S_APPROACH_ROCK
            elif steering_counter > 150:  # steering limit (about 5 secs turning)
                steering_direction = get_next_steering()
                Rover.mode = Rover.S_STOP
            
        # Check for Rover.mode status
        elif Rover.mode == Rover.S_FORWARD:
            if Rover.seeing_rock:
                Rover.mode = Rover.S_APPROACH_ROCK
            # Check the extent of navigable terrain
            elif len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                visibility_factor = get_visibility_factor()
                if visibility_factor < 0.25:
                    Rover.brake = Rover.brake_set
                    Rover.throttle = 0
                    steering_direction = get_next_steering()
                    Rover.mode = Rover.S_STOP
                if Rover.vel < visibility_factor*2.5:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average + offset to make it follow walls
                Rover.steer = mean_angle(Rover.nav_angles) + 5
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                steering_direction = get_next_steering()
                Rover.mode = Rover.S_STOP

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == Rover.S_STOP:
            # Now we're stopped and we have vision data to see if there's a path forward
            visibility_factor = get_visibility_factor()
            if Rover.seeing_rock:
                Rover.mode = Rover.S_SAW_ROCK
            elif visibility_factor > 0.4:
                Rover.mode = Rover.S_FORWARD
            # If we're in stop mode but still moving keep braking
            elif Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            else:
                Rover.throttle = 0
                # Release the brake to allow turning
                Rover.brake = 0
                # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                Rover.steer = steering_direction
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

