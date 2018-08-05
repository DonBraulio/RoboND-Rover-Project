import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    def steer_mean_angle(angles):
        return np.clip(np.mean(angles * 180/np.pi), -15, 15)

    def steer_to_find_rock():
        Rover.throttle = 0
        if Rover.vel:
            Rover.brake = Rover.brake_set
            Rover.steer = 0  # dont't steer, we might lose the rock
        elif Rover.seeing_rock:
            print("BRAKE: found the rock" )
            Rover.brake = 0
            Rover.steer = steer_mean_angle(Rover.nav_angles)
        else:
            print("Steering to find rock")
            Rover.brake = 0
            Rover.steer = -15
        return np.abs(Rover.steer)

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
                Rover.total_steer = 0
                Rover.mode = Rover.S_SAW_ROCK  # Still moving, we might lose the rock
            else:
                Rover.mode = Rover.S_STOP  # Not moving, we'll pick rock and continue as S_STOP
        # Otherwise, process Rover.mode state
        elif Rover.mode == Rover.S_APPROACH_ROCK:

            print("Found the ROCK! Approaching...")
            # We lost the rock, stop and find it
            if not Rover.seeing_rock:
                Rover.total_steer = 0
                Rover.mode = Rover.S_SAW_ROCK

            # Seeing rock: head towards the rock with speed inversely proportional to distance
            else:
                max_approaching_speed = 1.5
                target_speed = max_approaching_speed*np.min(Rover.nav_dists)/(Rover.img.shape[0]/4)
                if Rover.vel > 1.2*target_speed:
                    Rover.brake = Rover.brake_set
                    Rover.throttle = 0
                elif Rover.vel < target_speed:
                    Rover.brake = 0
                    Rover.throttle = Rover.throttle_set
                else:  # Go loosely
                    Rover.brake = 0
                    Rover.throttle = 0
                Rover.steer = steer_mean_angle(Rover.nav_angles)
        elif Rover.mode == Rover.S_SAW_ROCK:  # Reset total_steer before entering here
            print("We lost the rock, find it now!!!")
            Rover.total_steer += steer_to_find_rock()
            if Rover.seeing_rock:
                Rover.mode = Rover.S_APPROACH_ROCK
            elif np.abs(Rover.total_steer) >= 360:
                Rover.mode = Rover.S_STOP
            
        # Check for Rover.mode status
        elif Rover.mode == Rover.S_FORWARD:
            if Rover.seeing_rock:
                Rover.mode = Rover.S_APPROACH_ROCK
            # Check the extent of navigable terrain
            elif len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                angle_limit = 10*np.pi/180
                top_speed = 2.5
                angles_mask = np.abs(Rover.nav_angles) < angle_limit
                # Compare mean distance with img_height/4
                Rover.max_vel = top_speed*np.mean(Rover.nav_dists[angles_mask])/(Rover.img.shape[0]/4)
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = steer_mean_angle(Rover.nav_angles)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = Rover.S_STOP

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == Rover.S_STOP:
            if Rover.seeing_rock:
                Rover.mode = Rover.S_APPROACH_ROCK
            # If we're in stop mode but still moving keep braking
            elif Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = Rover.S_FORWARD
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

