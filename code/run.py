import biorob_single # contains signle leg simulation
import biorob_full   # contains full quadruped simulation

simulation = 'quadruped'       # choose between 'single leg' and 'quadruped'

if simulation == 'single leg':
    # --- Set to 1 for real-time simulation, 0 for sped-up simulation ---
    sleep_time = 0
    # --- print the graph with torques at the end of simulation ---
    print_graph = 0
    # --- choose the spring configuration. 1-3 are normal configs, 4 is for testing the toe retraction ---
    config = 1
    # --- add a delay to the controller (must be an int) ---
    delay = 0
    # --- choose length of simulation in seconds ---
    length = 10

    if config==4:
        fix_base = 1
    else:
        fix_base = 0

    biorob_single.run_simulation(flag_robot_model=3,     
                        flag_robot_state=0,           
                        flag_fixed_base=fix_base,           
                        flag_change_robot_dynamics=1,        
                        flag_init_config=2,                  
                        flag_kin_model=0,                    
                        flag_contact_dir=0,                  
                        flag_control_mode=1,                 
                        flag_clamp_x_force=0,                
                        flag_actuate_all_joints=1,           
                        flag_apply_zero_torques=0,           
                        flag_test=0,                         
                        flag_sleep_time=sleep_time,
                        flag_block_toe=0,
                        print_graph=print_graph,
                        spring_config=config, 
                        flag_record=0,
                        delay=delay,
                        length=length)
else:
    # --- Choose between standing simulation (0) or jumping (1)
    jump = 1

    biorob_full.run_simulation(flag_robot_model=4,                  # 4: biorob full
                               flag_robot_state=0,                  # 0: hopping, 1: bounding in place, 2: bounding forward
                               flag_fixed_base=0,                   # 1: fix the base on air or not (0)
                               flag_change_robot_dynamics=1,        # 1: change dynamics of the robot or not (0)
                               flag_init_config=2,                  # 0: position control - 1: resetJointState - anything else: do not apply the initial configuration
                               flag_kin_model=0,                    # 0: without equivalent leg - 1: with equivalent leg - anything else: error because we cannot calculate the forward kinematics
                               flag_contact_dir=0,                  # 0: push downward - 1: pull upward (change the direction of the applied contact force)
                               flag_control_mode=1,                 # 0: pd cartesian test - 1: full simulation with contact force
                               flag_clamp_x_force=0,                # 1: clamp the x force to mu * Fz or not (0)
                               flag_actuate_all_joints=1,           # 1: apply hip and knee torques and 0 torque on the rest of the joints - 0: apply only hip and knee torques
                               flag_apply_zero_torques=0,           # 1: apply zero torques on the joints or not (0)
                               flag_test=0,                         # 1: terminate simulation after initial configuration (before the main loop) or not (0)
                               jump=jump)                              # 1: jump, 0: stand at h=24cm   