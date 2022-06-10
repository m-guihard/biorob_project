# Modules
from multiprocessing.dummy import current_process
import pybullet as pb
import pybullet_data
import numpy as np
import time
from utils import *
import matplotlib.pyplot as plt


# Class
class BioRob_Single:
    def __init__(self, lengths,
                 mu_robot, rest_robot,
                 x_d, vx_d, z_d, vz_d,
                 kp_x, kd_x, kp_z, kd_z,
                 robot_start_pos, robot_start_orn,
                 flag_robot_model, flag_fixed_base, flag_kin_model, flag_control_mode,
                 flag_clamp_x_force, flag_actuate_all_joints, flag_apply_zero_torques):
        # --- general parameters --- #
        self.mu_robot = mu_robot; self.rest_robot = rest_robot
        self.robot_start_pos = robot_start_pos; self.robot_start_orn = robot_start_orn
        self.m = 0.044641 + 0.044644 + 0.003608 + 0.031037 + 0.036462 + 0.012001
        # --- control parameters --- #
        self.x_d = x_d; self.vx_d = vx_d; self.z_d = z_d; self.vz_d = vz_d
        self.kp_x = kp_x; self.kd_x = kd_x; self.kp_z = kp_z; self.kd_z = kd_z
        # --- flags --- #
        self.flag_robot_model = flag_robot_model
        self.flag_fixed_base = flag_fixed_base
        self.flag_kin_model = flag_kin_model
        self.flag_control_mode = flag_control_mode
        self.flag_clamp_x_force = flag_clamp_x_force
        self.flag_actuate_all_joints = flag_actuate_all_joints
        self.flag_apply_zero_torques = flag_apply_zero_torques
        # --- robot parameters --- #
        self.robot_id = pb.loadURDF(fileName='./models/biorob_single/urdf/biorob_single.urdf',
                                    basePosition=self.robot_start_pos,
                                    baseOrientation=self.robot_start_orn,
                                    useFixedBase=self.flag_fixed_base,
                                    flags=pb.URDF_USE_INERTIA_FROM_FILE & pb.URDF_MAINTAIN_LINK_ORDER)
        self.lengths = lengths; self.l_thigh = self.lengths[0]; self.l_calf = self.lengths[1]; self.l_foot1 = self.lengths[2]; self.l_foot2 = self.lengths[3]
        self.vertical = 0; self.hip = 1; self.thigh_rod = 2; self.knee = 3; self.heel = 4; self.foot_toe = 5
        self.world = -1; self.base = 0; self.thigh = 1; self.rod = 2; self.calf = 3; self.foot = 4; self.toe = 5
        self.joints = [self.hip, self.thigh_rod, self.knee, self.heel, self.foot_toe]
        self.act_joints = [self.hip, self.knee]
        self.links = [self.base, self.thigh, self.rod, self.calf, self.foot, self.toe]
        self.nb_joints = len(self.joints); self.nb_act_joints = len(self.act_joints)
        # --- add constraints --- #
        self.cst = pb.createConstraint(parentBodyUniqueId=self.robot_id,
                                       parentLinkIndex=self.rod,
                                       childBodyUniqueId=self.robot_id,
                                       childLinkIndex=self.foot,
                                       jointType=pb.JOINT_POINT2POINT,
                                       jointAxis=[0, 0, 0],
                                       parentFramePosition=[0, self.l_calf/2, 0],
                                       childFramePosition=[0, -0.09079, -0.01014])

    def extract_joint_states(self):
        hip = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.hip)
        knee = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.knee)
        heel = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.heel)
        return hip, knee, heel

    def apply_torques(self, torques):
        if not self.flag_apply_zero_torques:
            if self.flag_actuate_all_joints:
                pb.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                             jointIndices=self.joints,
                                             controlMode=pb.TORQUE_CONTROL,
                                             forces=[torques[0].item(), 0.0, torques[1].item(), 0.0, torques[2].item()/3])
                                            #  forces=[torques[0].item(), 0.0, torques[1].item(), 0.0, 0.05])
            else:
                pb.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                             jointIndices=self.act_joints,
                                             controlMode=pb.TORQUE_CONTROL,
                                             forces=torques)
        else:
            if self.flag_actuate_all_joints:
                pb.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                             jointIndices=self.joints,
                                             controlMode=pb.TORQUE_CONTROL,
                                             forces=np.zeros(self.nb_joints))
            else:
                pb.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                             jointIndices=self.act_joints,
                                             controlMode=pb.TORQUE_CONTROL,
                                             forces=np.zeros(self.nb_act_joints))

    def get_end_position(self):
        com = pb.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.toe)
        com_x, com_z = com[0][0], com[0][2]
        dist = 0.01953
        foot_toe = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.foot_toe)
        q4 = foot_toe[0]
        real_x = com_x - dist * np.cos(q4)
        real_z = self.robot_start_pos[2] - (com_z + dist * np.sin(q4))
        return real_x, real_z


# Function
def run_simulation(flag_robot_model, flag_robot_state,
                   flag_fixed_base, flag_change_robot_dynamics,
                   flag_init_config, flag_kin_model, flag_contact_dir,
                   flag_control_mode, flag_clamp_x_force, flag_actuate_all_joints,
                   flag_apply_zero_torques, flag_test, flag_sleep_time, flag_block_toe, 
                   print_graph, spring_config, flag_record, delay, length):
    # --- constant parameters for all cases --- #
    t_step = 1e-3; t_init = 20
    g = 9.81; torque_sat = 2.7
    mu_plane = 1; rest_plane = 0.5

    # --- setup pybullet --- #
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -g)
    pb.setPhysicsEngineParameter(fixedTimeStep=t_step,
                                 enableFileCaching=0,
                                 numSubSteps=1)
    pb.resetDebugVisualizerCamera(cameraDistance=0.4,
                                  cameraYaw=180,
                                  cameraPitch=0,
                                  cameraTargetPosition=[0, 0.4, 0.4])
    plane_id = pb.loadURDF(fileName='plane.urdf')
    pb.changeDynamics(bodyUniqueId=plane_id,
                      linkIndex=-1,
                      lateralFriction=mu_plane,
                      restitution=rest_plane)
    # --- robot model --- #
    if spring_config == 4: # mode 4 is suspended in the air for testing
        robot_start_pos = [0, 0, 0.5]
    else:
        robot_start_pos = [0, 0, 0.32]

    robot_start_orn = pb.getQuaternionFromEuler([0, 0, 0])
    x_d = -0.025; vx_d = 0.0; z_d = 0.3916; vz_d = 4.2

    if delay:
        x_d = 0
    
    # depending on the spring config, use appropriate gains
    if spring_config == 1:
        kp_x_b = 60
        kd_x_b = 3
        kp_z_b = 500
        kd_z_b = 100
    elif spring_config == 2:
        kp_x_b = 60
        kd_x_b = 3
        kp_z_b = 400
        kd_z_b = 10
        x_d = 0
    elif spring_config == 3:
        kp_x_b = 60
        kd_x_b = 3
        kp_z_b = 400
        kd_z_b = 100
    elif spring_config == 4:
        kp_x_b = 0
        kd_x_b = 0
        kp_z_b = 0
        kd_z_b = 0
    

    # --- instantiate the robot model --- #
    robot = BioRob_Single(lengths=[0.160, 0.143, 0.0639, 0.115],
                          mu_robot=1, rest_robot=0.5,
                          x_d=x_d, vx_d=vx_d, z_d=z_d, vz_d=vz_d,
                          kp_x=kp_x_b, kd_x=kd_x_b, kp_z=kp_z_b, kd_z=kd_z_b,
                          robot_start_pos=robot_start_pos, robot_start_orn=robot_start_orn,
                          flag_robot_model=flag_robot_model, flag_fixed_base=flag_fixed_base,
                          flag_kin_model=flag_kin_model, flag_control_mode=flag_control_mode,
                          flag_clamp_x_force=flag_clamp_x_force,
                          flag_actuate_all_joints=flag_actuate_all_joints,
                          flag_apply_zero_torques=flag_apply_zero_torques)
                          
    init_angles = (np.pi/12) * np.ones(2)
    flag_vertical = 1

    # --- time initialization --- #
    time_array = np.arange(0, t_init + length, t_step)
    time_len = len(time_array)

    # --- variables initialization --- #

    spring_torques = np.zeros((time_len, 3))
    motor_torques = np.zeros((time_len, 2))
    applied_torques = np.zeros((time_len, 3))

    # --- enable torque sensors --- #
    enable_torque_sensors(robot)

    # --- change robot dynamics --- #
    if flag_change_robot_dynamics:
        change_dynamics(robot)

    # --- initial configuration --- #
    if flag_init_config == 0:
        apply_config_poc(robot, init_angles, torque_sat)
    elif flag_init_config == 1:
        apply_config_rjs(robot, init_angles)
    # --- disable prismatic world joint --- #

    if spring_config != 4:
        disable_prismatic_control(robot, flag_vertical)

    if delay:
        delayed_forces = []

    if flag_record:
        record = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, 'C:/Users/maikg/Desktop/Project/quadruped/quadruped/recording.mp4')
    
    # --- simulation steps --- #
    state = 'jumping'
    prev_height = pb.getLinkState(bodyUniqueId=robot.robot_id, linkIndex=robot.base)[4][2]
    max_height = 0
    for i in range(time_len):
        if time_array[i] < t_init:
            pb.stepSimulation()
            continue
        elif time_array[i] == t_init:
            # --- disable default velocity control in all joints --- #
            disable_default_control(robot)
            pb.stepSimulation()
            continue

        # --- Calculate the spring torques --- #
        q_knee = pb.getJointState(bodyUniqueId=robot.robot_id, jointIndex=robot.knee)[0]
        q_toe = pb.getJointState(bodyUniqueId=robot.robot_id, jointIndex=robot.toe)[0]
        if spring_config == 1:
            toe_torque, knee_torque = torques_heel_tendon(-q_toe + 0.5, q_knee + 1, 1000, 0.02, 0.04, flag_block_toe)                                                                  # 1000/1500 is better
        elif spring_config == 2:
            toe_torque, knee_torque = torques_knee_tendon(-q_toe + 0.5, q_knee + 1, 1900, 0.04, flag_block_toe)
        elif spring_config == 3:
            toe_torque, knee_torque = torques_extended_tendon(-q_toe + 0.5, q_knee + 1, 400, 0.02, 0.04, flag_block_toe)
        elif spring_config == 4:
            toe_torque, knee_torque = torques_heel_tendon_test(-q_toe + 0.5, q_knee + 1, 0.02, 0.04, 0.04, 1000, 'flying', flag_block_toe)

        spring_torques[i] = [0, -knee_torque, toe_torque]

        current_height = pb.getLinkState(bodyUniqueId=robot.robot_id, linkIndex=robot.base)[4][2]
        if state == 'landing':
            # --- in landing, try to keep the foot below the hip --- 
            if current_height > prev_height:
                state = 'jumping'
            robot.kp_x = kp_x_b
            robot.kd_x = kd_x_b
            robot.kp_z = 0
            robot.kd_z = 0

        if state == 'jumping':
            # --- in jumping, push down to jump --- 
            if current_height >= 0.5:
                state = 'flying'
            robot.kp_x = kp_x_b
            robot.kd_x = kd_x_b
            robot.kp_z = kp_z_b
            robot.kd_z = kd_z_b

        elif state == 'flying':
            # --- in flying, keep the foot below the hip and the leg fully extended --- 
            if current_height < 0.4:
                state = 'landing'
            robot.kp_x = kp_x_b
            robot.kd_x = kd_x_b
            robot.kp_z = kp_z_b / 10
            robot.kd_z = 0

        # --- Calculate the motor torques --- #
        hip, knee, heel = robot.extract_joint_states()
        q = np.array([hip[0], knee[0], heel[0]])
        vq = np.array([hip[1], knee[1], heel[1]])
        if robot.flag_kin_model == 0:
            x, vx, z, vz, jac = calc_kinematics_org(robot, q, vq)
        elif robot.flag_kin_model == 1:
            x, vx, z, vz, jac = calc_kinematics_eqv(robot, q, vq)
        forces, _, _ = impedance_control(robot, x, vx, z, vz)

        # --- in case of control delay, save current command and apply it later --- 
        if delay:
            delayed_forces.append(forces)
            if len(delayed_forces) == delay + 1:
                forces = delayed_forces.pop(0)
            else:
                forces = [0, 0]
            spring_torques[i,2] = 0


        # --- transform force command into torque command --- 
        motor_torques[i] = np.matmul(np.transpose(jac), forces).reshape(2,)
        #  --- saturate the motor torques --- 
        motor_torques[i] = motor_torques[i].clip(-torque_sat, torque_sat)


        # ------------ Add motor and spring torques ---------------
        applied_torques[i] = spring_torques[i]
        applied_torques[i,0:2] += motor_torques[i] # index 2 is the toe, which has no motor

        # ------------ Apply total torque ---------------
        robot.apply_torques(applied_torques[i])

        # --- save current height for mode switching next cycle --- 
        prev_height = current_height

        # --- next step --- #
        pb.stepSimulation()

        # --- do the simulation in real-time or sped-up --- 
        if flag_sleep_time:
            time.sleep(t_step)

        # --- check if a new max height is reached --- 
        max_height = max(max_height, current_height)


    # --- end of simulation --- #
    if flag_record:
        pb.stopStateLogging(record)
    pb.disconnect()

    # --- plot the torques from the last 3s (most stable) of simulation --- 
    window1 = len(spring_torques)-3000
    window2 = len(spring_torques)
    plt.figure()
    # plt.subplot(3,1,spring_config)
    plt.plot(spring_torques[window1:window2, 1], label="knee spring torque", linewidth=1)
    plt.plot(motor_torques[window1:window2,1], label="knee motor torque", linewidth=1)
    plt.plot(applied_torques[window1:window2,1], label="total torque", linewidth=1)
    # plt.plot(applied_torques[window1:window2,0], label="hip motor", linewidth=1)
    # plt.plot(spring_torques[window1:window2, 2], label="toe spring", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel('timestep')
    plt.ylabel('Torque [Nm]')
    plt.savefig(f"S{spring_config}_D{delay}.png")
    # plt.savefig(f"torqueC.png")
    # plt.savefig(f"3_torque.png")
    if print_graph:
        plt.show()

    applied_torques[applied_torques==0] = 1e-6 # prevent division problems (should not happen)
    # --- torque saved at the knee motor --- 
    TsavedKnee = (1- (np.mean(np.abs(motor_torques[:,1])) / np.mean(np.abs(applied_torques[:,1]))))* 100
    # --- total torque saved (knee + hip)
    TsavedTot = (1- (np.mean(np.abs(motor_torques[:,0])+np.abs(motor_torques[:,1])) / np.mean(np.abs(applied_torques[:,0])+np.abs(applied_torques[:,1]))))* 100
    # print(f'Config {spring_config}: Torque saved: {np.mean(np.abs(spring_torques[:,1]) / sum_abs_torques) * 100:.3f}%')
    print(f'\n####### RESULTS #######\nSpring config: {spring_config}\nTsaved: {TsavedKnee:.3f}% / {TsavedTot:.3f}%\nMax height reached: {max_height:.2f} m\n#######################')

    return max_height
 


def torques_heel_tendon(toe_angle, knee_angle, k=1925, toe_radius=0.0185, knee_radius=0.0185, flag_block_toe=0):
    displacement = toe_angle * toe_radius + knee_angle * knee_radius
    tension = k * displacement
    toe_torque = toe_radius * tension
    knee_torque = knee_radius * tension
    knee_torque, toe_torque = max(knee_torque, 0), max(toe_torque, 0) # tendon can be stretched but not contracter
    if not flag_block_toe:
        # toe_torque += min(0.5 * toe_angle, 0)
        toe_torque += 0.5 * toe_angle
    return toe_torque, knee_torque

def torques_knee_tendon(toe_angle, knee_angle, k=9527, knee_radius=0.0185, flag_block_toe=0):
    displacement = knee_angle * knee_radius
    tension = k * displacement
    knee_torque = knee_radius * tension
    knee_torque = max(knee_torque, 0)
    if not flag_block_toe:
        toe_torque = 0.5 * toe_angle
    else:
        toe_torque = 0
    return toe_torque, knee_torque

def torques_extended_tendon(toe_angle, knee_angle, k=1745, toe_radius=0.0185, knee_radius=0.0185, flag_block_toe=0):
    displacement = toe_angle * toe_radius + knee_angle * knee_radius * 2
    tension = k * displacement
    toe_torque = toe_radius * tension
    knee_torque = knee_radius * tension * 2
    knee_torque, toe_torque = max(knee_torque, 0), max(toe_torque, 0)
    if not flag_block_toe:
        toe_torque += 0.5 * toe_angle
    return toe_torque, knee_torque

def torques_heel_tendon_test(toe_angle, knee_angle, r_toe, r_knee1, r_knee2, k, mode, flag_block_toe):
    displacement = toe_angle * (r_toe / r_knee1 * r_knee2) + knee_angle * r_knee2
    F1 = displacement * k
    F2 = F1 * r_knee2 / r_knee1
    if mode == "flying":
        knee_torque = 0
        toe_torque = F2 * r_toe
    else:
        knee_torque = F1 * r_knee2
        toe_torque = F2 * r_toe
    knee_torque, toe_torque = max(knee_torque, 0), max(toe_torque, 0)
    if not flag_block_toe:
        toe_torque += 0.5 * toe_angle
    return [toe_torque, knee_torque]

# main
if __name__ == '__main__':
    # for config in [1, 2, 3]:
    for config in [1]:
        # for delay in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        for delay in [0]:
            height = run_simulation(flag_robot_model=3,      # 3: biorob single without toe
                        flag_robot_state=0,                  # 0: hopping (only option for single leg)
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
                        flag_sleep_time=0,
                        flag_block_toe=0,
                        print_graph=0,
                        spring_config=config, 
                        flag_record=1,
                        delay=delay,
                        length=50)