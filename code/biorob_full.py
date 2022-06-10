# Modules
import pybullet as pb
import pybullet_data
import numpy as np
import time
from utils import *


# Class
class BioRob_Full:
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
        self.m = 0.721791502953507 + 4 * (0.0446567870646305 + 0.00360792918860103 + 0.0310372891847717 +
                                          0.0364619323149185 + 0.0120009876791377)
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
        self.robot_id = pb.loadURDF(fileName='./models/biorob_full/urdf/biorob_full.urdf',
                                    basePosition=self.robot_start_pos,
                                    baseOrientation=self.robot_start_orn,
                                    useFixedBase=self.flag_fixed_base,
                                    flags=pb.URDF_USE_INERTIA_FROM_FILE & pb.URDF_MAINTAIN_LINK_ORDER)
        self.lengths = lengths; self.l_thigh = self.lengths[0]; self.l_calf = self.lengths[1]; self.l_foot1 = self.lengths[2]; self.l_foot2 = self.lengths[3]
        self.fl_hip = 0; self.fl_thigh_rod = 1; self.fl_knee = 2; self.fl_ankle = 3; self.fl_foot_toe = 4
        self.fr_hip = 5; self.fr_thigh_rod = 6; self.fr_knee = 7; self.fr_ankle = 8; self.fr_foot_toe = 9
        self.bl_hip = 10; self.bl_thigh_rod = 11; self.bl_knee = 12; self.bl_ankle = 13; self.bl_foot_toe = 14
        self.br_hip = 15; self.br_thigh_rod = 16; self.br_knee = 17; self.br_ankle = 18; self.br_foot_toe = 19
        self.base = 0
        self.fl_thigh = 0; self.fl_rod = 1; self.fl_calf = 2; self.fl_foot = 3; self.fl_toe = 4
        self.fr_thigh = 5; self.fr_rod = 6; self.fr_calf = 7; self.fr_foot = 8; self.fr_toe = 9
        self.bl_thigh = 10; self.bl_rod = 11; self.bl_calf = 12; self.bl_foot = 13; self.bl_toe = 14
        self.br_thigh = 15; self.br_rod = 16; self.br_calf = 17; self.br_foot = 18; self.br_toe = 19
        self.joints = [self.fl_hip, self.fl_thigh_rod, self.fl_knee, self.fl_ankle, self.fl_foot_toe,
                       self.fr_hip, self.fr_thigh_rod, self.fr_knee, self.fr_ankle, self.fr_foot_toe,
                       self.bl_hip, self.bl_thigh_rod, self.bl_knee, self.bl_ankle, self.bl_foot_toe,
                       self.br_hip, self.br_thigh_rod, self.br_knee, self.br_ankle, self.br_foot_toe]
        self.act_joints = [self.fl_hip, self.fl_knee, self.fr_hip, self.fr_knee,
                           self.bl_hip, self.bl_knee, self.br_hip, self.br_knee]
        self.links = [self.base,
                      self.fl_thigh, self.fl_rod, self.fl_calf, self.fl_foot, self.fl_toe,
                      self.fr_thigh, self.fr_rod, self.fr_calf, self.fr_foot, self.fr_toe,
                      self.bl_thigh, self.bl_rod, self.bl_calf, self.bl_foot, self.bl_toe,
                      self.br_thigh, self.br_rod, self.br_calf, self.br_foot, self.br_toe]
        self.nb_joints = len(self.joints); self.nb_act_joints = len(self.act_joints)
        # --- add constraints --- #
        self.cst1 = pb.createConstraint(parentBodyUniqueId=self.robot_id,
                                        parentLinkIndex=self.fl_rod,
                                        childBodyUniqueId=self.robot_id,
                                        childLinkIndex=self.fl_foot,
                                        jointType=pb.JOINT_POINT2POINT,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0, self.l_calf/2, 0],
                                        childFramePosition=[0, -0.09079, -0.01014])
        self.cst2 = pb.createConstraint(parentBodyUniqueId=self.robot_id,
                                        parentLinkIndex=self.fr_rod,
                                        childBodyUniqueId=self.robot_id,
                                        childLinkIndex=self.fr_foot,
                                        jointType=pb.JOINT_POINT2POINT,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0, self.l_calf/2, 0],
                                        childFramePosition=[0, -0.09079, -0.01014])
        self.cst3 = pb.createConstraint(parentBodyUniqueId=self.robot_id,
                                        parentLinkIndex=self.bl_rod,
                                        childBodyUniqueId=self.robot_id,
                                        childLinkIndex=self.bl_foot,
                                        jointType=pb.JOINT_POINT2POINT,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0, self.l_calf/2, 0],
                                        childFramePosition=[0, -0.09079, -0.01014])
        self.cst4 = pb.createConstraint(parentBodyUniqueId=self.robot_id,
                                        parentLinkIndex=self.br_rod,
                                        childBodyUniqueId=self.robot_id,
                                        childLinkIndex=self.br_foot,
                                        jointType=pb.JOINT_POINT2POINT,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0, self.l_calf/2, 0],
                                        childFramePosition=[0, -0.09079, -0.01014])

    def extract_joint_states(self):
        fl_hip = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.fl_hip)
        fl_knee = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.fl_knee)
        fl_ankle = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.fl_ankle)
        fr_hip = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.fr_hip)
        fr_knee = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.fr_knee)
        fr_ankle = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.fr_ankle)
        bl_hip = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.bl_hip)
        bl_knee = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.bl_knee)
        bl_ankle = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.bl_ankle)
        br_hip = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.br_hip)
        br_knee = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.br_knee)
        br_ankle = pb.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.br_ankle)
        return fl_hip, fl_knee, fl_ankle, fr_hip, fr_knee, fr_ankle, bl_hip, bl_knee, bl_ankle, br_hip, br_knee, br_ankle

    def apply_torques(self, torques):
        if not self.flag_apply_zero_torques:
            if self.flag_actuate_all_joints:
                pb.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                             jointIndices=self.joints,
                                             controlMode=pb.TORQUE_CONTROL,
                                             forces=[torques[0].item(), 0.0, torques[1].item(), 0.0, torques[2].item(),
                                                     torques[3].item(), 0.0, torques[4].item(), 0.0, torques[5].item(),
                                                     torques[6].item(), 0.0, torques[7].item(), 0.0, torques[8].item(),
                                                     torques[9].item(), 0.0, torques[10].item(), 0.0, torques[11].item()])
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


# Function
def run_simulation(flag_robot_model, flag_robot_state,
                   flag_fixed_base, flag_change_robot_dynamics,
                   flag_init_config, flag_kin_model, flag_contact_dir,
                   flag_control_mode, flag_clamp_x_force, flag_actuate_all_joints,
                   flag_apply_zero_torques, flag_test, jump):
    # --- probably constant parameters for all cases --- #
    t_step = 1e-3; t_init = 2; nb_periods = 8
    g = 9.81; torque_sat = 2.7
    mu_plane = 0.7; rest_plane = 0.5
    flag_end_init = False
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
    t_stance = 0.100; t_swing = 0.300
    robot_start_pos = [0, 0, 0.32+0.1]
    robot_start_orn = pb.getQuaternionFromEuler([0, 0, 0])

    x_d = -0.02; vx_d = 0.0; z_d = 0.35; vz_d = 0
    kp_x = 60; kd_x = 3; kp_z = 100; kd_z = 10

    robot = BioRob_Full(lengths=[0.160, 0.143, 0.0639, 0.115],
                        mu_robot=1, rest_robot=0.5,
                        x_d=x_d, vx_d=vx_d, z_d=z_d, vz_d=vz_d,
                        kp_x=kp_x, kd_x=kd_x, kp_z=kp_z, kd_z=kd_z,
                        robot_start_pos=robot_start_pos, robot_start_orn=robot_start_orn,
                        flag_robot_model=flag_robot_model, flag_fixed_base=flag_fixed_base,
                        flag_kin_model=flag_kin_model, flag_control_mode=flag_control_mode,
                        flag_clamp_x_force=flag_clamp_x_force, flag_actuate_all_joints=flag_actuate_all_joints,
                        flag_apply_zero_torques=flag_apply_zero_torques)
    init_angles = (np.pi/12) * np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    k = 2; flag_vertical = 0
    # --- time initialization --- #
    time_array = np.arange(0, t_init + nb_periods * (t_stance + t_swing), t_step)
    time_len = len(time_array)
    # --- derived parameters --- #
    t_air = (t_swing - t_stance) / 2.
    t_stride = t_stance + t_swing
    t_alpha = t_stance / 2.
    c = 0.7
    alpha_z = (robot.m * g * t_stride) / (k * c * t_stance)
    # --- variables initialization --- #
    torques_axis = np.zeros((8, time_len))
    fl_tot_axis = np.zeros((2, time_len)); fl_cnt_axis = np.zeros((2, time_len)); fl_fb_axis = np.zeros((2, time_len))
    fr_tot_axis = np.zeros((2, time_len)); fr_cnt_axis = np.zeros((2, time_len)); fr_fb_axis = np.zeros((2, time_len))
    bl_tot_axis = np.zeros((2, time_len)); bl_cnt_axis = np.zeros((2, time_len)); bl_fb_axis = np.zeros((2, time_len))
    br_tot_axis = np.zeros((2, time_len)); br_cnt_axis = np.zeros((2, time_len)); br_fb_axis = np.zeros((2, time_len))
    com_axis = np.zeros((6, time_len))
    cf_z_axis = np.zeros((2, time_len)); cf_x_axis = np.zeros((2, time_len))
    fl_fb_x_axis = np.zeros((2, time_len)); fl_fb_z_axis = np.zeros((2, time_len))
    fr_fb_x_axis = np.zeros((2, time_len)); fr_fb_z_axis = np.zeros((2, time_len))
    bl_fb_x_axis = np.zeros((2, time_len)); bl_fb_z_axis = np.zeros((2, time_len))
    br_fb_x_axis = np.zeros((2, time_len)); br_fb_z_axis = np.zeros((2, time_len))
    curr_period = 0; count_period = np.zeros(time_len)

    applied_torques = np.zeros((time_len, 12))
    spring_torques = np.zeros((time_len, 8))
    motor_torques = np.zeros((time_len, 8))

    # --- enable torque sensors --- #
    enable_torque_sensors(robot)
    # --- change robot dynamics --- #
    if flag_change_robot_dynamics:
        change_dynamics(robot)
        # pb.stepSimulation()
    # --- initial configuration --- #
    if flag_init_config == 0:
        apply_config_poc(robot, init_angles, torque_sat)
        # pb.stepSimulation()
    elif flag_init_config == 1:
        apply_config_rjs(robot, init_angles)
        # pb.stepSimulation()
    else:
        pass
    # --- disable prismatic world joint --- #
    disable_prismatic_control(robot, flag_vertical)
    # pb.stepSimulation()
    # --- print system states before main loop --- #
    print_states(robot, plane_id)
    # --- test --- #
    if flag_test:
        for j in range(100000):
            pb.stepSimulation()
            time.sleep(t_step)
        pb.disconnect()
        return
    # --- set contact force direction --- #
    if flag_contact_dir == 0:
        cf_x_dir = 1; cf_z_dir = 1
    else:
        cf_x_dir = 1; cf_z_dir = -1
    # --- simulation steps --- #

    state = 'jumping'
    prev_height = pb.getLinkState(bodyUniqueId=robot.robot_id, linkIndex=robot.base)[4][2]
    max_height = 0
    for i in range(time_len):
        # --- counting the period --- #
        count_period[i] = curr_period
        if time_array[i] < t_init:
            pass
        elif time_array[i] == t_init:
            # --- disable default velocity control in all joints --- #
            disable_default_control(robot)
            flag_end_init = True
            curr_period = curr_period + 1
        elif (time_array[i]-t_init)/curr_period >= t_stance+t_swing:
            flag_end_init = True
            curr_period = curr_period + 1
        
        current_height = pb.getLinkState(bodyUniqueId=robot.robot_id, linkIndex=robot.base)[4][2]
        if time_array[i] < t_init + 2 and jump:
            robot.z_d = 0.25
        if time_array[i] >= t_init + 2 and jump:
            robot.x_d = -0.04; robot.vx_d = 0.0; robot.z_d = 0.4; robot.vz_d = 1
            kp_x_b = 60
            kd_x_b = 3
            kp_z_b = 20
            kd_z_b = 2
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
                if current_height >= 0.45:
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
        if jump:
            k = 600
        else:
            k = 150
        # --- extract the coordinates, generate contact forces and required torques --- #
        for j in range(4): # loop for each leg
            hip = pb.getJointState(bodyUniqueId=robot.robot_id, jointIndex=0+5*j)
            knee = pb.getJointState(bodyUniqueId=robot.robot_id, jointIndex=2+5*j)
            heel = pb.getJointState(bodyUniqueId=robot.robot_id, jointIndex=3+5*j)
            toe = pb.getJointState(bodyUniqueId=robot.robot_id, jointIndex=4+5*j)

            # --- Spring torques ---
            sign = -np.sign((j%2)-0.5)
            toe_torque, knee_torque = torques_heel_tendon(-toe[0]*sign, knee[0] + 1.0 , k, 0.02, 0.04, 0)
            spring_torques[i, 2*j:2*j+2] = [-knee_torque, toe_torque*sign]
            # --- Motor torques ---
            x, vx, z, vz, jac = calc_kinematics_org(robot, np.array([hip[0], knee[0], heel[0]]), np.array([hip[1], knee[1], heel[1]]))
            forces, _, _ = impedance_control(robot, x, vx, z, vz)
            # forces[1]=0
            # --- transform force command into torque command --- 
            motor_torques[i, 2*j:2*j+2] = np.matmul(np.transpose(jac), forces).reshape(2,)
            #  --- saturate the motor torques --- 
            motor_torques[i, 2*j:2*j+2] = motor_torques[i, 2*j:2*j+2].clip(-torque_sat, torque_sat)

            applied_torques[i, 3*j+1:3*j+3] = spring_torques[i, 2*j:2*j+2]
            applied_torques[i, 3*j:3*j+2] += motor_torques[i, 2*j:2*j+2]

        

        # --- after t_init finishes, apply the calculated torques --- #
        if flag_end_init:
            robot.apply_torques(applied_torques[i])
        # --- next step --- #
        pb.stepSimulation()
        # time.sleep(t_step)
        prev_height = current_height
        max_height = max(max_height, current_height)

    # --- end of simulation --- #
    pb.disconnect()

    print(f"max height: {max_height}")

    if jump:
        window1 = 4500
        window2 = 12000
        plt.figure()
        plt.plot(spring_torques[window1:window2,0], label='knee spring')
        plt.plot(motor_torques[window1:window2,1], label='knee motor')
        plt.plot(applied_torques[window1:window2,1],label='knee total')
        # plt.legend(loc='right')
        plt.legend(loc='lower right')
        plt.xlabel('timestep')
        plt.ylabel('Torque [Nm]')
        # plt.show()
        TsavedKnee = (1- (np.mean(np.abs(motor_torques[4000:,1])) / np.mean(np.abs(applied_torques[4000:,1]))))* 100
        print(f'saved: {TsavedKnee}')
    else:
        window1 = 2150
        window2 = 12000
        plt.figure()
        plt.plot(spring_torques[window1:window2,0], label='knee spring')
        plt.plot(motor_torques[window1:window2,1], label='knee motor')
        plt.plot(applied_torques[window1:window2,1],label='knee total')
        # plt.legend(loc='right')
        plt.legend(loc='lower right')
        plt.xlabel('timestep')
        plt.ylabel('Torque [Nm]')
        plt.title(f'Percentage of weight supported by spring at knee joint: {(1-abs(motor_torques[-1,1]/applied_torques[-1,1]))*100:.2f}%')
        # plt.show()
    return

def torques_heel_tendon(toe_angle, knee_angle, k=1925, toe_radius=0.0185, knee_radius=0.0185, flag_block_toe=0):
    displacement = toe_angle * toe_radius + knee_angle * knee_radius
    tension = k * displacement
    toe_torque = toe_radius * tension
    knee_torque = knee_radius * tension
    knee_torque, toe_torque = max(knee_torque, 0.0), max(toe_torque, 0.0) # tendon can be stretched but not contracter
    if not flag_block_toe:
        toe_torque += 0.5 * toe_angle
    return toe_torque, knee_torque

# main
if __name__ == '__main__':
    run_simulation(flag_robot_model=4,                  # 4: biorob full
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
                   jump=0)                              # 1: jump, 0: stand at h=24cm   