# Modules
import pybullet as pb
import numpy as np
import matplotlib.pyplot as plt


# Functions
def enable_torque_sensors(robot):
    for j in robot.act_joints:
        pb.enableJointForceTorqueSensor(bodyUniqueId=robot.robot_id, jointIndex=j)


def change_dynamics(robot):
    for j in robot.links:
        pb.changeDynamics(bodyUniqueId=robot.robot_id,
                          linkIndex=j,
                          lateralFriction=robot.mu_robot,
                          restitution=robot.rest_robot)


def apply_config_poc(robot, angles, torque_sat):
    # pb.setJointMotorControlArray(bodyUniqueId=robot.robot_id,
    #                              jointIndices=robot.act_joints,
    #                              controlMode=pb.POSITION_CONTROL,
    #                              targetPositions=angles,
    #                              forces=torque_sat * np.ones(robot.nb_act_joints))
    for j in range(robot.nb_act_joints):
        pb.setJointMotorControl2(bodyUniqueId=robot.robot_id,
                                 jointIndex=robot.act_joints[j],
                                 controlMode=pb.POSITION_CONTROL,
                                 targetPosition=angles[j],
                                 force=torque_sat,
                                 targetVelocity=0)


def apply_config_rjs(robot, angles):
    for j in range(robot.nb_act_joints):
        pb.resetJointState(bodyUniqueId=robot.robot_id,
                           jointIndex=robot.act_joints[j],
                           targetValue=angles[j])


def print_states(robot, plane_id):
    print("******* Joint States *******")
    for j in robot.joints:
        temp = pb.getJointInfo(bodyUniqueId=robot.robot_id, jointIndex=j)
        print(f'{f"Joint Name: {temp[1]}":<28}, {f"Joint Index: {temp[0]}":<15}, Joint Damping: {temp[6]}, '
              f'Joint Friction: {temp[7]}, Joint Lower Limit: {temp[8]:.6f}, joint Upper Limit: {temp[9]:.6f}, '
              f'Joint Max Force: {temp[10]}, Joint Max Velocity: {temp[11]}')
    print("******* Plane State *******")
    temp = pb.getDynamicsInfo(bodyUniqueId=plane_id, linkIndex=-1)
    print(f'{f"Link Index: {-1}":<14}, Link Mass: {temp[0]:.6f}, Link Lateral Friction: {temp[1]}, '
          f'Link Restitution: {temp[5]}, Link Rolling Friction: {temp[6]}, Link Spinning Friction: {temp[7]}, '
          f'Contact Damping: {temp[8]}, Contact Stiffness: {temp[9]}')
    print("******* Link States *******")
    for j in robot.links:
        temp = pb.getDynamicsInfo(bodyUniqueId=robot.robot_id, linkIndex=j)
        print(f'{f"Link Index: {j}":<14}, Link Mass: {temp[0]:.6f}, Link Lateral Friction: {temp[1]}, '
              f'Link Restitution: {temp[5]}, Link Rolling Friction: {temp[6]}, Link Spinning Friction: {temp[7]}, '
              f'Contact Damping: {temp[8]}, Contact Stiffness: {temp[9]}')


def disable_prismatic_control(robot, flag_vertical):
    if flag_vertical:
        pb.setJointMotorControl2(bodyUniqueId=robot.robot_id,
                                 jointIndex=robot.vertical,
                                 controlMode=pb.VELOCITY_CONTROL,
                                 force=0.0)


def disable_default_control(robot):
    pb.setJointMotorControlArray(bodyUniqueId=robot.robot_id,
                                 jointIndices=robot.joints,
                                 controlMode=pb.VELOCITY_CONTROL,
                                 forces=np.zeros(robot.nb_joints))


def extract_base_coordinates(robot):
    if robot.base == -1:
        pos, orn = pb.getBasePositionAndOrientation(bodyUniqueId=robot.robot_id)
        x_com, z_com = pos[0], pos[2]
        orn_euler = pb.getEulerFromQuaternion(orn)
        thy_com = orn_euler[1]
        lin_vel, ang_vel = pb.getBaseVelocity(bodyUniqueId=robot.robot_id)
        vx_com, vz_com, wy_com = lin_vel[0], lin_vel[2], ang_vel[1]
        return x_com, z_com, thy_com, vx_com, vz_com, wy_com
    else:
        pos, orn, _, _, _, _, lin_vel, ang_vel = \
            pb.getLinkState(bodyUniqueId=robot.robot_id, linkIndex=robot.base, computeLinkVelocity=1)
        x_com, z_com = pos[0], pos[2]
        orn_euler = pb.getEulerFromQuaternion(orn)
        thy_com = orn_euler[1]
        vx_com, vz_com, wy_com = lin_vel[0], lin_vel[2], ang_vel[1]
        return x_com, z_com, thy_com, vx_com, vz_com, wy_com


def calc_kinematics(robot, q, vq):
    # --- modify angles' references --- #
    q0 = q[0]; q1 = q[1]
    vq0 = vq[0]; vq1 = vq[1]
    # --- calculate x and vx of the foot w.r.t. to the hip joint --- #
    x = robot.l_upper*np.sin(q0) + robot.l_lower*np.sin(q0+q1)
    vx = robot.l_upper*np.cos(q0)*vq0 + robot.l_lower*np.cos(q0+q1)*(vq0+vq1)
    # --- calculate z and vz of the foot w.r.t. to the hip joint --- #
    z = robot.l_upper*np.cos(q0) + robot.l_lower*np.cos(q0+q1)
    vz = -robot.l_upper*np.sin(q0)*vq0 + -robot.l_lower*np.sin(q0+q1)*(vq0+vq1)
    # --- calculate leg's jacobian --- #
    j_11 = robot.l_upper*np.cos(q0) + robot.l_lower*np.cos(q0+q1)
    j_12 = robot.l_lower*np.cos(q0+q1)
    j_21 = -robot.l_upper*np.sin(q0) + -robot.l_lower*np.sin(q0+q1)
    j_22 = -robot.l_lower*np.sin(q0+q1)
    return x, vx, z, vz, np.array([[j_11, j_12], [j_21, j_22]])


def calc_kinematics_eqv(robot, q, vq):
    # --- modify angles' references --- #
    q0 = q[0]+np.pi/4; q2 = q[1]+np.pi/2; q3 = q2
    vq0 = vq[0]; vq2 = vq[1]; vq3 = vq[2]
    # --- calculate equivalent leg --- #
    l = np.sqrt(robot.l_calf**2 + robot.l_foot2**2 + 2*robot.l_calf*robot.l_foot2*np.cos(q3))
    dl_dq3 = (-2*robot.l_calf*robot.l_foot2*np.sin(q3)) / (2*l)
    dl_dt = dl_dq3 * vq3
    # --- calculate angle alpha --- #
    u = (robot.l_foot2/l) * np.sin(np.pi-q3)
    du_dq3 = -(robot.l_foot2/l)*np.cos(np.pi-q3) + -(robot.l_foot2/(l**2))*np.sin(np.pi-q3)*dl_dq3
    du_dt = -(robot.l_foot2/l)*np.cos(np.pi-q3)*vq3 + -(robot.l_foot2/(l**2))*np.sin(np.pi-q3)*dl_dt
    alpha = np.arcsin(u)
    dalpha_dq3 = (1/np.sqrt(1-(u**2))) * du_dq3
    dalpha_dt = (1/np.sqrt(1-(u**2))) * du_dt
    # --- calculate x and vx of the foot w.r.t. to the hip joint --- #
    x = robot.l_thigh*np.cos(q0) + l*np.cos(q0+q2-alpha)
    vx = -robot.l_thigh*np.sin(q0)*vq0 + dl_dt*np.cos(q0+q2-alpha) + -l*np.sin(q0+q2-alpha)*(vq0+vq2-dalpha_dt)
    # --- calculate z and vz of the foot w.r.t. to the hip joint --- #
    z = robot.l_thigh*np.sin(q0) + l*np.sin(q0+q2-alpha)
    vz = robot.l_thigh*np.cos(q0)*vq0 + dl_dt*np.sin(q0+q2-alpha) + l*np.cos(q0+q2-alpha)*(vq0+vq2-dalpha_dt)
    # --- calculate leg's jacobian --- #
    j_11 = -robot.l_thigh*np.sin(q0) + -l*np.sin(q0+q2-alpha)
    j_12 = dl_dq3*np.cos(q0+q2-alpha) + -l*np.sin(q0+q2-alpha)*(1-dalpha_dq3)
    j_21 = robot.l_thigh*np.cos(q0) + l*np.cos(q0+q2-alpha)
    j_22 = dl_dq3*np.sin(q0+q2-alpha) + l*np.cos(q0+q2-alpha)*(1-dalpha_dq3)
    return x, vx, z, vz, np.array([[j_11, j_12], [j_21, j_22]])


def calc_kinematics_org(robot, q, vq):
    # --- modify angles' references --- #
    q0 = q[0]+np.pi/4; q2 = q[1]+np.pi/2; q3 = q2
    vq0 = vq[0]; vq2 = vq[1]; vq3 = vq[2]
    # --- calculate x and vx of the foot w.r.t. to the hip joint --- #
    x = robot.l_thigh*np.cos(q0) + robot.l_calf*np.cos(q0+q2) + robot.l_foot2*np.cos(q0)
    vx = -robot.l_thigh*np.sin(q0)*vq0 + -robot.l_calf*np.sin(q0+q2)*(vq0+vq2) + -robot.l_foot2*np.sin(q0)*vq0
    # --- calculate z and vz of the foot w.r.t. to the hip joint --- #
    z = robot.l_thigh*np.sin(q0) + robot.l_calf*np.sin(q0+q2) + robot.l_foot2*np.sin(q0)
    vz = robot.l_thigh*np.cos(q0)*vq0 + robot.l_calf*np.cos(q0+q2)*(vq0+vq2) + robot.l_foot2*np.cos(q0)*vq0
    # --- calculate leg's jacobian --- #
    j_11 = -robot.l_thigh*np.sin(q0) + -robot.l_calf*np.sin(q0+q2) + -robot.l_foot2*np.sin(q0)
    j_12 = -robot.l_calf*np.sin(q0+q2)
    j_21 = robot.l_thigh*np.cos(q0) + robot.l_calf*np.cos(q0+q2) + robot.l_foot2*np.cos(q0)
    j_22 = robot.l_calf*np.cos(q0+q2)
    return x, vx, z, vz, np.array([[j_11, j_12], [j_21, j_22]])


def impedance_control(robot, x, vx, z, vz):
    fb_x_p = -robot.kp_x * (x - robot.x_d)
    fb_x_d = -robot.kd_x * (vx - robot.vx_d)
    fb_z_p = -robot.kp_z * (z - robot.z_d)
    fb_z_d = -robot.kd_z * (vz - robot.vz_d)
    imp_x = fb_x_p + fb_x_d
    imp_z = fb_z_p + fb_z_d
    return np.array([[imp_x], [imp_z]]), np.array([[fb_x_p], [fb_x_d]]), np.array([[fb_z_p], [fb_z_d]])


def calc_torques_solo_single(robot, torque_sat, mu_plane, cf_x_dir, cf_z_dir, cf_x, cf_z, flag_end_init):
    hip, knee = robot.extract_joint_states()
    q = np.array([hip[0], knee[0]])
    vq = np.array([hip[1], knee[1]])
    x, vx, z, vz, jac = calc_kinematics(robot, q, vq)
    act_cnt = np.array([[cf_x_dir*cf_x[0]], [cf_z_dir*cf_z[0]]])
    if flag_end_init:
        act_fb, fb_x, fb_z = impedance_control(robot, x, vx, z, vz)
    else:
        act_fb, fb_x, fb_z = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
    if robot.flag_control_mode == 0:
        act_tot = act_fb
    else:
        act_tot = act_cnt + act_fb
    if robot.flag_clamp_x_force:
        if abs(act_tot[0]) > mu_plane * abs(act_tot[1]) and act_tot[0] != 0:
            act_tot[0] = mu_plane * abs(act_tot[1]) * (act_tot[0] / abs(act_tot[0]))
    if flag_end_init:
        torques = np.matmul(np.transpose(jac), act_tot)
        for j in range(len(torques)):
            if abs(torques[j]) > torque_sat:
                torques[j] = torque_sat * (torques[j] / abs(torques[j]))
    else:
        torques = np.array([[hip[3]], [knee[3]]])
    return torques, act_tot, act_cnt, act_fb, fb_x, fb_z, x, vx, z, vz


def calc_torques_solo_full(robot, torque_sat, mu_plane, cf_x_dir, cf_z_dir, cf_x, cf_z, flag_end_init):
    fl_hip, fl_knee, fr_hip, fr_knee, bl_hip, bl_knee, br_hip, br_knee = robot.extract_joint_states()
    q = np.array([fl_hip[0], fl_knee[0], fr_hip[0], fr_knee[0], bl_hip[0], bl_knee[0], br_hip[0], br_knee[0]])
    vq = np.array([fl_hip[1], fl_knee[1], fr_hip[1], fr_knee[1], bl_hip[1], bl_knee[1], br_hip[1], br_knee[1]])
    x_fl, vx_fl, z_fl, vz_fl, jac_fl = calc_kinematics(robot, [q[0], q[1]], [vq[0], vq[1]])
    x_fr, vx_fr, z_fr, vz_fr, jac_fr = calc_kinematics(robot, [q[2], q[3]], [vq[2], vq[3]])
    x_bl, vx_bl, z_bl, vz_bl, jac_bl = calc_kinematics(robot, [q[4], q[5]], [vq[4], vq[5]])
    x_br, vx_br, z_br, vz_br, jac_br = calc_kinematics(robot, [q[6], q[7]], [vq[6], vq[7]])
    act_cnt_fl = np.array([[cf_x_dir*cf_x[0]/2], [cf_z_dir*cf_z[0]/2]])
    act_cnt_fr = np.array([[cf_x_dir*cf_x[0]/2], [cf_z_dir*cf_z[0]/2]])
    act_cnt_bl = np.array([[cf_x_dir*cf_x[1]/2], [cf_z_dir*cf_z[1]/2]])
    act_cnt_br = np.array([[cf_x_dir*cf_x[1]/2], [cf_z_dir*cf_z[1]/2]])
    if flag_end_init:
        act_fb_fl, fb_x_fl, fb_z_fl = impedance_control(robot, x_fl, vx_fl, z_fl, vz_fl)
        act_fb_fr, fb_x_fr, fb_z_fr = impedance_control(robot, x_fr, vx_fr, z_fr, vz_fr)
        act_fb_bl, fb_x_bl, fb_z_bl = impedance_control(robot, x_bl, vx_bl, z_bl, vz_bl)
        act_fb_br, fb_x_br, fb_z_br = impedance_control(robot, x_br, vx_br, z_br, vz_br)
    else:
        act_fb_fl, fb_x_fl, fb_z_fl = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
        act_fb_fr, fb_x_fr, fb_z_fr = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
        act_fb_bl, fb_x_bl, fb_z_bl = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
        act_fb_br, fb_x_br, fb_z_br = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
    if robot.flag_control_mode == 0:
        act_tot_fl = act_fb_fl
        act_tot_fr = act_fb_fr
        act_tot_bl = act_fb_bl
        act_tot_br = act_fb_br
    else:
        act_tot_fl = act_cnt_fl + act_fb_fl
        act_tot_fr = act_cnt_fr + act_fb_fr
        act_tot_bl = act_cnt_bl + act_fb_bl
        act_tot_br = act_cnt_br + act_fb_br
    if robot.flag_clamp_x_force:
        for item in [act_tot_fl, act_tot_fr, act_tot_bl, act_tot_br]:
            if abs(item[0]) > mu_plane * abs(item[1]) and item[0] != 0:
                item[0] = mu_plane * abs(item[1]) * (item[0] / abs(item[0]))
    if flag_end_init:
        torques_fl = np.matmul(np.transpose(jac_fl), act_tot_fl)
        torques_fr = np.matmul(np.transpose(jac_fr), act_tot_fr)
        torques_bl = np.matmul(np.transpose(jac_bl), act_tot_bl)
        torques_br = np.matmul(np.transpose(jac_br), act_tot_br)
        torques = np.vstack((torques_fl, torques_fr, torques_bl, torques_br))
        for j in range(len(torques)):
            if abs(torques[j]) > torque_sat:
                torques[j] = torque_sat * (torques[j] / abs(torques[j]))
    else:
        torques = np.array(
            [[fl_hip[3]], [fl_knee[3]], [fr_hip[3]], [fr_knee[3]], [bl_hip[3]], [bl_knee[3]], [br_hip[3]],
             [br_knee[3]]])
    return torques, \
           act_tot_fl, act_tot_fr, act_tot_bl, act_tot_br, \
           act_cnt_fl, act_cnt_fr, act_cnt_bl, act_cnt_br, \
           act_fb_fl, act_fb_fr, act_fb_bl, act_fb_br, \
           fb_x_fl, fb_x_fr, fb_x_bl, fb_x_br, \
           fb_z_fl, fb_z_fr, fb_z_bl, fb_z_br, \
           x_fl, x_fr, x_bl, x_br, \
           vx_fl, vx_fr, vx_bl, vx_br, \
           z_fl, z_fr, z_bl, z_br, \
           vz_fl, vz_fr, vz_bl, vz_br


def calc_torques_biorob_single(robot, torque_sat, mu_plane, cf_x_dir, cf_z_dir, cf_x, cf_z, flag_end_init):
    hip, knee, heel = robot.extract_joint_states()
    q = np.array([hip[0], knee[0], heel[0]])
    vq = np.array([hip[1], knee[1], heel[1]])
    if robot.flag_kin_model == 0:
        x, vx, z, vz, jac = calc_kinematics_org(robot, q, vq)
    elif robot.flag_kin_model == 1:
        x, vx, z, vz, jac = calc_kinematics_eqv(robot, q, vq)
    act_cnt = np.array([[cf_x_dir*cf_x[0]], [cf_z_dir*cf_z[0]]])
    if flag_end_init:
        act_fb, fb_x, fb_z = impedance_control(robot, x, vx, z, vz)
    else:
        act_fb, fb_x, fb_z = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
    if robot.flag_control_mode == 0:
        act_tot = act_fb
    else:
        act_tot = act_cnt + act_fb
    if robot.flag_clamp_x_force:
        if abs(act_tot[0]) > mu_plane * abs(act_tot[1]) and act_tot[0] != 0:
            act_tot[0] = mu_plane * abs(act_tot[1]) * (act_tot[0] / abs(act_tot[0]))
    if flag_end_init:
        torques = np.matmul(np.transpose(jac), act_tot)
        for j in range(len(torques)):
            if abs(torques[j]) > torque_sat:
                torques[j] = torque_sat * (torques[j] / abs(torques[j]))
    else:
        torques = np.array([[hip[3]], [knee[3]]])
    return torques, act_tot, act_cnt, act_fb, fb_x, fb_z, x, vx, z, vz


def calc_torques_biorob_full(robot, torque_sat, mu_plane, cf_x_dir, cf_z_dir, cf_x, cf_z, flag_end_init):
    fl_hip, fl_knee, fl_ankle, fr_hip, fr_knee, fr_ankle, \
    bl_hip, bl_knee, bl_ankle, br_hip, br_knee, br_ankle = robot.extract_joint_states()
    q = np.array([fl_hip[0], fl_knee[0], fl_ankle[0], fr_hip[0], fr_knee[0], fr_ankle[0],
                  bl_hip[0], bl_knee[0], bl_ankle[0], br_hip[0], br_knee[0], br_ankle[0]])
    # q = np.multiply(q, -1 * np.array([-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1]))
    # print(q.shape)
    vq = np.array([fl_hip[1], fl_knee[1], fl_ankle[1], fr_hip[1], fr_knee[1], fr_ankle[1],
                   bl_hip[1], bl_knee[1], bl_ankle[1], br_hip[1], br_knee[1], br_ankle[1]])
    # vq = np.multiply(vq, -1 * np.array([-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1]))
    # print(vq.shape)
    if robot.flag_kin_model == 0:
        x_fl, vx_fl, z_fl, vz_fl, jac_fl = calc_kinematics_org(robot, [q[0], q[1], q[2]], [vq[0], vq[1], vq[2]])
        x_fr, vx_fr, z_fr, vz_fr, jac_fr = calc_kinematics_org(robot, [q[3], q[4], q[5]], [vq[3], vq[4], vq[5]])
        x_bl, vx_bl, z_bl, vz_bl, jac_bl = calc_kinematics_org(robot, [q[6], q[7], q[8]], [vq[6], vq[7], vq[8]])
        x_br, vx_br, z_br, vz_br, jac_br = calc_kinematics_org(robot, [q[9], q[10], q[11]], [vq[9], vq[10], vq[11]])
    elif robot.flag_kin_model == 1:
        x_fl, vx_fl, z_fl, vz_fl, jac_fl = calc_kinematics_eqv(robot, [q[0], q[1], q[2]], [vq[0], vq[1], vq[2]])
        x_fr, vx_fr, z_fr, vz_fr, jac_fr = calc_kinematics_eqv(robot, [q[3], q[4], q[5]], [vq[3], vq[4], vq[5]])
        x_bl, vx_bl, z_bl, vz_bl, jac_bl = calc_kinematics_eqv(robot, [q[6], q[7], q[8]], [vq[6], vq[7], vq[8]])
        x_br, vx_br, z_br, vz_br, jac_br = calc_kinematics_eqv(robot, [q[9], q[10], q[11]], [vq[9], vq[10], vq[11]])
    act_cnt_fl = np.array([[cf_x_dir*cf_x[0]/2], [cf_z_dir*cf_z[0]/2]])
    act_cnt_fr = np.array([[cf_x_dir*cf_x[0]/2], [cf_z_dir*cf_z[0]/2]])
    act_cnt_bl = np.array([[cf_x_dir*cf_x[1]/2], [cf_z_dir*cf_z[1]/2]])
    act_cnt_br = np.array([[cf_x_dir*cf_x[1]/2], [cf_z_dir*cf_z[1]/2]])
    if flag_end_init:
        act_fb_fl, fb_x_fl, fb_z_fl = impedance_control(robot, x_fl, vx_fl, z_fl, vz_fl)
        act_fb_fr, fb_x_fr, fb_z_fr = impedance_control(robot, x_fr, vx_fr, z_fr, vz_fr)
        act_fb_bl, fb_x_bl, fb_z_bl = impedance_control(robot, x_bl, vx_bl, z_bl, vz_bl)
        act_fb_br, fb_x_br, fb_z_br = impedance_control(robot, x_br, vx_br, z_br, vz_br)
    else:
        act_fb_fl, fb_x_fl, fb_z_fl = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
        act_fb_fr, fb_x_fr, fb_z_fr = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
        act_fb_bl, fb_x_bl, fb_z_bl = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
        act_fb_br, fb_x_br, fb_z_br = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
    if robot.flag_control_mode == 0:
        act_tot_fl = act_fb_fl
        act_tot_fr = act_fb_fr
        act_tot_bl = act_fb_bl
        act_tot_br = act_fb_br
    else:
        act_tot_fl = act_cnt_fl + act_fb_fl
        act_tot_fr = act_cnt_fr + act_fb_fr
        act_tot_bl = act_cnt_bl + act_fb_bl
        act_tot_br = act_cnt_br + act_fb_br
    if robot.flag_clamp_x_force:
        for item in [act_tot_fl, act_tot_fr, act_tot_bl, act_tot_br]:
            if abs(item[0]) > mu_plane * abs(item[1]) and item[0] != 0:
                item[0] = mu_plane * abs(item[1]) * (item[0] / abs(item[0]))
    if flag_end_init:
        torques_fl = np.matmul(np.transpose(jac_fl), act_tot_fl)
        torques_fr = np.matmul(np.transpose(jac_fr), act_tot_fr)
        torques_bl = np.matmul(np.transpose(jac_bl), act_tot_bl)
        torques_br = np.matmul(np.transpose(jac_br), act_tot_br)
        torques = np.vstack((torques_fl, torques_fr, torques_bl, torques_br))
        for j in range(len(torques)):
            if abs(torques[j]) > torque_sat:
                torques[j] = torque_sat * (torques[j] / abs(torques[j]))
        # torques = np.multiply(torques, np.array([[-1], [-1], [1], [1], [-1], [-1], [1], [1]]))
        # print(torques.shape)
    else:
        torques = np.array([[fl_hip[3]], [fl_knee[3]], [fr_hip[3]], [fr_knee[3]], [bl_hip[3]], [bl_knee[3]], [br_hip[3]], [br_knee[3]]])
    return torques, \
           act_tot_fl, act_tot_fr, act_tot_bl, act_tot_br, \
           act_cnt_fl, act_cnt_fr, act_cnt_bl, act_cnt_br, \
           act_fb_fl, act_fb_fr, act_fb_bl, act_fb_br, \
           fb_x_fl, fb_x_fr, fb_x_bl, fb_x_br, \
           fb_z_fl, fb_z_fr, fb_z_bl, fb_z_br, \
           x_fl, x_fr, x_bl, x_br, \
           vx_fl, vx_fr, vx_bl, vx_br, \
           z_fl, z_fr, z_bl, z_br, \
           vz_fl, vz_fr, vz_bl, vz_br


def bezier_force_profile(t, t_stance, t_alpha, alpha):
    bs1 = t/t_alpha
    bs2 = (t-t_alpha)/(t_stance-t_alpha)
    bc1 = alpha * np.array([0.0, 0.8, 1.0, 1.0])
    bc2 = alpha * np.array([1.0, 1.0, 0.8, 0.0])
    if 0 <= bs1 <= 1:
        force = bc1[0]*((1-bs1)**3) + bc1[1]*3*bs1*((1-bs1)**2) + bc1[2]*3*(1-bs1)*(bs1**2) + bc1[3]*(bs1**3)
    elif 0 < bs2 <= 1:
        force = bc2[0]*((1-bs2)**3) + bc2[1]*3*bs2*((1-bs2)**2) + bc2[2]*3*(1-bs2)*(bs2**2) + bc2[3]*(bs2**3)
    else:
        force = 0.0
    return force


def gen_z_contact_force(t, t_stance, t_swing, t_air, t_alpha, t_init, val_period, alpha_z, flag_robot_state):
    temp = t-t_init-(val_period-1)*(t_stance+t_swing)
    if flag_robot_state == 0:
        if val_period == 0:
            front_force = 0.0
        elif temp <= t_stance+t_air:
            front_force = bezier_force_profile(temp, t_stance, t_alpha, alpha_z)
        else:
            front_force = 0.0
        back_force = front_force
    elif flag_robot_state == 1 or flag_robot_state == 2:
        if val_period == 0:
            front_force = 0.0; back_force = 0.0
        elif temp <= t_stance+t_air:
            front_force = bezier_force_profile(temp, t_stance, t_alpha, alpha_z)
            back_force = 0.0
        else:
            front_force = 0.0
            back_force = bezier_force_profile(temp-t_stance-t_air, t_stance, t_alpha, alpha_z)
    return front_force, back_force


def gen_x_contact_force(t, t_stance, t_swing, t_air, t_alpha, t_init, val_period, alpha_x, flag_robot_state):
    temp = t-t_init-(val_period-1)*(t_stance+t_swing)
    if flag_robot_state == 0 or flag_robot_state == 1:
        front_force = 0.0; back_force = 0.0
    elif flag_robot_state == 2:
        if val_period == 0:
            front_force = 0.0; back_force = 0.0
        elif temp <= t_stance+t_air:
            front_force = bezier_force_profile(temp, t_stance, t_alpha, -alpha_x)
            back_force = 0.0
        else:
            front_force = 0.0
            back_force = bezier_force_profile(temp-t_stance-t_air, t_stance, t_alpha, alpha_x)
    elif flag_robot_state == 3:
        if val_period == 0:
            front_force = 0.0; back_force = 0.0
        elif temp <= t_stance+t_air:
            front_force = bezier_force_profile(temp, t_stance, t_alpha, -alpha_x)
            back_force = -front_force
        else:
            front_force = 0.0; back_force = 0.0
    return front_force, back_force


def plot_single(flag_robot_model, time_array, count_period, m, g, z_d,
                torques_axis, torque_sat, com_axis, cf_z_axis,
                act_tot_axis, act_cnt_axis, act_fb_axis,
                fb_x_axis, fb_z_axis):
    fig, gph = plt.subplots(3, 3, constrained_layout=True, figsize=(24, 12))
    gph[0, 0].plot(time_array, torques_axis[0, :])
    gph[0, 0].plot(time_array, torques_axis[1, :])
    gph[0, 0].hlines(torque_sat, 0, 1, transform=gph[0, 0].get_yaxis_transform(), colors='r')
    gph[0, 0].hlines(-torque_sat, 0, 1, transform=gph[0, 0].get_yaxis_transform(), colors='r')
    gph[0, 0].grid()
    gph[0, 0].set_title('joint torques')
    gph[0, 0].set_xlabel('t (s)')
    gph[0, 0].set_ylabel('T (N.m)')
    gph[0, 0].legend(['hip', 'knee'])
    gph[0, 0].set_ylim([-torque_sat - 0.5, torque_sat + 0.5])
    gph[0, 1].plot(time_array, act_tot_axis[1, :])
    gph[0, 1].plot(time_array, act_cnt_axis[1, :], '--')
    gph[0, 1].plot(time_array, act_fb_axis[1, :], '--')
    gph[0, 1].grid()
    gph[0, 1].set_title('z virtual forces')
    gph[0, 1].set_xlabel('t (s)')
    gph[0, 1].set_ylabel('F (N)')
    gph[0, 1].legend(['total', 'contact', 'feedback'])
    gph[0, 2].plot(time_array, act_tot_axis[0, :])
    gph[0, 2].plot(time_array, act_cnt_axis[0, :], '--')
    gph[0, 2].plot(time_array, act_fb_axis[0, :], '--')
    gph[0, 2].grid()
    gph[0, 2].set_title('x virtual forces')
    gph[0, 2].set_xlabel('t (s)')
    gph[0, 2].set_ylabel('F (N)')
    gph[0, 2].legend(['total', 'contact', 'feedback'])
    gph[1, 0].plot(time_array, com_axis[1, :])
    gph[1, 0].hlines(z_d, 0, 1, transform=gph[1, 0].get_yaxis_transform(), colors='r')
    gph[1, 0].grid()
    gph[1, 0].set_title('z_com vs t')
    gph[1, 0].set_xlabel('t (s)')
    gph[1, 0].set_ylabel('z_com (m)')
    gph[1, 1].plot(time_array, count_period)
    gph[1, 1].grid()
    gph[1, 1].set_title('period vs t')
    gph[1, 1].set_xlabel('t (s)')
    gph[1, 1].set_ylabel('period')
    gph[1, 2].plot(time_array, cf_z_axis[0, :])
    gph[1, 2].hlines(m * g, 0, 1, transform=gph[1, 2].get_yaxis_transform(), colors='r')
    gph[1, 2].grid()
    gph[1, 2].set_title('z contact force vs t')
    gph[1, 2].set_xlabel('t (s)')
    gph[1, 2].set_ylabel('Fz (N)')
    gph[2, 0].plot(time_array, fb_x_axis[0, :])
    gph[2, 0].plot(time_array, fb_x_axis[1, :])
    gph[2, 0].grid()
    gph[2, 0].set_title('x feedback forces')
    gph[2, 0].set_xlabel('t (s)')
    gph[2, 0].set_ylabel('F (N)')
    gph[2, 0].legend(['Kp_x', 'Kd_x'])
    gph[2, 1].plot(time_array, fb_z_axis[0, :])
    gph[2, 1].plot(time_array, fb_z_axis[1, :])
    gph[2, 1].grid()
    gph[2, 1].set_title('z feedback forces')
    gph[2, 1].set_xlabel('t (s)')
    gph[2, 1].set_ylabel('F (N)')
    gph[2, 1].legend(['Kp_z', 'Kd_z'])
    if flag_robot_model == 0:
        fig.savefig('./solo_single_hopping.png')
    elif flag_robot_model == 2:
        fig.savefig('./biorob_single_hopping.png')
    elif flag_robot_model == 3:
        fig.savefig('./biorob_single_without_toes_hopping.png')


def plot_full(flag_robot_model, flag_robot_state,
              time_array, count_period, m, g, vx_d, z_d,
              torques_axis, torque_sat,
              com_axis, cf_x_axis, cf_z_axis,
              fl_tot_axis, fl_cnt_axis, fl_fb_axis,
              fr_tot_axis, fr_cnt_axis, fr_fb_axis,
              bl_tot_axis, bl_cnt_axis, bl_fb_axis,
              br_tot_axis, br_cnt_axis, br_fb_axis,
              fl_fb_x_axis, fl_fb_z_axis, fr_fb_x_axis, fr_fb_z_axis,
              bl_fb_x_axis, bl_fb_z_axis, br_fb_x_axis, br_fb_z_axis):
    fig, gph = plt.subplots(4, 4, constrained_layout=True, figsize=(24, 12))
    gph[0, 0].plot(time_array, torques_axis[0, :])
    gph[0, 0].plot(time_array, torques_axis[1, :])
    gph[0, 0].hlines(torque_sat, 0, 1, transform=gph[0, 0].get_yaxis_transform(), colors='r')
    gph[0, 0].hlines(-torque_sat, 0, 1, transform=gph[0, 0].get_yaxis_transform(), colors='r')
    gph[0, 0].grid()
    gph[0, 0].set_title('front left leg')
    gph[0, 0].set_xlabel('t (s)')
    gph[0, 0].set_ylabel('T (N.m)')
    gph[0, 0].legend(['hip', 'knee'])
    gph[1, 0].plot(time_array, torques_axis[2, :])
    gph[1, 0].plot(time_array, torques_axis[3, :])
    gph[1, 0].hlines(torque_sat, 0, 1, transform=gph[1, 0].get_yaxis_transform(), colors='r')
    gph[1, 0].hlines(-torque_sat, 0, 1, transform=gph[1, 0].get_yaxis_transform(), colors='r')
    gph[1, 0].grid()
    gph[1, 0].set_title('front right leg')
    gph[1, 0].set_xlabel('t (s)')
    gph[1, 0].set_ylabel('T (N.m)')
    gph[1, 0].legend(['hip', 'knee'])
    gph[2, 0].plot(time_array, torques_axis[4, :])
    gph[2, 0].plot(time_array, torques_axis[5, :])
    gph[2, 0].hlines(torque_sat, 0, 1, transform=gph[2, 0].get_yaxis_transform(), colors='r')
    gph[2, 0].hlines(-torque_sat, 0, 1, transform=gph[2, 0].get_yaxis_transform(), colors='r')
    gph[2, 0].grid()
    gph[2, 0].set_title('back left leg')
    gph[2, 0].set_xlabel('t (s)')
    gph[2, 0].set_ylabel('T (N.m)')
    gph[2, 0].legend(['hip', 'knee'])
    gph[3, 0].plot(time_array, torques_axis[6, :])
    gph[3, 0].plot(time_array, torques_axis[7, :])
    gph[3, 0].hlines(torque_sat, 0, 1, transform=gph[3, 0].get_yaxis_transform(), colors='r')
    gph[3, 0].hlines(-torque_sat, 0, 1, transform=gph[3, 0].get_yaxis_transform(), colors='r')
    gph[3, 0].grid()
    gph[3, 0].set_title('back right leg')
    gph[3, 0].set_xlabel('t (s)')
    gph[3, 0].set_ylabel('T (N.m)')
    gph[3, 0].legend(['hip', 'knee'])
    gph[0, 1].plot(time_array, fl_tot_axis[1, :])
    gph[0, 1].plot(time_array, fl_cnt_axis[1, :], '--')
    gph[0, 1].plot(time_array, fl_fb_axis[1, :], '--')
    gph[0, 1].grid()
    gph[0, 1].set_title('front left leg - z forces')
    gph[0, 1].set_xlabel('t (s)')
    gph[0, 1].set_ylabel('F (N)')
    gph[0, 1].legend(['total', 'contact', 'feedback'])
    gph[1, 1].plot(time_array, fr_tot_axis[1, :])
    gph[1, 1].plot(time_array, fr_cnt_axis[1, :], '--')
    gph[1, 1].plot(time_array, fr_fb_axis[1, :], '--')
    gph[1, 1].grid()
    gph[1, 1].set_title('front right leg - z forces')
    gph[1, 1].set_xlabel('t (s)')
    gph[1, 1].set_ylabel('F (N)')
    gph[1, 1].legend(['total', 'contact', 'feedback'])
    gph[2, 1].plot(time_array, bl_tot_axis[1, :])
    gph[2, 1].plot(time_array, bl_cnt_axis[1, :], '--')
    gph[2, 1].plot(time_array, bl_fb_axis[1, :], '--')
    gph[2, 1].grid()
    gph[2, 1].set_title('back left leg - z forces')
    gph[2, 1].set_xlabel('t (s)')
    gph[2, 1].set_ylabel('F (N)')
    gph[2, 1].legend(['total', 'contact', 'feedback'])
    gph[3, 1].plot(time_array, br_tot_axis[1, :])
    gph[3, 1].plot(time_array, br_cnt_axis[1, :], '--')
    gph[3, 1].plot(time_array, br_fb_axis[1, :], '--')
    gph[3, 1].grid()
    gph[3, 1].set_title('back right leg - z forces')
    gph[3, 1].set_xlabel('t (s)')
    gph[3, 1].set_ylabel('F (N)')
    gph[3, 1].legend(['total', 'contact', 'feedback'])
    gph[0, 2].plot(time_array, fl_tot_axis[0, :])
    gph[0, 2].plot(time_array, fl_cnt_axis[0, :], '--')
    gph[0, 2].plot(time_array, fl_fb_axis[0, :], '--')
    gph[0, 2].grid()
    gph[0, 2].set_title('front left leg - x forces')
    gph[0, 2].set_xlabel('t (s)')
    gph[0, 2].set_ylabel('F (N)')
    gph[0, 2].legend(['total', 'contact', 'feedback'])
    gph[1, 2].plot(time_array, fr_tot_axis[0, :])
    gph[1, 2].plot(time_array, fr_cnt_axis[0, :], '--')
    gph[1, 2].plot(time_array, fr_fb_axis[0, :], '--')
    gph[1, 2].grid()
    gph[1, 2].set_title('front right leg - x forces')
    gph[1, 2].set_xlabel('t (s)')
    gph[1, 2].set_ylabel('F (N)')
    gph[1, 2].legend(['total', 'contact', 'feedback'])
    gph[2, 2].plot(time_array, bl_tot_axis[0, :])
    gph[2, 2].plot(time_array, bl_cnt_axis[0, :], '--')
    gph[2, 2].plot(time_array, bl_fb_axis[0, :], '--')
    gph[2, 2].grid()
    gph[2, 2].set_title('back left leg - x forces')
    gph[2, 2].set_xlabel('t (s)')
    gph[2, 2].set_ylabel('F (N)')
    gph[2, 2].legend(['total', 'contact', 'feedback'])
    gph[3, 2].plot(time_array, br_tot_axis[0, :])
    gph[3, 2].plot(time_array, br_cnt_axis[0, :], '--')
    gph[3, 2].plot(time_array, br_fb_axis[0, :], '--')
    gph[3, 2].grid()
    gph[3, 2].set_title('back right leg - x forces')
    gph[3, 2].set_xlabel('t (s)')
    gph[3, 2].set_ylabel('F (N)')
    gph[3, 2].legend(['total', 'contact', 'feedback'])
    gph[0, 3].plot(time_array, np.zeros(len(time_array)))
    gph[0, 3].plot(time_array, np.zeros(len(time_array)))
    gph[0, 3].grid()
    gph[0, 3].set_title('front left foot - measured forces')
    gph[0, 3].set_xlabel('t (s)')
    gph[0, 3].set_ylabel('F (N)')
    gph[0, 3].legend(['x', 'z'])
    gph[1, 3].plot(time_array, np.zeros(len(time_array)))
    gph[1, 3].plot(time_array, np.zeros(len(time_array)))
    gph[1, 3].grid()
    gph[1, 3].set_title('front right foot - measured forces')
    gph[1, 3].set_xlabel('t (s)')
    gph[1, 3].set_ylabel('F (N)')
    gph[1, 3].legend(['x', 'z'])
    gph[2, 3].plot(time_array, np.zeros(len(time_array)))
    gph[2, 3].plot(time_array, np.zeros(len(time_array)))
    gph[2, 3].grid()
    gph[2, 3].set_title('back left foot - measured forces')
    gph[2, 3].set_xlabel('t (s)')
    gph[2, 3].set_ylabel('F (N)')
    gph[2, 3].legend(['x', 'z'])
    gph[3, 3].plot(time_array, np.zeros(len(time_array)))
    gph[3, 3].plot(time_array, np.zeros(len(time_array)))
    gph[3, 3].grid()
    gph[3, 3].set_title('back right foot - measured forces')
    gph[3, 3].set_xlabel('t (s)')
    gph[3, 3].set_ylabel('F (N)')
    gph[3, 3].legend(['x', 'z'])
    if flag_robot_model == 1:
        if flag_robot_state == 0:
            fig.savefig('./solo_full_hopping_01.png')
        elif flag_robot_state == 1:
            fig.savefig('./solo_full_bounding_01.png')
        elif flag_robot_state == 2:
            fig.savefig('./solo_full_forward_01.png')
    elif flag_robot_model == 4:
        if flag_robot_state == 0:
            fig.savefig('./biorob_full_hopping_01.png')
        elif flag_robot_state == 1:
            fig.savefig('./biorob_full_bounding_01.png')
        elif flag_robot_state == 2:
            fig.savefig('./biorob_full_forward_01.png')
    elif flag_robot_model == 5:
        if flag_robot_state == 0:
            fig.savefig('./biorob_full_without_toes_hopping_01.png')
        elif flag_robot_state == 1:
            fig.savefig('./biorob_full_without_toes_bounding_01.png')
        elif flag_robot_state == 2:
            fig.savefig('./biorob_full_without_toes_forward_01.png')
    fig, gph = plt.subplots(3, 4, constrained_layout=True, figsize=(24, 12))
    gph[0, 0].plot(time_array, com_axis[0, :])
    gph[0, 0].grid()
    gph[0, 0].set_title('x_com vs t')
    gph[0, 0].set_xlabel('t (s)')
    gph[0, 0].set_ylabel('x_com (m)')
    gph[0, 1].plot(time_array, com_axis[3, :])
    gph[0, 1].hlines(vx_d, 0, 1, transform=gph[0, 1].get_yaxis_transform(), colors='r')
    gph[0, 1].grid()
    gph[0, 1].set_title('vx_com vs t')
    gph[0, 1].set_xlabel('t (s)')
    gph[0, 1].set_ylabel('vx_com (m/s)')
    gph[0, 2].plot(time_array, com_axis[1, :])
    gph[0, 2].hlines(z_d, 0, 1, transform=gph[0, 2].get_yaxis_transform(), colors='r')
    gph[0, 2].grid()
    gph[0, 2].set_title('z_com vs t')
    gph[0, 2].set_xlabel('t (s)')
    gph[0, 2].set_ylabel('z_com (m)')
    gph[0, 3].plot(time_array, com_axis[4, :])
    gph[0, 3].grid()
    gph[0, 3].set_title('vz_com vs t')
    gph[0, 3].set_xlabel('t (s)')
    gph[0, 3].set_ylabel('vz_com (m/s)')
    gph[1, 0].plot(time_array, com_axis[2, :])
    gph[1, 0].grid()
    gph[1, 0].set_title('thy_com vs t')
    gph[1, 0].set_xlabel('t (s)')
    gph[1, 0].set_ylabel('th_com (rad)')
    gph[1, 1].plot(time_array, com_axis[5, :])
    gph[1, 1].grid()
    gph[1, 1].set_title('wy_com vs t')
    gph[1, 1].set_xlabel('t (s)')
    gph[1, 1].set_ylabel('wy_com (rad/s)')
    gph[1, 2].plot(com_axis[2, :], com_axis[5, :])
    gph[1, 2].grid()
    gph[1, 2].set_title('angular velocity vs angular position of the COM')
    gph[1, 2].set_xlabel('th_com (rad)')
    gph[1, 2].set_ylabel('wy_com (rad/s)')
    gph[1, 3].plot(time_array, count_period)
    gph[1, 3].grid()
    gph[1, 3].set_title('period vs t')
    gph[1, 3].set_xlabel('t (s)')
    gph[1, 3].set_ylabel('period')
    gph[2, 0].plot(time_array, cf_x_axis[0, :])
    gph[2, 0].plot(time_array, cf_x_axis[1, :])
    gph[2, 0].grid()
    gph[2, 0].set_title('x contact forces vs t')
    gph[2, 0].set_xlabel('t (s)')
    gph[2, 0].set_ylabel('Fx (N)')
    gph[2, 0].legend(['front', 'back'])
    gph[2, 1].plot(time_array, cf_z_axis[0, :])
    gph[2, 1].plot(time_array, cf_z_axis[1, :])
    gph[2, 1].hlines(m*g, 0, 1, transform=gph[2, 1].get_yaxis_transform(), colors='r')
    gph[2, 1].grid()
    gph[2, 1].set_title('z contact forces vs t')
    gph[2, 1].set_xlabel('t (s)')
    gph[2, 1].set_ylabel('Fz (N)')
    gph[2, 1].legend(['front', 'back'])
    if flag_robot_model == 1:
        if flag_robot_state == 0:
            fig.savefig('./solo_full_hopping_02.png')
        elif flag_robot_state == 1:
            fig.savefig('./solo_full_bounding_02.png')
        elif flag_robot_state == 2:
            fig.savefig('./solo_full_forward_02.png')
    elif flag_robot_model == 4:
        if flag_robot_state == 0:
            fig.savefig('./biorob_full_hopping_02.png')
        elif flag_robot_state == 1:
            fig.savefig('./biorob_full_bounding_02.png')
        elif flag_robot_state == 2:
            fig.savefig('./biorob_full_forward_02.png')
    elif flag_robot_model == 5:
        if flag_robot_state == 0:
            fig.savefig('./biorob_full_without_toes_hopping_02.png')
        elif flag_robot_state == 1:
            fig.savefig('./biorob_full_without_toes_bounding_02.png')
        elif flag_robot_state == 2:
            fig.savefig('./biorob_full_without_toes_forward_02.png')
    fig, gph = plt.subplots(4, 4, constrained_layout=True, figsize=(24, 12))
    gph[0, 0].plot(time_array, fl_fb_x_axis[0, :])
    gph[0, 0].plot(time_array, fl_fb_x_axis[1, :])
    gph[0, 0].grid()
    gph[0, 0].set_title('front left leg - x feedback forces')
    gph[0, 0].set_xlabel('t (s)')
    gph[0, 0].set_ylabel('feedback force (N)')
    gph[0, 0].legend(['Kp_x', 'Kd_x'])
    gph[0, 1].plot(time_array, fl_fb_z_axis[0, :])
    gph[0, 1].plot(time_array, fl_fb_x_axis[1, :])
    gph[0, 1].grid()
    gph[0, 1].set_title('front left leg - z feedback forces')
    gph[0, 1].set_xlabel('t (s)')
    gph[0, 1].set_ylabel('feedback force (N)')
    gph[0, 1].legend(['Kp_z', 'Kd_z'])
    gph[1, 0].plot(time_array, fr_fb_x_axis[0, :])
    gph[1, 0].plot(time_array, fr_fb_x_axis[1, :])
    gph[1, 0].grid()
    gph[1, 0].set_title('front right leg - x feedback forces')
    gph[1, 0].set_xlabel('t (s)')
    gph[1, 0].set_ylabel('feedback force (N)')
    gph[1, 0].legend(['Kp_x', 'Kd_x'])
    gph[1, 1].plot(time_array, fr_fb_z_axis[0, :])
    gph[1, 1].plot(time_array, fr_fb_z_axis[1, :])
    gph[1, 1].grid()
    gph[1, 1].set_title('front right leg - z feedback forces')
    gph[1, 1].set_xlabel('t (s)')
    gph[1, 1].set_ylabel('feedback force (N)')
    gph[1, 1].legend(['Kp_z', 'Kd_z'])
    gph[2, 0].plot(time_array, bl_fb_x_axis[0, :])
    gph[2, 0].plot(time_array, bl_fb_x_axis[1, :])
    gph[2, 0].grid()
    gph[2, 0].set_title('back left leg - x feedback forces')
    gph[2, 0].set_xlabel('t (s)')
    gph[2, 0].set_ylabel('feedback force (N)')
    gph[2, 0].legend(['Kp_x', 'Kd_x'])
    gph[2, 1].plot(time_array, bl_fb_z_axis[0, :])
    gph[2, 1].plot(time_array, bl_fb_z_axis[1, :])
    gph[2, 1].grid()
    gph[2, 1].set_title('back left leg - z feedback forces')
    gph[2, 1].set_xlabel('t (s)')
    gph[2, 1].set_ylabel('feedback force (N)')
    gph[2, 1].legend(['Kp_z', 'Kd_z'])
    gph[3, 0].plot(time_array, br_fb_x_axis[0, :])
    gph[3, 0].plot(time_array, br_fb_x_axis[1, :])
    gph[3, 0].grid()
    gph[3, 0].set_title('back right leg - x feedback forces')
    gph[3, 0].set_xlabel('t (s)')
    gph[3, 0].set_ylabel('feedback force (N)')
    gph[3, 0].legend(['Kp_x', 'Kd_x'])
    gph[3, 1].plot(time_array, br_fb_z_axis[0, :])
    gph[3, 1].plot(time_array, br_fb_z_axis[1, :])
    gph[3, 1].grid()
    gph[3, 1].set_title('back right leg - z feedback forces')
    gph[3, 1].set_xlabel('t (s)')
    gph[3, 1].set_ylabel('feedback force (N)')
    gph[3, 1].legend(['Kp_z', 'Kd_z'])
    if flag_robot_model == 1:
        if flag_robot_state == 0:
            fig.savefig('./solo_full_hopping_03.png')
        elif flag_robot_state == 1:
            fig.savefig('./solo_full_bounding_03.png')
        elif flag_robot_state == 2:
            fig.savefig('./solo_full_forward_03.png')
    elif flag_robot_model == 4:
        if flag_robot_state == 0:
            fig.savefig('./biorob_full_hopping_03.png')
        elif flag_robot_state == 1:
            fig.savefig('./biorob_full_bounding_03.png')
        elif flag_robot_state == 2:
            fig.savefig('./biorob_full_forward_03.png')
    elif flag_robot_model == 5:
        if flag_robot_state == 0:
            fig.savefig('./biorob_full_without_toes_hopping_03.png')
        elif flag_robot_state == 1:
            fig.savefig('./biorob_full_without_toes_bounding_03.png')
        elif flag_robot_state == 2:
            fig.savefig('./biorob_full_without_toes_forward_03.png')
    