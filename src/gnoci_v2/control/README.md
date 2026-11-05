
Policy inputs are:

noisy_gyro,  # 3
noisy_gravity,  # 3
info["command"],  # 3
noisy_joint_angles - self._default_pose,  # 6
noisy_joint_vel,  # 6
info["last_act"],  # 6
phase,
