MASSES = {
    # torso
    "body_main": 166.0,
    "body_back": 62.0,
    "body_back_lid": 15.0,
    "head_battery": 89.0,
    "head_button": 13.0,
    "head_fan": 6.0,
    "head_powerboard": 21.0,
    "head_r_pi": 66.0,
    "head_voltmeter": 4.0,
    "imu": 1.0,
    "urt_2_controller": 13.0,
    # servos
    "serv": 59.0,
    "serv_horn": 1.0,
    "serv_horn_back": 1.0,
    # legs
    "hip": 18.0,
    "upper_leg": 26.0,
    "lower_leg": 45.0,
    "foot_base": 11.0,
    "foot_side": 11.0,
    "foot_limit_switch": 3.0,
}

# Added to the torso's centre of mass, in metres in the torso frame (x forward,
# y left, z up), on top of where the meshes put it. Not measured: fitted on
# 2026-11-10 by replaying the robot's floor recordings (logs/floor) in the sim,
# which put it 30-50 mm forward of the model's (about 22 mm for the whole robot).
# dim-muddy-fiber, trained at +40 mm, then fell backward on the robot, and its
# recordings fitted +20 mm, so it's now 20 mm.
# The torso's mass and inertia still come from its meshes.
TORSO_COM_OFFSET = (0.02, 0.0, 0.0)
