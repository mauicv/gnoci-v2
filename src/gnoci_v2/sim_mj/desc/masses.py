# Part masses in grams, written to robot.xml by scripts/update_masses.py.
# "est." marks parts of the 2026-11-18 redesign that haven't been weighed yet: the
# printed ones are estimated as CAD volume x 0.65 g/cm3 (the median for the weighed
# printed parts of the old design), or x the same part's old ratio where it existed;
# the TPU sole as volume x 1.21 g/cm3 (solid TPU). Weigh them and replace the numbers.
MASSES = {
    # torso
    "body_main": 73.0,
    "body_top": 81.0,
    "body_battery_lid": 14.0,
    "mounting_1": 18.0,
    "mounting_2": 12.0,
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
    "hip": 20.0,
    "upper_leg": 26.0,
    "lower_leg": 38.0,
    "foot_base": 18.0,
    "foot_outer_side": 30.0,
    "foot_inner_side": 24.0, 
    "foot_tpu_layer": 10.0,
    "foot_limit_switch": 3.0,
}
