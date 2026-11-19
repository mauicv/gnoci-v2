# Part masses in grams, written to robot.xml by scripts/update_masses.py.
# "est." marks parts of the 2026-11-18 redesign that haven't been weighed yet: the
# printed ones are estimated as CAD volume x 0.65 g/cm3 (the median for the weighed
# printed parts of the old design), or x the same part's old ratio where it existed;
# the TPU sole as volume x 1.21 g/cm3 (solid TPU). Weigh them and replace the numbers.
MASSES = {
    # torso
    "body_main": 105.0,  # est. (old body_main: 166 g, 253 cm3; new 160 cm3)
    "body_top": 52.0,  # est. (80 cm3)
    "body_battery_lid": 15.0,  # est. (24 cm3, like the old 15 g lid)
    "mounting_1": 12.0,  # est. (18 cm3)
    "mounting_2": 7.0,  # est. (11 cm3)
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
    "hip": 22.0,  # est. (old 18 g, 30 cm3; new 36 cm3)
    "upper_leg": 40.0,  # est. (old 26 g, 35 cm3; new 54 cm3)
    "lower_leg": 46.0,  # est. (old 45 g, 90 cm3; new 92 cm3)
    "foot_base": 23.6,  # est. (old 11 g, 14 cm3; 2026-11-18 redesign 47 cm3 = 36 g; 2026-11-19 feet 30.6 cm3)
    "foot_outer_side": 39.0,  # est. (60 cm3)
    "foot_inner_side": 32.0,  # est. (50 cm3)
    "foot_tpu_layer": 9.5,  # est. (8.1 cm3 of TPU, 2026-11-19 feet; was 13 g for 11.1 cm3)
    "foot_limit_switch": 3.0,
}
