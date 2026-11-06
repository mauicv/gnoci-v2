# TODO

## Model
- [x] Add part masses and inertias (guessed)
- [x] Measure part masses and update the visual geom `mass` values
- [x] Add joint limits
- [x] Add sensor block (IMU, foot pos/linvel, foot-floor contact)
- [x] Set IMU site orientation to match the real chip
- [x] Tune actuators (kp, damping, force limits, armature) to match the servos
- [x] Replace foot collision meshes with primitive boxes

## Sim / training
- [x] Port `joystick.py` to 6 DoF (joint names, pose weights, noise scales, indices)
- [x] Add `task_to_xml` and task definitions to `constants.py`
- [x] Get the joystick smoke test passing
- [x] Get the reset smoke test passing
- [x] Get the step smoke test passing
- [ ] Remove leftover Berkeley Humanoid docstrings and headers
- [x] Add a training script

## Control

### Servos
- [x] Figure out what the policy takes as input
- [x] Add code to save policy as onnx
- [x] Double check servo controller input units
- [x] Double check servo controller output units
- [x] Verify servo angle wrap behaviour
- [x] Verify servo rotational oreintations
- [x] Return joint posistions/velocities

### IMU
- [x] IMU logic ...
- [x] G vector (complementary filter)
- [x] gyro vector

### Next
- [ ] Set kds to 0 in servo controller.
- [ ] Find some way of comparing robot to sim...

## Misc
- [ ] Write the README
