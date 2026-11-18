# Notes: sim2real review and rebuild plan

Written 2026-11-16, after the robot broke. Evidence is from this repo's logs, sim checks with `walk-time-hold`, and the Open Duck Mini (ODM) code and git history in `code_ref/`. `sim2real-gaps.md` is an earlier list; where the two disagree, this one is the later view.

## What most contributes to sim2real failure

1. **The gait has to run on foot edges.**
   - With 3 joints per leg and the foot fixed to the shin, two flat feet lock the torso: no sway, no weight shift, no turning, unless a foot tips or slides. (Geometric reasoning from the model, not tested on the robot.)
   - Sim, `walk-time-hold`: a loaded foot is flat 0–18% of the time, on one edge 46–89%, on a single corner 6–54%. Both feet are down together 1–7% of the time. The left foot stays rolled about 10° whenever loaded.
   - The sim foot is a sharp rigid box. The real foot has pads or rubber, rounded edges, and play in the leg. The tilt-release tests show the largest sim/robot differences exactly there.
   - The robot's joints replay within 0.02–0.03 rad of the sim, yet the body moves differently. So the mismatch is body and contact, not servos.
2. **The gait has thin margins even in the sim.**
   - Falls in 37% of 20 s runs with pushes; 15% with only 20 ms extra action delay and no pushes.
   - Lifts its feet about 50 mm and never stands still, even at zero command.
   - Knees hit the sim's 3.35 Nm torque limit 2–6% of the time.
3. **Turning is done by scrubbing the feet.**
   - No hip yaw joint. The sim reaches 0.25 rad/s when asked for 0.7.
   - A foot on one corner has no resistance to twisting. The robot yaws 6–8× more than the sim.
4. **Mass distribution is unmeasured.** The robot looks front-heavy. This matters mainly because the margins are thin.
5. **Play in the legs isn't modelled.** ODM's model has ±0.5° backlash joints; this one has none. In the tilt tests the servo readings explain only about 2 mm of the roughly 3 mm per degree of give needed between the feet.
6. **The policy senses less than ODM's.** ODM feeds raw accelerometer, foot contact switches, three past actions and motor targets. This policy gets a filtered gravity estimate, no contacts, one past action.

## Not the problem

- Servo model, action scale, observation pipeline (hang test, replays).
- Latency: measured and trained for.
- IMU calibration: reads +0.5 to +1.5° pitch standing at home.
- Foot impacts: the sim shows the same 3–4 g spikes.
- Sideways drift: the sim policy drifts 0.08 m/s sideways at zero command, so it's the policy, not a weight imbalance.
- Leg spacing: same as ODM (feet 82 mm from the midline against 81 mm).

## Feet: what the tests showed

- Tilt release, first overshoot as a share of the release angle: foam pads 0.62–0.67, 1 mm rubber 0.53–0.59, bare 0.16–0.47, sim 0.26–0.31.
- Soft layers add bounce. Bare feet probably slide (in the sim, bounce disappears below a friction of about 0.6–0.7).
- Plot and script: `gait_analysis/tilt_release.png`, `gait_analysis/tilt_release.py`.

## gnoci v2 against Open Duck Mini

| | gnoci v2 | ODM |
|---|---|---|
| Joints per leg | 3 (hip roll, hip pitch, knee) | 5 (adds hip yaw, ankle pitch) |
| Mass | 1.05 kg | 2.1 kg |
| Centre of mass above the sole | 0.28 m | about 0.21 m |
| Hip roll axis above the sole | 272 mm | 159 mm |
| Foot | 81 × 34 mm box | 103 × 41 mm TPU sole (mesh) |
| Backlash in the model | none | ±0.5° per joint |
| Gait shaping | phase reward, 50 mm foot lift | imitation of a walk-engine reference |
| Action scale | 0.5 | 0.25, with a 5.24 rad/s target-speed limit |

ODM's history: about 8 months and roughly 1,250 commits from first runtime code (Jul 2024) to "pretty nice walk on the real robot" (24 Mar 2025). Most effort went on the IMU, joint velocity readings, action filtering, servo parameters, backlash and joint offsets. Imitation was there from the first Playground commit, a month before it walked, so it shaped the gait; it wasn't the sim2real fix.

## Rebuild plan (6 servos, no added joints)

Decided: no extra servos, no ankle (budget, complexity, and the aim of a cheapest-possible robot).

- **Shorter legs.** Aim for a hip height of 160–180 mm. Lower knee torque, feet larger relative to height.
- **Rounded soles.** Flat middle, heel, toe and side edges rounded at about 8–10 mm radius. Hard material with a thin grippy surface. Shape it so MuJoCo can match it: two round rails per foot (capsules), or use the sole mesh for collision as ODM does.
- **Lower, more horizontal body.** Lowers the centre of mass and resists pitch and yaw (yaw inertia is 0.0045 kg·m² against ODM's 0.0133).
- **Centre of mass over the middle of the foot.** Battery on a sliding mount. Weigh parts as you build, update `desc/masses.py`, then measure the real centre of mass and put it in the model.
- **Stiff joints.** Support each joint on both sides; thick printed parts. Measure the remaining play.
- **IMU** mounted rigidly near the centre of mass, at the model's `imu` site.
- **Keep the leg spacing.**

Considered and dropped:
- Ankle servo without an encoder: workable in principle, but adds complexity and unknowns.
- Passive parallelogram link to keep the foot level: optional; try rounded soles first.

## Training changes (free)

- Shrink or drop the yaw command at first; sideways walking barely works either (0.046 m/s when asked for 0.15).
- Foot lift about 30 mm, scaled to the shorter legs.
- Backlash joints of about ±0.5° in the model.
- Foot switches in the observation.
- Randomize contact stiffness and sole radius as well as friction.
- Gate before any robot test: at most a few percent of falls in the sim with pushes and with 20 ms extra delay.
- One change per model, and the same three tests each time: tilt release, x0.6 floor run, full-scale floor run.

## Open questions

- Servo voltage on battery and on the leash. The sim allows 3.35 Nm (ODM's identified value); the 7.4 V STS3215 may be rated nearer 1.9 Nm stall (from memory, not checked).
- Real foot friction: tilted-board test.
- Whether the new proportions and soles improve the sim margins: mock them up in the model before printing.
