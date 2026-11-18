"""Turn an onshape-to-robot export into desc/robot.xml and desc/assets/.

The export (robot.xml + assets/ written by onshape-to-robot) has the geometry: the
parts, their placement and the joints. Everything tuned by hand comes from the
current desc/robot.xml and is kept: <compiler>, <option>, <default>, <actuator>,
<sensor>, <equality>, and each joint's range. So to change servo parameters,
sensors or joint ranges, edit desc/robot.xml; to change the shape of the robot,
re-export from Onshape and run this.

What it does to the export:
- Bodies that aren't attached to the robot in the assembly (onshape-to-robot gives
  them a freejoint, e.g. head_battery) are merged into body_main.
- Leg bodies are named after the joint that drives them: {side}_hip,
  {side}_upper_leg, {side}_lower_leg (the export names some after servo horns).
- The torso frame is moved to the midpoint between the two hip_lat axes, and the
  torso is placed at the height where the soles touch the floor at home.
- Joint axes are flipped where needed so both hip_lat joints turn about world +x and
  every hip and knee about world +y.
- Collision meshes and <inertial>s are dropped: inertia comes from the visual
  geoms (see the compiler's inertiagrouprange) and masses from desc/masses.py.
- Each sole is modelled as four spheres of --sole-radius, at the corners of the flat
  patch of the {side}_foot_tpu_layer part. With a flat patch and edges rounded at
  that radius (spherical corners), that is the exact shape on a flat floor.
- Sites: imu at the centre of the imu part, {l,r}_foot at the middle of each sole.

Usage:
  uv run python scripts/import_onshape.py [--export onshape-to-robot]
  uv run python scripts/update_masses.py
"""

import argparse
import copy
import pathlib
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

DESC = pathlib.Path(__file__).resolve().parents[1] / "src/gnoci_v2/training/desc"
SIDES = ("left", "right")
LEG_JOINTS = ("hip_lat", "hip", "knee")
LEG_BODIES = {"hip_lat": "hip", "hip": "upper_leg", "knee": "lower_leg"}
# World axis each joint must turn about, positive by the right-hand rule.
JOINT_AXES = {"hip_lat": (1, 0, 0), "hip": (0, 1, 0), "knee": (0, 1, 0)}
SOLE_PART = "{side}_foot_tpu_layer"
IMU_PART = "imu"
SITE_SHORT = {"left": "l", "right": "r"}


def fmt(values) -> str:
  return " ".join(f"{v:.6g}" if abs(v) > 1e-9 else "0" for v in values)


def mesh_world_vertices(m, d, mesh_name):
  for g in range(m.ngeom):
    if (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2
        and m.mesh(m.geom_dataid[g]).name == mesh_name):
      mid = m.geom_dataid[g]
      va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
      return m.mesh_vert[va:va + vn] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
  raise SystemExit(f"No visual geom with mesh {mesh_name!r} in the export")


def to_body(d, body_id, world_point):
  return d.xmat[body_id].reshape(3, 3).T @ (world_point - d.xpos[body_id])


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--export", type=pathlib.Path, default=pathlib.Path("onshape-to-robot"))
  parser.add_argument("--sole-radius", type=float, default=0.007,
                      help="Radius (m) of the rounded edges of the sole.")
  args = parser.parse_args()

  # Geometry from the export, at its zero pose.
  m = mujoco.MjModel.from_xml_path(str(args.export / "robot.xml"))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  anchor = {m.joint(j).name: d.xanchor[j].copy() for j in range(m.njnt)}
  axis = {m.joint(j).name: d.xaxis[j].copy() for j in range(m.njnt)}
  body_of_joint = {m.joint(j).name: m.body(m.jnt_bodyid[j]).name for j in range(m.njnt)}
  origin = (anchor["left_hip_lat"] + anchor["right_hip_lat"]) / 2

  r = args.sole_radius
  soles = {}
  for side in SIDES:
    v = mesh_world_vertices(m, d, SOLE_PART.format(side=side))
    lo, hi = v.min(0), v.max(0)
    centres = [np.array([x, y, lo[2] + r]) for x in (lo[0] + r, hi[0] - r)
               for y in (lo[1] + r, hi[1] - r)]
    flat = v[v[:, 2] < lo[2] + 1e-4]
    if np.ptp(flat[:, 2]) > 1e-4 or len(flat) < 4:
      raise SystemExit(f"The {side} sole isn't flat on the floor at the zero pose")
    soles[side] = (lo, hi, centres)
  sole_z = soles["left"][0][2]
  if abs(soles["right"][0][2] - sole_z) > 1e-4:
    raise SystemExit("The two soles are at different heights at the zero pose")
  home_z = origin[2] - sole_z
  imu_v = mesh_world_vertices(m, d, IMU_PART)
  imu_site = (imu_v.min(0) + imu_v.max(0)) / 2 - origin

  # Edit the export's XML.
  tree = ET.parse(args.export / "robot.xml", ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
  export = tree.getroot()
  worldbody = export.find("worldbody")
  main_body = worldbody.find("body[@name='body_main']")
  for attr, identity in (("pos", [0, 0, 0]), ("quat", [1, 0, 0, 0])):
    if not np.allclose([float(x) for x in main_body.get(attr, fmt(identity)).split()], identity):
      raise SystemExit("Expected body_main at the export's origin")

  # Merge loose bodies into body_main (they sit at the origin like body_main).
  for body in list(worldbody.findall("body")):
    if body is main_body:
      continue
    if body.find("freejoint") is None or not np.allclose(
        [float(x) for x in body.get("pos", "0 0 0").split()], 0):
      raise SystemExit(f"Don't know how to merge body {body.get('name')}")
    print(f"Merging loose part {body.get('name')} into body_main")
    for child in body:
      if child.tag in ("geom",) or child.tag is ET.Comment:
        main_body.insert(len([c for c in main_body if c.tag != "body"]), child)
    worldbody.remove(body)

  # Shift body_main's contents so its frame is at the hip_lat midpoint.
  for child in main_body:
    if child.tag in ("geom", "body", "site", "camera"):
      p = np.array([float(x) for x in child.get("pos", "0 0 0").split()])
      child.set("pos", fmt(p - origin))
  main_body.set("pos", fmt([0, 0, home_z]))
  main_body.set("childclass", "gnoci_v2")
  main_body.insert(0, ET.Element("freejoint", name="floating_base"))
  main_body.insert(1, ET.Element("camera", name="track", mode="trackcom", pos="0.6 -0.6 0.2",
                                 xyaxes="0.7071 0.7071 0 -0.1231 0.1231 0.9847"))
  main_body.insert(2, ET.Element("site", name="imu", pos=fmt(imu_site), group="5"))

  template = ET.parse(DESC / "robot.xml").getroot()
  ranges = {j.get("name"): j.get("range") for j in template.iter("joint")}

  bodies = {b.get("name"): b for b in main_body.iter("body")}
  for side in SIDES:
    for joint in LEG_JOINTS:
      name = f"{side}_{joint}"
      body = bodies[body_of_joint[name]]
      body.set("name", f"{side}_{LEG_BODIES[joint]}")
      j = body.find("joint")
      j.set("class", "sts3215")
      if ranges.get(name):
        j.set("range", ranges[name])
      if np.dot(axis[name], JOINT_AXES[joint]) < 0:
        j.set("axis", fmt(-np.array([float(x) for x in j.get("axis").split()])))
      elif np.dot(axis[name], JOINT_AXES[joint]) < 0.999:
        raise SystemExit(f"{name} isn't along {JOINT_AXES[joint]}: {axis[name]}")

    # Sole spheres and the foot site, in the lower leg's frame.
    leg = bodies[body_of_joint[f"{side}_knee"]]
    leg_id = m.body(body_of_joint[f"{side}_knee"]).id
    lo, hi, centres = soles[side]
    for c, label in zip(centres, ("heel_right", "heel_left", "toe_right", "toe_left")):
      leg.append(ET.Element("geom", name=f"{side}_sole_{label}", type="sphere", **{"class": "foot"},
                            size=f"{r:g}", pos=fmt(to_body(d, leg_id, c))))
    middle = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    leg.append(ET.Element("site", name=f"{SITE_SHORT[side]}_foot",
                          pos=fmt(to_body(d, leg_id, middle)), group="5"))

  # Joint (and so qpos and actuator) order is the body tree's: left leg first.
  for side in SIDES:
    leg = main_body.find(f"body[@name='{side}_hip']")
    main_body.remove(leg)
    main_body.append(leg)

  # Drop collision meshes, inertials and onshape-to-robot's link and joint comments
  # (they use the export's body names); visual geoms get their mass from masses.py.
  for body in [worldbody, main_body, *main_body.iter("body")]:
    for child in list(body):
      if (child.tag == "inertial"
          or (child.tag == "geom" and child.get("class") == "collision")
          or (child.tag is ET.Comment and child.text.strip().startswith(("Link ", "Joint from ")))):
        body.remove(child)
      elif child.tag == "geom" and child.get("class") == "visual":
        child.set("mass", "0")

  # Assemble the new robot.xml: hand-maintained sections from the template.
  out = ET.Element("mujoco", model="gnoci_v2")
  for tag in ("compiler", "option", "default"):
    out.append(copy.deepcopy(template.find(tag)))
  out.append(worldbody)
  out.append(export.find("asset"))
  for tag in ("actuator", "sensor"):
    out.append(copy.deepcopy(template.find(tag)))
  keyframe = ET.SubElement(out, "keyframe")
  ET.SubElement(keyframe, "key", name="home", qpos=fmt([0, 0, home_z, 1, 0, 0, 0] + [0] * 6),
                ctrl=fmt([0] * 6))
  out.append(copy.deepcopy(template.find("equality")))
  ET.indent(out, space="  ")

  header = (
      '<?xml version="1.0" ?>\n'
      f"<!-- Generated from an onshape-to-robot export by scripts/import_onshape.py.\n"
      "     The compiler, option, default, actuator, sensor and equality sections and the\n"
      "     joint ranges are hand-maintained: the script copies them from the previous\n"
      "     robot.xml. Masses come from masses.py (scripts/update_masses.py). -->\n")
  (DESC / "robot.xml").write_text(header + ET.tostring(out, encoding="unicode") + "\n")

  # Replace the meshes.
  used = {mesh.get("file") for mesh in out.iter("mesh")}
  for f in (DESC / "assets").glob("*.stl"):
    f.unlink()
  for name in sorted(used):
    shutil.copy2(args.export / "assets" / name, DESC / "assets" / name)

  print(f"Wrote {DESC / 'robot.xml'} and {len(used)} meshes")
  print(f"  hip_lat axes {np.linalg.norm(anchor['left_hip_lat'] - anchor['right_hip_lat']) * 1000:.1f} mm apart,"
        f" {home_z * 1000:.1f} mm above the soles at home")
  print(f"  soles {(hi[0] - lo[0]) * 1000:.1f} x {(hi[1] - lo[1]) * 1000:.1f} mm, edge radius {r * 1000:g} mm")
  print("Now run scripts/update_masses.py")


if __name__ == "__main__":
  main()
