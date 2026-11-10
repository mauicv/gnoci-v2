"""Write the part masses from desc/masses.py to the visual geoms in robot.xml.

MASSES is in grams and keyed by part, without the side prefix or servo number
(left_serv_2_horn -> serv_horn). robot.xml is in kilograms.

Then it writes the torso's <inertial>: its mass, centre of mass and inertia as
computed from its meshes, with TORSO_COM_OFFSET added to the centre of mass.

Usage:
  uv run python scripts/update_masses.py
"""

import argparse
import pathlib
import re

import mujoco
import numpy as np

from gnoci_v2.training.desc import masses

ROBOT_XML = pathlib.Path(masses.__file__).with_name("robot.xml")
SCENE_XML = ROBOT_XML.with_name("scene.xml")  # robot.xml alone doesn't compile: its sensors name the floor.
TORSO = "body_main"
GEOM_MASS = re.compile(r'(<geom\b[^>]*\bmesh="([^"]+)"[^>]*\bmass=")[^"]*(")')
TORSO_OPEN = re.compile(rf'(<body name="{TORSO}"[^>]*>\n)')
TORSO_INERTIAL = re.compile(r' *<!-- Torso inertial, written by scripts/update_masses.py.*?-->\n *<inertial [^>]*/>\n', re.S)


def part_name(mesh: str) -> str:
  name = re.sub(r"^(left|right)_", "", mesh)
  name = re.sub(r"^serv_\d+", "serv", name)
  return re.sub(r"_\d+$", "", name)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xml", type=pathlib.Path, default=ROBOT_XML)
  args = parser.parse_args()

  xml = args.xml.read_text()
  meshes = [match[2] for match in GEOM_MASS.finditer(xml)]
  parts = {part_name(mesh) for mesh in meshes}
  if parts != masses.MASSES.keys():
    raise SystemExit(
        f"Parts in {args.xml} and MASSES differ.\n"
        f"  Not in MASSES: {sorted(parts - masses.MASSES.keys())}\n"
        f"  Not in {args.xml.name}: {sorted(masses.MASSES.keys() - parts)}"
    )

  def replace(match: re.Match) -> str:
    kilograms = masses.MASSES[part_name(match[2])] / 1000
    return f"{match[1]}{kilograms:g}{match[3]}"

  args.xml.write_text(GEOM_MASS.sub(replace, xml))
  total = sum(masses.MASSES[part_name(mesh)] for mesh in meshes) / 1000
  print(f"Updated {len(meshes)} geoms in {args.xml} ({total:g} kg total)")
  write_torso_inertial(args.xml)


def write_torso_inertial(xml_path: pathlib.Path) -> None:
  """Writes the torso's <inertial>: from its meshes, centre of mass + TORSO_COM_OFFSET."""
  xml = TORSO_INERTIAL.sub("", xml_path.read_text())
  xml_path.write_text(xml)
  # Compile without an <inertial>, so MuJoCo computes the torso's from its meshes.
  model = mujoco.MjSpec.from_file(str(xml_path.with_name(SCENE_XML.name))).compile()
  body = model.body(TORSO).id
  mass, iquat, inertia = model.body_mass[body], model.body_iquat[body], model.body_inertia[body]
  ipos = model.body_ipos[body] + np.array(masses.TORSO_COM_OFFSET)
  num = lambda values: " ".join(f"{v:.6g}" for v in values)
  if not TORSO_OPEN.search(xml):
    raise SystemExit(f"No <body name=\"{TORSO}\"> in {xml_path}")
  inertial = (
      "      <!-- Torso inertial, written by scripts/update_masses.py: mass and inertia from the\n"
      "           meshes, centre of mass from the meshes + TORSO_COM_OFFSET"
      f" ({num(masses.TORSO_COM_OFFSET)} m). -->\n"
      f'      <inertial pos="{num(ipos)}" quat="{num(iquat)}" mass="{mass:.6g}" diaginertia="{num(inertia)}" />\n'
  )
  xml_path.write_text(TORSO_OPEN.sub(lambda m: m[1] + inertial, xml, count=1))
  print(f"Torso: {mass:.3f} kg, centre of mass {num(model.body_ipos[body])} -> {num(ipos)} m")


if __name__ == "__main__":
  main()
