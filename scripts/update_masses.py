"""Write the part masses from desc/masses.py to the visual geoms in robot.xml.

MASSES is in grams and keyed by part, without the side prefix or servo number
(left_serv_2_horn -> serv_horn). robot.xml is in kilograms.

Usage:
  uv run python scripts/update_masses.py
"""

import argparse
import pathlib
import re

from gnoci_v2.training.desc import masses

ROBOT_XML = pathlib.Path(masses.__file__).with_name("robot.xml")
GEOM_MASS = re.compile(r'(<geom\b[^>]*\bmesh="([^"]+)"[^>]*\bmass=")[^"]*(")')


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


if __name__ == "__main__":
  main()
