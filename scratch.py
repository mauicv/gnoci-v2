import argparse
import mujoco
import mujoco.viewer

import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from gnoci_v2.training import constants as consts
from gnoci_v2.training.base import get_assets
from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict



if __name__ == '__main__':
    xml_path = consts.SIM_PATH / "desc" / "scene.xml"
    _model_assets = get_assets()
    _mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=_model_assets
    )
    _mj_model.opt.timestep = 0.001

    _mj_model.vis.global_.offwidth = 3840
    _mj_model.vis.global_.offheight = 2160

    _mjx_model = mjx.put_model(_mj_model)
    _xml_path = xml_path

    data = mujoco.MjData(_mj_model)
    mujoco.viewer.launch(_mj_model, data)
