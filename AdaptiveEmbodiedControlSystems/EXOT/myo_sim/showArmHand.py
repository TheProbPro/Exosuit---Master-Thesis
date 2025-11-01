import mujoco
import mujoco.viewer
import time
import os


MODEL_FILE = "arm/myoarm.xml"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)  # 控制仿真速度

