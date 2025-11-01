import unittest
import os
import mujoco
import mujoco.viewer
import time
import numpy as np
from pathlib import Path

model_paths = [
    # Basic models
    "basic/myomuscle.xml",

    # finger models
    "finger/finger_v0.xml",
    "finger/myofinger_v0.xml", 
    "finger/motorfinger_v0.xml",

    # elbow models
    "elbow/myoelbow_1dof6muscles_1dofexo.xml",
    "elbow/myoelbow_1dof6muscles.xml",
    "elbow/myoelbow_2dof6muscles.xml",
    "elbow/myoelbow_1dof6muscles_1dofSoftexo_Ideal.xml",
    "elbow/myoelbow_1dof6muscles_1dofSoftexo_sim2.xml",

    # arms
    "arm/myoarm_simple.xml",
    "arm/myoarm.xml",

    # hand models
    "hand/myohand.xml",

    # leg models
    "leg/myolegs.xml",
    "leg/myolegs_abdomen.xml",
    "osl/myolegs_osl.xml",

    # head
    "head/myohead_simple.xml",

    # torso
    "torso/myotorso.xml",
    "torso/myotorso_exosuit.xml",
    "torso/myotorso_rigid.xml",
    "torso/myotorso_abdomen.xml",

    # full body models
    "body/myobody.xml",
    "body/myoupperbody.xml",

    # scene
    "scene/myosuite_scene_noPedestal.xml",
    "scene/myosuite_scene.xml",
    "scene/myosuite_quad.xml",
    "scene/myosuite_logo.xml",
]

class MyoSuiteSimRunner:
    """运行MyoSuite仿真的类"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.current_model = None
        self.current_data = None
        self.viewer = None
    
    def get_sim(self, model_path: str):
        """加载模型"""
        # 解析完整路径
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(self.base_dir, model_path)
        
        if not os.path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")

        print(f"Loading: {fullpath}")

        # 加载模型
        if model_path.endswith(".mjb"):
            model = mujoco.MjModel.from_binary_path(fullpath)
        elif model_path.endswith(".xml"):
            model = mujoco.MjModel.from_xml_path(fullpath)
        else:
            raise ValueError("Unsupported model format")

        data = mujoco.MjData(model)
        return model, data

    def run_simulation(self, model_path, simulation_time=5.0):
        """运行单个仿真"""
        try:
            print(f"\n{'='*60}")
            print(f"Starting simulation: {model_path}")
            print(f"{'='*60}")
            
            # 加载模型
            model, data = self.get_sim(model_path)
            self.current_model = model
            self.current_data = data
            
            print(f"✅ Model loaded successfully!")
            print(f"   Geometry count: {model.ngeom}")
            print(f"   Joint count: {model.nq}")
            print(f"   Muscle count: {model.nu if hasattr(model, 'nu') else 'N/A'}")
            print(f"   Simulation time: {simulation_time} seconds")
            
            # 运行仿真
            with mujoco.viewer.launch_passive(model, data) as viewer:
                self.viewer = viewer
                print("🎮 Simulation started! Press Ctrl+C to stop early.")
                
                start_time = time.time()
                step_count = 0
                
                while time.time() - start_time < simulation_time:
                    if not viewer.is_running():
                        break
                    
                    # 执行仿真步骤
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    
                    step_count += 1
                    time.sleep(0.001)  # 控制仿真速度
                
                print(f"Simulation completed! Steps: {step_count}")
                
        except Exception as e:
            print(f"❌ Error running simulation {model_path}: {e}")
            return False
        
        return True

    def run_all_simulations(self, simulation_time=3.0):
        """运行所有仿真"""
        successful = 0
        failed = 0
        failed_files = []
        
        for model_path in model_paths:
            try:
                if self.run_simulation(model_path, simulation_time):
                    successful += 1
                    print(f"✅ SUCCESS: {model_path}")
                else:
                    failed += 1
                    failed_files.append(model_path)
                    print(f"❌ FAILED: {model_path}")
            
            except KeyboardInterrupt:
                print("\n⏹️ Simulation interrupted by user")
                break
            except Exception as e:
                failed += 1
                failed_files.append(model_path)
                print(f"❌ ERROR: {model_path} - {e}")
            
            # 在仿真之间添加短暂暂停
            time.sleep(1.0)
        
        # 输出汇总结果
        print(f"\n{'='*60}")
        print("SIMULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if failed_files:
            print("\nFailed simulations:")
            for file in failed_files:
                print(f"  - {file}")

    def run_specific_simulation(self, model_path, simulation_time=10.0):
        """运行特定的仿真"""
        return self.run_simulation(model_path, simulation_time)

def main():
    """主函数"""
    print("🤖 MyoSuite Simulation Runner")
    print("=" * 50)
    
    runner = MyoSuiteSimRunner()
    
    while True:
        print("\nOptions:")
        print("1. Run all simulations (quick test)")
        print("2. Run all simulations (full test)") 
        print("3. Run specific simulation")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            # 快速测试，每个仿真运行2秒
            runner.run_all_simulations(simulation_time=2.0)
            
        elif choice == "2":
            # 完整测试，每个仿真运行5秒
            runner.run_all_simulations(simulation_time=5.0)
            
        elif choice == "3":
            # 运行特定仿真
            print("\nAvailable models:")
            for i, path in enumerate(model_paths, 1):
                print(f"{i:2d}. {path}")
            
            try:
                selection = int(input("Enter model number: "))
                if 1 <= selection <= len(model_paths):
                    sim_time = float(input("Simulation time (seconds): ") or "10.0")
                    runner.run_specific_simulation(model_paths[selection-1], sim_time)
                else:
                    print("Invalid selection!")
            except ValueError:
                print("Please enter a valid number!")
                
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice!")

# 如果您还想保留单元测试
class TestSims(unittest.TestCase):
    def get_sim(self, model_path: str = None):
        """测试用的加载方法"""
        runner = MyoSuiteSimRunner()
        if model_path:
            fullpath = os.path.join(os.path.dirname(__file__), model_path)
            if not os.path.exists(fullpath):
                raise IOError(f"File {fullpath} does not exist")
            
            if model_path.endswith(".mjb"):
                return mujoco.MjModel.from_binary_path(fullpath)
            elif model_path.endswith(".xml"):
                return mujoco.MjModel.from_xml_path(fullpath)

    def test_sims(self):
        """测试所有仿真文件是否能加载"""
        successful = 0
        for model_path in model_paths:
            print(f"Testing: {model_path}")
            try:
                model = self.get_sim(model_path)
                if model:
                    print(f"✅ Load successful")
                    successful += 1
                else:
                    print(f"❌ Load failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\nLoad success rate: {successful}/{len(model_paths)}")

if __name__ == '__main__':
    # 如果要运行交互式仿真，取消下面的注释
    main()
    
    # 如果要运行单元测试，取消下面的注释
    # unittest.main(verbosity=2)