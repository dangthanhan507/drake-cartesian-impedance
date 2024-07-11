import numpy as np
from pydrake.multibody.parsing import LoadModelDirectives, ProcessModelDirectives
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig, AddMultibodyPlant
from pydrake.geometry import SceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import SnoptSolver
from pydrake.systems.framework import DiagramBuilder
from pydrake.all import StartMeshcat, PiecewisePose, Quaternion, PiecewisePolynomial, TrajectorySource, LeafSystem, Context, ConstantVectorSource
from pydrake.systems.analysis import Simulator
from pydrake.visualization import AddDefaultVisualization

from cartesian_impedance import CartesianImpedance
from typing import Tuple
def load_iiwa_setup(plant: MultibodyPlant, scene_graph: SceneGraph = None):
    parser = Parser(plant, scene_graph)
    directive_path = "iiwa.yaml"
    directives = LoadModelDirectives(directive_path)
    models = ProcessModelDirectives(directives=directives, plant=plant, parser=parser)
    
if __name__ == '__main__':
    meshcat = StartMeshcat()
    
    config = MultibodyPlantConfig()
    config.time_step = 1e-3
    config.penetration_allowance = 1e-9
    
    builder = DiagramBuilder()
    plant_scenegraph = AddMultibodyPlant(config, builder)
    plant: MultibodyPlant = plant_scenegraph[0]
    scene_graph: SceneGraph = plant_scenegraph[1]
    load_iiwa_setup(plant, scene_graph)
    plant.Finalize()
    
    JOINT0 = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, 0.0, 0.0])
    ENDTIME = 10.0
    
    plant_context = plant.CreateDefaultContext()
    plant.SetPositions(plant_context, JOINT0)
    ee_pose0 = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetBodyByName("iiwa_link_7").body_frame())
    
    # rpy_desired_rot = RollPitchYaw(ee_pose0.rotation()).vector()
    rpy_desired_rot = RollPitchYaw(0, np.pi, 0).vector()
    desired_trans = ee_pose0.translation() - np.array([0.2, 0.0, 0.3])
    desired_pose = np.hstack((rpy_desired_rot, desired_trans))
    
    kp = np.array([70.0, 70.0, 70.0, 1000.0, 1000.0, 1000.0])
    kd = np.array([20.0, 20.0, 20.0, 500.0, 500.0, 500.0])
    controller_block = builder.AddSystem(CartesianImpedance(plant, kp, kd))
    desired_pose_block = builder.AddSystem(ConstantVectorSource(desired_pose))
    
    builder.Connect(desired_pose_block.get_output_port(), controller_block.get_input_port(0))
    builder.Connect(plant.get_state_output_port(), controller_block.get_input_port(1))
    builder.Connect(controller_block.get_output_port(), plant.get_actuation_input_port())
    
    AddDefaultVisualization(builder, meshcat)
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    plant.SetPositions(plant_context, JOINT0)
    simulator.Initialize()
    simulator.set_target_realtime_rate(10000.0)
    meshcat.StartRecording()
    simulator.AdvanceTo(ENDTIME)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    
    input()