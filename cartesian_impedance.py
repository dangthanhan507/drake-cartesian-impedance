from pydrake.systems.framework import LeafSystem, Context, ValueProducer
from pydrake.all import JacobianWrtVariable
from pydrake.multibody.plant import MultibodyPlant
from pydrake.common.value import Value
from pydrake.math import RollPitchYaw, RigidTransform, RotationMatrix
from pydrake.common.eigen_geometry import Quaternion, AngleAxis
from pydrake.all import MultibodyForces, SpatialAcceleration
import numpy as np

from pydrake.multibody.inverse_kinematics import InverseKinematics

def quat_diff(quat1: Quaternion, quat2: Quaternion):
    # Quaternion Difference(q1,q2) - yoinked from some stackoverflow post
    if (quat1.wxyz().dot(quat2.wxyz()) < 0.0):
        quat2.set_wxyz(-quat2.wxyz())
    
    error_quaternion = quat2.multiply(quat1.inverse())
    return error_quaternion
    

class CartesianImpedance(LeafSystem):
    def __init__(self, plant_arm: MultibodyPlant, kp: np.ndarray, kd: np.ndarray):
        LeafSystem.__init__(self)
        self._plant = plant_arm
        self._plant_context = plant_arm.CreateDefaultContext()
        self._W = plant_arm.world_frame()
        self._G = plant_arm.GetBodyByName("iiwa_link_7").body_frame()
        
        self._kp = kp
        self._kd = kd
        
        self.DeclareVectorInputPort("desired_pose", 6)
        self.DeclareVectorInputPort("estimated_state", 14)
        self.DeclareVectorOutputPort("tau", 7, self._calc_tau)
        
    def _calc_tau(self, context: Context, output):
        desired_pose = self.get_input_port(0).Eval(context)
        state = self.get_input_port(1).Eval(context)
        
        desired_rot = desired_pose[:3] # in rpy
        desired_xyz = desired_pose[3:]
        
        q, dq = state[:7], state[7:]
        
        self._plant.SetPositionsAndVelocities(self._plant_context, state)
        
        desired_quaternion: Quaternion = RollPitchYaw(desired_rot[0], desired_rot[1], desired_rot[2]).ToQuaternion()
        
        # get ee jacobian
        J_ee = self._plant.CalcJacobianSpatialVelocity(self._plant_context,
                                                        JacobianWrtVariable.kQDot,
                                                        self._G,
                                                        [0,0,0],
                                                        self._W,
                                                        self._W)
        
        # get ee pose
        rigid_transform_ee = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)
        translation_ee = rigid_transform_ee.translation()
        quaternion_ee  = rigid_transform_ee.rotation().ToQuaternion()
        
        M = self._plant.CalcMassMatrixViaInverseDynamics(self._plant_context)
        C = self._plant.CalcBiasTerm(self._plant_context)
        tau_g = self._plant.CalcGravityGeneralizedForces(self._plant_context)
        tau_ext = tau_g + C
        
        
        M_ee = np.linalg.inv(J_ee @ np.linalg.inv(M) @ J_ee.T)
        tau_ext_ee = M_ee @ J_ee @ np.linalg.inv(M) @ tau_ext
        
        qdiff = quat_diff(desired_quaternion, quaternion_ee)
        angle_axis_error = AngleAxis(qdiff)
        rot_error = -1 * angle_axis_error.angle() * angle_axis_error.axis()
        translation_error = desired_xyz - translation_ee
        
        pd_rot = (self._kp[:3] * rot_error) - (self._kd[:3] * (J_ee[:3,:] @ dq))
        pd_t   = (self._kp[3:] * translation_error) - (self._kd[3:] * (J_ee[3:,:] @ dq))
        pd = np.concatenate((pd_rot, pd_t))
        
        
        f_u = pd
        u = J_ee.T @ f_u - tau_g - C
        
        output.SetFromVector(u)