# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Slight variant of pick place task.
"""
import numpy as np
from collections import OrderedDict

from robosuite.environments.manipulation.pick_place import PickPlace, PickPlaceCan
from robosuite.utils.placement_samplers import UniformRandomSampler

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class PickPlace_D0(PickPlace):
    """
    Slightly easier task where we limit z-rotation to 0 to 90 degrees for all object initializations (instead of full 360).
    """
    def __init__(self, **kwargs):
        assert "z_rotation" not in kwargs
        super().__init__(
            z_rotation=(0., np.pi / 2.),
            **kwargs,
        )


# === Custom (project) envs: single-can pick-and-place with widened spawn + free z-rotation ===
# Mirrors the Square_RandRot{Mid,Wide} envs in nut_assembly.py, but for robosuite PickPlaceCan.
# PickPlace builds its object sampler internally in _get_placement_initializer (it does NOT accept
# a placement_initializer kwarg like NutAssembly does), so the RandRot variants override that method
# and re-set the collision sampler's x/y range + z-rotation. Uses the MG_Can env interface.
class Can_D0(PickPlaceCan, SingleArmEnv_MG):
    """
    Augment robosuite PickPlaceCan for mimicgen (adds the MG edit_model_xml / grasp / camera hooks).
    """
    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)


class Can_RandRot(Can_D0):
    """
    Base for the RandRot variants. PickPlaceCan always spawns ALL FOUR objects (milk/bread/cereal/can)
    in the source bin -- and the source demos encode all four -- so we cannot drop the distractors.
    Instead we split placement into two samplers: the CAN gets a mid/wide box + free z-rotation
    (the quantity we vary), and the other three keep the default full-bin placement so the scene
    always packs (a single shared sampler over a shrunken box fails with "Cannot place all objects").
    Subclasses set CAN_FRAC (fraction of the bin half-extent the can may spawn over).
    """
    CAN_FRAC = 1.0

    def _get_placement_initializer(self):
        super()._get_placement_initializer()          # builds default collision + visual samplers
        comp = self.placement_initializer
        comp.samplers.pop("CollisionObjectSampler", None)   # drop the all-in-one collision sampler

        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05
        can = self.objects[self.object_to_id["can"]]
        others = [o for i, o in enumerate(self.objects) if i != self.object_to_id["can"]]

        can_sampler = UniformRandomSampler(
            name="CanSampler",
            mujoco_objects=can,
            x_range=[-bin_x_half * self.CAN_FRAC, bin_x_half * self.CAN_FRAC],
            y_range=[-bin_y_half * self.CAN_FRAC, bin_y_half * self.CAN_FRAC],
            rotation=(0., 2. * np.pi),                 # free rotation (the varied quantity)
            rotation_axis="z",
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.bin1_pos,
            z_offset=self.z_offset,
        )
        other_sampler = UniformRandomSampler(
            name="OtherObjectSampler",
            mujoco_objects=others,
            x_range=[-bin_x_half, bin_x_half],         # distractors keep the full-bin placement
            y_range=[-bin_y_half, bin_y_half],
            rotation=self.z_rotation,
            rotation_axis="z",
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.bin1_pos,
            z_offset=self.z_offset,
        )
        # can first (so distractors are placed avoiding it), then the original visual-object samplers
        new_samplers = OrderedDict()
        new_samplers["CanSampler"] = can_sampler
        new_samplers["OtherObjectSampler"] = other_sampler
        new_samplers.update(comp.samplers)
        comp.samplers = new_samplers


class Can_RandRotWide(Can_RandRot):
    """Can spawned over the FULL source bin + free z-rotation. OOD eval (analogue of Square_RandRotWide)."""
    CAN_FRAC = 1.0


class Can_RandRotMid(Can_RandRot):
    """Can spawned over a HALF-WIDTH central box + free z-rotation. Training distribution (analogue of Square_RandRotMid)."""
    CAN_FRAC = 0.5
