import numpy as np
import os
import glob
import re
from pathlib import Path

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    RigidTransform,
    StartMeshcat,
    MeshcatVisualizer,
)

ROBOT_FAKE_HAND_PATH = "models/g1/g1_29dof.urdf"
ROBOT_SPHERE_HAND_PATH = "models/g1/g1_29dof_spherehand.urdf"
LARGEBOX_PATH = "models/largebox/largebox.urdf"
CHAIR_PATH = "models/chair/chair.urdf"


def create_plant(
    robot_model_path: str, object_model_path: str | list[str] | None = None
):
    builder = DiagramBuilder()
    plant = MultibodyPlant(1e-3)
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)
    parser.AddModels(robot_model_path)
    if object_model_path is not None:
        if isinstance(object_model_path, list):
            parser.SetAutoRenaming(True)
            for model_path in object_model_path:
                parser.AddModels(model_path)
        else:
            parser.AddModels(object_model_path)
    parser.AddModels("models/ground_box.sdf")
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("ground_link"),
        RigidTransform(np.array([0, 0, -0.5])),
    )
    plant.Finalize()
    meshcat = StartMeshcat()
    vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    return plant, vis, diagram


def draw_q_knots(vis, plant, diagram, q_knots, fps):
    vis.DeleteRecording()
    vis.StartRecording()
    t_knots = np.arange(len(q_knots)) * fps
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    vis_context = vis.GetMyMutableContextFromRoot(context)
    for t, q in zip(t_knots, q_knots):
        context.SetTime(t)
        plant.SetPositions(plant_context, q)
        vis.ForcedPublish(vis_context)
    vis.StopRecording()
    vis.PublishRecording()


def _natural_sort_key(text: str):
    parts = re.findall(r"\d+|\D+", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def find_files(base_path: str, filter: str = "", extension: str = ".npz"):
    pattern = os.path.join(base_path, f"*{filter}*{extension}")
    files = glob.glob(pattern)
    return sorted(files, key=lambda p: _natural_sort_key(os.path.basename(p)))


def visualize_robot_object():
    robot_object_files = find_files("robot-object", filter="original")
    plant, vis, diagram = create_plant(ROBOT_FAKE_HAND_PATH, LARGEBOX_PATH)
    for robot_object_file in robot_object_files:
        print(str(Path(robot_object_file).stem))
        data = np.load(robot_object_file)
        q_knots = data["qpos"]
        fps = data["fps"]
        draw_q_knots(vis, plant, diagram, q_knots, 1 / fps)
        input()


def visualize_robot_terrain():
    robot_terrain_files = find_files("robot-terrain", filter="z_scale_1.0")
    for robot_terrain_file in robot_terrain_files:
        file_name = str(Path(robot_terrain_file).stem)
        print(file_name)
        object_folder_name = "models/terrain/" + file_name[:8]
        z_scale = file_name[8:]
        object_model_path = object_folder_name + "/multi_boxes" + z_scale + ".urdf"
        plant, vis, diagram = create_plant(ROBOT_SPHERE_HAND_PATH, object_model_path)
        data = np.load(robot_terrain_file)
        q_knots = data["qpos"]
        fps = data["fps"]
        draw_q_knots(vis, plant, diagram, q_knots, 1 / fps)
        input()


def visualize_robot_object_terrain():
    robot_object_terrain_files = find_files("robot-object-terrain")
    for robot_object_terrain_file in robot_object_terrain_files:
        file_name = str(Path(robot_object_terrain_file).stem)
        print(file_name)
        terrain_folder_name = "models/terrain/" + file_name[:8]
        if "z_scale" in file_name:
            z_scale = file_name[-12:]
        else:
            z_scale = "_z_scale_1.0"
        terrain_model_path = terrain_folder_name + "/multi_boxes" + z_scale + ".urdf"
        if "original" in file_name:
            object_model_path = [CHAIR_PATH, terrain_model_path]
        else:
            chair_scale = file_name[9:25]
            chair_model_path = "models/chair/" + chair_scale + ".urdf"
            object_model_path = [chair_model_path, terrain_model_path]
        plant, vis, diagram = create_plant(ROBOT_SPHERE_HAND_PATH, object_model_path)
        data = np.load(robot_object_terrain_file)
        q_knots = data["qpos"]
        fps = data["fps"]
        draw_q_knots(vis, plant, diagram, q_knots, 1 / fps)
        input()


if __name__ == "__main__":
    task = "object-terrain"
    if task == "object":
        visualize_robot_object()
    elif task == "terrain":
        visualize_robot_terrain()
    elif task == "object-terrain":
        visualize_robot_object_terrain()
