import os
import time
import numpy as np
import torch
import argparse
import yaml
from tqdm import tqdm
import redsdf
from redsdf.envs.apf import GoToAPF, CollisionAvoidanceAPF, HriController
from redsdf.envs.dist_field import SmplDistField, TableDistField, HriDistFieldSphere
from redsdf.envs.human_robot_interaction.human_robot_env import HumanRobotEnv


def manifold_controller(env, control_frequency, dist_field='manifold',
                        max_action_human=1.5, dist_effect_human=0.15,
                        max_action_table=1.5, dist_effect_table=0.05, use_cuda=False):
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

    goto_apf = GoToAPF(env.robot.kinematics, frame_ids=[102], max_action=2.0, stiffness=20, damping=0.5, i_gain=5,
                       perturb=False)
    if dist_field == 'manifold':
        poi_config_file = os.path.dirname(redsdf.package_dir) + "/reactive_control/yamls/point_config_hri.yaml"
        with open(poi_config_file) as f:
            poi_config = yaml.load(f, Loader=yaml.FullLoader)

        smpl_model_file = os.path.dirname(redsdf.package_dir) + "/trained_sdf/human.pt"
        smpl_manifold_model = torch.load(smpl_model_file, map_location=device)
        if not hasattr(smpl_manifold_model.nn_model, "radius"):
            smpl_manifold_model.nn_model.__setattr__("radius", 0.)
        smpl_dist_field = SmplDistField(env.robot.kinematics, smpl_manifold_model, poi_config, device=device)

        cube_model_file = os.path.dirname(redsdf.package_dir) + "/trained_sdf/cube.pt"
        smpl_manifold_model = torch.load(cube_model_file)
        cube_dist_field = TableDistField(env.robot.kinematics, smpl_manifold_model, poi_config, device=device)

        human_avoid_apf = CollisionAvoidanceAPF(env.robot.kinematics, [smpl_dist_field], max_action=max_action_human,
                                                dist_effect=dist_effect_human, device=device)
        table_avoid_apf = CollisionAvoidanceAPF(env.robot.kinematics, [cube_dist_field], max_action=max_action_table,
                                                dist_effect=dist_effect_table, device=device)
        apf_controller = HriController(env.robot.kinematics, [goto_apf, human_avoid_apf, table_avoid_apf],
                                       step_size=1 / control_frequency)
    elif dist_field == 'sphere':
        poi_config_file = os.path.dirname(redsdf.package_dir) + "/reactive_control/yamls/tiago_hri_sphere.yaml"
        with open(poi_config_file) as f:
            poi_config = yaml.load(f, Loader=yaml.FullLoader)

        sphere_dist_field = HriDistFieldSphere(env.robot.kinematics, poi_config, device)
        sphere_apf = CollisionAvoidanceAPF(env.robot.kinematics, [sphere_dist_field], max_action=max_action_human,
                                           dist_effect=dist_effect_human, device=device)
        apf_controller = HriController(env.robot.kinematics, [goto_apf, sphere_apf], step_size=1 / control_frequency)
    elif dist_field == 'None' or dist_field is None:
        apf_controller = HriController(env.robot.kinematics, [goto_apf], step_size=1 / control_frequency)
    else:
        raise ValueError("Unknown dist field")

    return apf_controller


def experiment(dist_field: str = 'manifold',
               max_action_human: float = 1.5,
               dist_effect_human: float = 0.15,
               max_action_table: float = 1.5,
               dist_effect_table: float = 0.05,
               debug_gui: bool = False,
               use_cuda: bool = False,
               seed: int = 0,
               results_dir: str = './logs'
               ):
    print("start experiment:", locals())
    results_dir = os.path.join(results_dir, str(seed))
    os.makedirs(results_dir, exist_ok=True)
    control_frequency = 60.
    n_intermediate_steps = np.ceil(240. / control_frequency).astype(int)

    np.random.seed(seed)
    gui = 'pyrender' if debug_gui else None

    visualize_smpl = False
    visualize_pcl = False
    if gui == 'pyrender':
        visualize_smpl = True
        visualize_pcl = False

    env = HumanRobotEnv(n_intermediate_steps=n_intermediate_steps, gui=gui, control_mode='position',
                        visualize_smpl=visualize_smpl, visualize_pcl=visualize_pcl)
    exp_time = 30.

    targets = np.load(os.path.dirname(redsdf.package_dir) + "/object_models/hri_record/targets.npy")

    controller = manifold_controller(env, control_frequency, dist_field=dist_field,
                                     max_action_human=max_action_human, dist_effect_human=dist_effect_human,
                                     max_action_table=max_action_table, dist_effect_table=dist_effect_table,
                                     use_cuda=use_cuda)

    save_dir = os.path.join(results_dir, "exp")
    try:
        experiment_logs = {'targets': list(),
                           'collisions_table': list(),
                           'collisions_human': list(),
                           'target_reached_count': list(),
                           'smoothness': list(),
                           'reach_time': list(),
                           'computation_time': list()}
        for target in tqdm(targets):
            has_collision = False
            has_reach = False
            reach_count = 0
            reach_time = exp_time
            smoothness_sum = 0.
            prev_vel = np.zeros(controller.action_dim)
            computation_time = 0.

            state = env.reset()
            target_idx = 0
            env.set_target([target[target_idx]])
            controller.set_target([target[target_idx]])
            for i in range(int(exp_time * control_frequency)):
                t_start = time.time()
                best_action = controller.update(state)
                computation_time += time.time() - t_start

                state = env.step(best_action)

                n_collisions_table = len(env.client.getContactPoints(env.robot.model_id, env.table))
                n_collisions_human = 0
                for human_model_id in env.human.models:
                    n_collisions_human += len(env.client.getContactPoints(env.robot.model_id, human_model_id))
                if n_collisions_human + n_collisions_table > 0:
                    print("collisions!")
                    has_collision = True
                    break

                smoothness_sum += np.sum(((controller.control_vel - prev_vel) * control_frequency) ** 2)
                prev_vel = controller.control_vel
                # Reach Target Criterion:
                if not has_reach:
                    dist = env.check_result()
                    if np.all(dist < 0.03):
                        reach_count += 1
                    else:
                        reach_count = 0
                    if reach_count > 30:
                        target_idx += 1
                        if target_idx < target.shape[0]:
                            env.set_target([target[target_idx]])
                            controller.set_target([target[target_idx]])
                        else:
                            has_reach = True
                            reach_time = i / control_frequency

            # Check Result
            experiment_logs['collisions_human'].append(n_collisions_human)
            experiment_logs['collisions_table'].append(n_collisions_table)
            experiment_logs['targets'].append(target)
            experiment_logs['target_reached_count'].append(target_idx)
            experiment_logs['reach_time'].append(reach_time)
            experiment_logs['smoothness'].append(smoothness_sum / (i + 1))
            experiment_logs['computation_time'].append(computation_time / (i + 1))

        experiment_logs['collisions_human'] = np.array(experiment_logs['collisions_human'])
        experiment_logs['collisions_table'] = np.array(experiment_logs['collisions_table'])
        experiment_logs['targets'] = np.array(experiment_logs['targets'])
        experiment_logs['target_reached_count'] = np.array(experiment_logs['target_reached_count'])
        experiment_logs['reach_time'] = np.array(experiment_logs['reach_time'])
        experiment_logs['smoothness'] = np.array(experiment_logs['smoothness'])
        experiment_logs['computation_time'] = np.array(experiment_logs['computation_time'])
        np.savez(save_dir, **experiment_logs)

    except KeyboardInterrupt:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_field', type=str, default="manifold")
    parser.add_argument('--max_action_human', type=float, default=1.5)
    parser.add_argument('--dist_effect_human', type=float, default=0.15)
    parser.add_argument('--max_action_table', type=float, default=1.5)
    parser.add_argument('--dist_effect_table', type=float, default=0.05)
    parser.add_argument('--debug_gui', action="store_true", default=False)
    parser.add_argument('--use_cuda', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default="./logs")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
