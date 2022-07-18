import time
import os
import numpy as np
import torch
import yaml
from tqdm import tqdm
import argparse
import redsdf
from redsdf.envs.apf import TiagoAPFController, GoToAPF, CollisionAvoidanceAPF
from redsdf.envs.dist_field import TiagoDistFieldPoI, TiagoDistFieldSphere
from redsdf.envs.whole_body_control.tiago_env import TiagoEnv


def manifold_controller(env, control_frequency, dist_field='manifold'):
    device = 'cuda'
    if dist_field == 'manifold':
        poi_config_file = os.path.dirname(redsdf.package_dir) + "/reactive_control/yamls/query_point_config.yaml"
        with open(poi_config_file) as f:
            poi_config = yaml.load(f, Loader=yaml.FullLoader)

        manifold_model_file = os.path.dirname(redsdf.package_dir) + "/object_models/tiago.pt"
        tiago_manifold_model = torch.load(manifold_model_file)
        if not hasattr(tiago_manifold_model.nn_model, "radius"):
            tiago_manifold_model.nn_model.__setattr__("radius", 0.)
        tiago_dist_field = TiagoDistFieldPoI(env.robot.kinematics, tiago_manifold_model, poi_config, device=device)
        collision_avoid_apf = CollisionAvoidanceAPF(env.robot.kinematics, [tiago_dist_field], max_action=0.8,
                                                    dist_effect=0.1, device=device)
    elif dist_field == 'sphere':
        poi_config_file = os.path.dirname(redsdf.package_dir) + "/reactive_control/yamls/tiago_sphere_config.yaml"
        with open(poi_config_file) as f:
            poi_config = yaml.load(f, Loader=yaml.FullLoader)

        tiago_dist_field = TiagoDistFieldSphere(env.robot.kinematics, poi_config, device=device)
        collision_avoid_apf = CollisionAvoidanceAPF(env.robot.kinematics, [tiago_dist_field], max_action=2.0,
                                                    dist_effect=0.2, device=device)
    else:
        raise ValueError("Unknown dist field")

    goto_apf = GoToAPF(env.robot.kinematics, max_action=2, stiffness=20, damping=1, i_gain=5, perturb=False)
    apf_controller = TiagoAPFController(env.robot.kinematics, [goto_apf, collision_avoid_apf],
                                        step_size=1 / control_frequency)
    return apf_controller


def main(dist_field, results_dir, debug_gui, seed):
    control_frequency = 60.
    n_intermediate_steps = np.ceil(240. / control_frequency).astype(int)
    np.random.seed(0)
    gui = None
    if debug_gui:
        gui = "default"
    env = TiagoEnv(n_intermediate_steps=n_intermediate_steps, gui=gui, control_mode='position')
    exp_time = 30.
    targets = [env.sample_target() for _ in range(1000)]
    controller = manifold_controller(env, control_frequency, dist_field=dist_field)

    save_dir = results_dir
    try:
        experiment_logs = {'targets': list(),
                           'collisions': list(),
                           'final_distance': list(),
                           'smoothness': list(),
                           'reach_time': list(),
                           'computation_time': list(),
                           'trajectory': list()}
        for target in tqdm(targets):
            has_collision = False
            has_reach = False
            reach_count = 0
            reach_time = exp_time
            smoothness_sum = 0.
            prev_vel = np.zeros(controller.action_dim)
            computation_time = 0.

            state = env.reset()
            controller.reset()
            env.set_targets(target)
            controller.set_target(target)
            traj_i = list()
            for i in range(int(exp_time * control_frequency)):
                t_start = time.time()
                best_action = controller.update(state)
                computation_time += time.time() - t_start

                action = env.wrap_control_action(torso=best_action[0], l_arm=best_action[1:8],
                                                 r_arm=best_action[10:17])
                state = env.step(action)

                if not has_collision and len(env.client.getContactPoints(env.robot.model_id, env.robot.model_id)) > 0:
                    print(env.client.getContactPoints(env.robot.model_id, env.robot.model_id))
                    has_collision = True
                    break

                traj_i.append(np.concatenate([action, controller.control_vel]).astype(np.single))
                smoothness_sum += np.sum(((controller.control_vel - prev_vel) * control_frequency) ** 2)
                prev_vel = controller.control_vel
                # Reach Target Criterion:
                if not has_reach:
                    dist = env.check_result()
                    if np.all(dist < 0.03):
                        reach_count += 1
                    else:
                        reach_count = 0
                    if reach_count > 10:
                        has_reach = True
                        reach_time = (i+1) / control_frequency

            # Check Result
            experiment_logs['collisions'].append(has_collision)
            experiment_logs['targets'].append(env.targets)
            experiment_logs['final_distance'].append(env.check_result())
            experiment_logs['reach_time'].append(reach_time)
            experiment_logs['smoothness'].append(smoothness_sum / (i+1))
            experiment_logs['computation_time'].append(computation_time / (i+1))
            experiment_logs['trajectory'].append(traj_i)

        experiment_logs['targets'] = np.array(experiment_logs['targets'])
        experiment_logs['final_distance'] = np.array(experiment_logs['final_distance'])
        experiment_logs['reach_time'] = np.array(experiment_logs['reach_time'])
        experiment_logs['smoothness'] = np.array(experiment_logs['smoothness'])
        experiment_logs['trajectory'] = np.array(experiment_logs['trajectory'])
        np.savez(save_dir, **experiment_logs)

    except KeyboardInterrupt:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_field', type=str, default="manifold")
    parser.add_argument('--results_dir', type=str, default="./logs")
    parser.add_argument('--debug_gui', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    main(**args)