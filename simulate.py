import pickle
import numpy as np
from tqdm import tqdm
import simulator
from simulator import MappingEnv, Mapper, Planner, Viewer
import experiments
from pprint import pprint

# *********** SIMULATION inputPARAMS & outputSCHEMA **********
input_exps = experiments.FG_N1

input_exps_params = input_exps["params"]
input_exps_name = input_exps["name"]

agents_data_schema = np.dtype({'names': ["run_index", "step_index", "agent_id", "position",
                                         "action", "fp_ij", "belief_entropy", "belief_mse",
                                         "agg_belief_entropy", "agg_belief_mse", "belief_var_mean"],
                               'formats': ["i4", "i4", "i4", "(3,)f4", "U5", "(4,2)int", "f4", "f4", "f4", "f4", "f4"]})

output_exps_data = []
for iep_index, iep in enumerate(input_exps_params):
    output_exps_data.append({"input_params":iep,
                             "env":{},
                             "map_ground_truth": np.ndarray,
                             "agents_data":np.zeros(iep["n_runs"]*iep["n_steps"]*iep["n_agents"],
                                                    dtype=agents_data_schema)})


# *********** SIMULATION ***********

for iep_index, iep in enumerate(input_exps_params):

    print(f"\nExp index: {iep_index + 1}\n")
    pprint(iep)

    env = MappingEnv(field_len=50.0,
                     fov=np.pi / 3,
                     **iep)

    mapper = Mapper(env.n_cell,
                    env.min_space_z,
                    env.max_space_z,
                    **iep)

    planner = Planner(env.action_to_direction,
                      env.altitude_to_size,
                      env.position_graph,
                      env.position_to_data,
                      env.regions_limits,
                      env.optimal_altitude,
                      **iep)

    viewer = Viewer(400, 400, env.min_space_z, env.max_space_z, env.n_agents)

    print(env)

    output_exps_data[iep_index]["env"] = {"min_space_z":env.min_space_z,
                                               "max_space_z":env.max_space_z,
                                               "min_field_x":env.min_field_x,
                                               "max_field_x":env.max_field_x,
                                               "min_field_y":env.min_field_y,
                                               "max_field_y":env.max_field_y,
                                               "v_displacement":env.v_displacement,
                                               "h_displacement":env.h_displacement,
                                               }

    agents_data_index = 0
    for run_index in tqdm(range(iep["n_runs"])):

        map_ground_truth = env.generate_map()
        env.reset_map_beliefs()
        env.reset_agents_position(**iep)
        planner.reset_sweep()
        viewer.set_image(map_ground_truth)

        observations = []
        saturated = False
        for step_index in range(iep["n_steps"]):

            if iep["render"]:
                viewer.render(env.agents, step_index, 0)

            planner.compute_map_belief_entropies()

            actions, actions_data = planner.get_actions(env.agents, observations)

            #planner.compute_agg_map_belief()

            for agent in env.agents:

                record = (run_index, step_index, agent.id,
                          agent.state.position, actions[agent.id],
                          np.fromiter(agent.camera.get_fp_vertices_ij(agent.state.position)[0].values(), dtype="(2,)int"),
                          np.sum(simulator.map_belief_entropies[:,:,agent.id]), # metric 1 - total map belief entropy
                          (np.square(map_ground_truth - simulator.map_beliefs[:, :, agent.id])).mean(), # metric 2 - mse between map gt and map belief
                          0,#np.sum(simulator.agg_map_belief_entropy),
                          0,#(np.square(map_ground_truth - simulator.agg_map_belief)).mean()
                          np.var(simulator.map_beliefs, axis=2).mean()
                          )

                output_exps_data[iep_index]["agents_data"][agents_data_index] = record
                agents_data_index+=1

            env.step(actions)

            observations = env.get_observations(map_ground_truth)

            mapper.set_pairwise_potential_z(env.agents, observations)

            mapper.update_map_beliefs(env.agents, observations)

            mapper.update_news_and_fuse_map_beliefs(env.agents, observations)

            if env.saturation():
                if not saturated:
                    saturated = True


file_name = f"{input_exps_name}.pickle"
print(file_name)
with open(file_name, "wb") as file:
    pickle.dump(output_exps_data, file)

