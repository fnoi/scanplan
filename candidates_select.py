import pickle
import open3d as o3d
import multiprocessing as mp
import os

from tqdm import tqdm

from tools.greedy import greedy_search
from tools import timer
from tools.utils import load_model
from tools.utils_nongeom import gather_config
from tools.TSP import get_route_new_graph
from tools.evolutionary_algorithm import evolutionary_algorithm_search


def candidates_select():
    print(f'\n[start - candidates_select ]')
    t = timer.Timer()
    t.start()
    mp_threads = os.cpu_count()

    config = gather_config(config_list=['candidates_create.json', 'candidates_assess.json', 'candidates_select.json'])
    t.report_lap(achieved='config load complete')

    model = load_model(config)
    t.report_lap(achieved='model load complete')

    # read candidates from graph #TODO: slim down / modularize loader
    with open('handover/1_candidates.pkl', 'rb') as f:
        candidates = pickle.load(f)
    with open('handover/0_candidate_graph.pkl', 'rb') as f:
        candidate_graph = pickle.load(f)
    t.report_lap(achieved='candidates load complete')

    with open('handover/1_visibility_table.pkl', 'rb') as f:
        visibility_table = pickle.load(f)
    t.report_lap(achieved='visibility table load complete')

    with open('handover/1_overlap_table_relative.pkl', 'rb') as jar:
        overlap_table_relative = pickle.load(jar)
    t.report_lap(achieved='overlap load complete')

    with open('handover/1_model.pkl', 'rb') as jar:
        model = pickle.load(jar)
    t.report_lap(achieved='model')

    # get scene for corner cutting check
    mesh_scene = o3d.io.read_triangle_mesh(config['model file'])
    mesh_scene = o3d.t.geometry.TriangleMesh.from_legacy(mesh_scene)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_scene)  # required?

    if config['strategy option'] == 'greedy':
        strategy = greedy_search(mask_visibility=visibility_table,
                                 table_overlap=overlap_table_relative,
                                 candidate_graph=candidate_graph,
                                 candidates=candidates,
                                 config=config,
                                 model=model,
                                 n=1,
                                 report=False)

    elif config['strategy option'].lower() == 'ea' or config['strategy option'].lower() == 'evolutionary algorithm':
        strats = []
        # give ea an better shot with more options: all greedy starting options
        # with mp.Pool(processes=mp_threads) as pool:
        #     strat = list(
        #         tqdm(
        #             pool.imap(
        #                 greedy_search, [v for v in range(len(candidates))],
        #                 visibility_table, overlap_table_relative, config, model,
        #                 table_overlap=None, candidate_graph=None,
        #                 report=False, n=1, report_prec=2
        #             ),
        #             desc="gathering greedy solutions",
        #             total=len(candidates)
        #         )
        #     )

        # for variant in tqdm(range(len(candidates)), desc="gathering greedy solutions"):
        for variant in tqdm(range(10), desc="gathering greedy solutions"):
            strat = greedy_search(mask_visibility=visibility_table,
                                  table_overlap=overlap_table_relative,
                                  candidate_graph=candidate_graph,
                                  candidates=candidates,
                                  config=config,
                                  model=model,
                                  n=variant,
                                  report=False)
            if variant == 0:
                pickle.dump(strat,
                            open(f"{config['project path']}{config['work dir']}"
                                 f"2_greedy_strategy.pkl", "wb")
                            )
                config['first greedy coverage'] = strat['coverage achieved absolute']
            strats.append(strat['scanpoint ids'])

        # here start ea
        new_strat, cover = evolutionary_algorithm_search(
            config=config,
            overlap_relative=overlap_table_relative,
            model=model,
            strategy=strats,
            candidates=candidates,
            candidate_graph=candidate_graph
        )
        strategy = {'scanpoint ids': new_strat,
                    'coverage achieved relative': round(cover / candidates[0]['max coverage area'], 2),
                    'coverage achieved absolute': round(cover, 2)}

    if config['TSP']:
        strategy['route'], strategy['full_path'] = get_route_new_graph(
            strategy_ptids=strategy['scanpoint ids'],
            neighborhood_graph=candidate_graph,
            scene=scene
        )

    # maybe change the name to 2_strategy?
    pickle.dump(strategy,
                open(f"{config['project path']}{config['work dir']}"
                     f"2_{config['strategy option']}_strategy.pkl", "wb")
                )

    print(
        f"EA: {(strategy['coverage achieved absolute'], len(strategy['scanpoint ids']))}"
        f"\ngreedy: {(config['first greedy coverage'], len(strats[0]))}")
    print(f'[  end - candidates_select ]')


if __name__ == "__main__":
    candidates_select()
