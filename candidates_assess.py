import os
import numpy as np
import pickle
import multiprocessing as mp

from tqdm import tqdm

from tools import timer
from tools import parallel
from tools.utils_nongeom import gather_config
from tools.utils import load_model, model2scene, overlap_wrap, table_base_v2
from tools.helpers import graph2candidates

from tools.raycast import cast_plus_eval, final_eval


def candidates_assess():
    print(f'\n[start - candidates_assess ]')
    t = timer.Timer()
    t.start()

    config = gather_config(config_list=['candidates_create.json', 'candidates_assess.json'])

    t.report_lap(achieved='config load complete')

    # read model
    model = load_model(config)
    t.report_lap(achieved='model load complete')

    # read candidates from graph
    with open('handover/0_candidate_graph.pkl', 'rb') as jar:
        neighborhood_graph = pickle.load(jar)
    candidates = graph2candidates(config=config, graph=neighborhood_graph)
    t.report_lap(achieved='candidates load complete')

    raw_arrays = parallel.init_arrays(model=model, candidates=candidates)
    mp_threads = os.cpu_count()
    parallel.config = config

    # compute all rays
    with mp.Pool(
            processes=mp_threads,
            initializer=parallel.rayprep_init,
            initargs=(raw_arrays["midpoints"], raw_arrays["midpoints shape"],
                      raw_arrays["candidates"], raw_arrays["candidates shape"])
    ) as pool:
        ray = list(
            tqdm(
                pool.imap(
                    parallel.rayprep_worker,
                    range(raw_arrays["candidates shape"][0])
                ),
                desc='ray preparation',
                total=raw_arrays["candidates shape"][0]
            )
        )

    for i in range(len(candidates)):
        candidates[i]['rays'] = ray[i]

    list_id, scene = model2scene(model=model, config=config)

    if config['debug skip candidate'] != 1:
        candidates = [c for c in candidates[::config['debug skip candidate']]]
    i = [i for i in range(len(candidates))]

    for j in tqdm(i, desc='ray-cast and filter hits'):
        candidates[j]['hit ids'].append(
            cast_plus_eval(
                scene=scene, raytensor=candidates[j]['rays'], config=config
            )
        )
        candidates[j]['rays'] = None

    for np_line, candidate in zip(raw_arrays["candidate hits ra np"], candidates):
        hit_extent = len(candidate['hit ids'][0])
        np_line[0:hit_extent] = np.asarray(candidate['hit ids'])  # relevance?

    with mp.Pool(
            processes=mp_threads,
            initializer=parallel.eval_init,
            initargs=(raw_arrays["midpoints"], raw_arrays["midpoints shape"],
                      raw_arrays["candidates"], raw_arrays["candidates shape"],
                      raw_arrays["candidate hits"], raw_arrays["candidate hits shape"],
                      raw_arrays["face normals"], raw_arrays["face normals shape"], config)
    ) as pool:
        qualified = list(
            tqdm(pool.imap(parallel.eval_worker, range(raw_arrays["candidates shape"][0])),
                 desc="visibility evaluation", total=raw_arrays["candidates shape"][0])
        )

    raw_arrays["candidate hits ra np"][:] = -1
    for np_line, qualifieds in zip(raw_arrays["candidate hits ra np"], qualified):
        qualified_extent = len(qualifieds)
        np_line[0:qualified_extent] = np.asarray(qualifieds)

    for qualifieds, candidate in zip(qualified, candidates):
        if len(qualifieds) > 0:
            candidate['qualified hits'] = qualifieds
            area_covered = np.sum(raw_arrays["face areas ra np"][np.asarray(qualifieds)])
            candidate['area covered'] = area_covered
            candidate['face ratio global'] = len(candidate['qualified hits']) / model['size']
            candidate['area ratio global'] = area_covered / model['area total']
        else:
            candidate['qualified hits'] = []
            candidate['area covered'] = 0
            candidate['face ratio global'] = 0
            candidate['area ratio global'] = 0

    candidates = final_eval(model=model, candidates=candidates, config=config)

    visibility_table = table_base_v2(model=model, candidates=candidates, config=config)
    raw_arrays['visibility mask ra np'] = visibility_table

    # speedup sandbox
    t.report_lap("sandbox in")
    with mp.Pool(
            processes=mp_threads,
            initializer=parallel.overlap_init_rev,
            initargs=(raw_arrays["visibility mask"], raw_arrays["visibility mask shape"],
                      raw_arrays["face areas"], raw_arrays["face areas shape"],
                      raw_arrays["overlap combos"], raw_arrays["overlap combos shape"])
    ) as pool:
        mvv_tbd = list(
            tqdm(pool.imap(parallel.overlap_worker_rev, range(raw_arrays["overlap combos shape"][0])),
                 desc="overlap calculation", total=raw_arrays["overlap combos shape"][0])
        )


    t.report_lap("sandbox timer")
    # with mp.Pool(
    #         processes=mp_threads,
    #         initializer=parallel.overlap_init,
    #         initargs=(raw_arrays["candidate hits"], raw_arrays["candidate hits shape"],
    #                   raw_arrays["face areas"], raw_arrays["face areas shape"],
    #                   raw_arrays["overlap combos"], raw_arrays["overlap combos shape"])
    # ) as pool:
    #     mvv_tbd = list(
    #         tqdm(pool.imap(parallel.overlap_worker, range(raw_arrays["overlap combos shape"][0])),
    #              desc="overlap calculation", total=raw_arrays["overlap combos shape"][0])
    #     )
    # t.report_lap("mp RA timer")

    overlap_areamat, overlap_relmat = overlap_wrap(resulting_stuff=mvv_tbd, candidates=candidates)

    os.makedirs(f'{config["project path"]}{config["work dir"]}', exist_ok=True)
    pickle.dump(visibility_table,
                open(f"{config['project path']}{config['work dir']}"
                     f"{'1_visibility_table.pkl'}", "wb")
                )
    pickle.dump(overlap_areamat,
                open(f"{config['project path']}{config['work dir']}"
                     f"{'1_overlap_table_areas.pkl'}", "wb")
                )
    pickle.dump(overlap_relmat,
                open(f"{config['project path']}{config['work dir']}"
                     f"{'1_overlap_table_relative.pkl'}", "wb")
                )

    pickle.dump(candidates,
                open(f"{config['project path']}{config['work dir']}"
                     f"{'1_candidates.pkl'}", "wb")
                )

    pickle.dump(model,
                open(f"{config['project path']}{config['work dir']}"
                     f"{'1_model.pkl'}", "wb")
                )

    t.report('completed candidates_assess')
    print(f'[  end - candidates_assess ]')


if __name__ == "__main__":
    candidates_assess()
