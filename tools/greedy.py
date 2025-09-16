import copy
import numpy as np

from tools.evaluation import one_fitness_greedy


def weight_update(table):
    """helper function for weighted greedy"""
    frequency = np.count_nonzero(table, axis=0)
    table_weighted: np.ndarray = np.divide(table, frequency, out=np.zeros_like(table), where=table != 0)

    return table_weighted


def greedy_search(mask_visibility, candidates, config, model,
                  table_overlap=None, candidate_graph=None,
                  report=False, n=1, report_prec=2):
    """search for suitable strategy based on visibility table"""

    if report:
        print(f'looking for greedy solution n={n}')

    choice = []
    best_ind = None
    greedy_choice = None
    choice_left = [vp for vp in range(len(candidates))]
    covered_sum = 0
    # pbar = tqdm(total=100)

    working_mask = copy.deepcopy(mask_visibility)
    score = np.empty(len(candidates), dtype=np.half)
    for i, c in enumerate(candidates):
        score[i] = np.sum(model['area'][working_mask[i, :]])

    # greedy loop: iteration until requirements met
    while True:
        # pbar.update(1)

        # 1st iteration
        if not choice:
            best_ind = np.argsort(score)[-n]
            greedy_choice = best_ind

        # from 2nd iteration
        else:
            if "fitness" in config['strategy specs'] and choice != []:
                candidates_for_next = []
                for vp_index in choice_left:
                    candidate_for_next = copy.deepcopy(choice)
                    candidate_for_next.append(vp_index)
                    candidates_for_next.append(candidate_for_next)

                candidate_score = np.asarray( # TODO change function call back if bug is fixed
                    [one_fitness_greedy(individual=candidate,
                                 overlap_table=table_overlap,
                                 candidates=candidates,
                                 model=model,
                                 config=config,
                                 visibility_table=working_mask
                                 )
                     for candidate in candidates_for_next]
                )

                best_ind = np.argmin(candidate_score)
                greedy_choice = candidates_for_next[best_ind][-1]

            elif "weighted" in config['strategy specs']:
                a = 0  # TODO
            else:
                score = np.empty(len(candidates), dtype=np.half)
                for i, c in enumerate(candidates):
                    score[i] = np.sum(model['area'][working_mask[i, :]])
                # score = np.sum(working_mask, axis=1)
                best_ind = np.argmax(score)
                greedy_choice = best_ind
                # greedy_best = np.argsort(score)[-n]  # max

        # for all
        choice.append(greedy_choice)
        choice_left.remove(greedy_choice)
        face_hits = working_mask[greedy_choice]
        choice_hit_ind = list(np.nonzero(face_hits))
        working_mask[:, choice_hit_ind] = 0.0

        abs_cover = np.sum(model['area'][tuple(choice_hit_ind)])
        covered_sum += abs_cover

        if config['greedy type'] == ["coverage goal"]:
            thresh = config['coverage goal'][0] * candidates[0]['max coverage area']
            if covered_sum > thresh or abs(covered_sum - thresh) <= 1e-9 or np.count_nonzero(working_mask) == 0:
                break
        elif config['greedy type'] == ["count goal"]:
            thresh = None
            if np.count_nonzero(working_mask) == 0:
                break
        else:
            raise 'invalid greedy type specified'  # TODO: assert correct config in initial gather_config / load step
    # pbar.close()

    strat_stats = {'coverage goal relative': config['coverage goal'],
                   'coverage goal absolute': thresh,
                   'coverage 100p': candidates[0]['max coverage area'],
                   'viewpoint count': len(choice),
                   'viewpoint ids': choice,
                   'coverage achieved relative':
                       round(covered_sum, report_prec) / round(candidates[0]['max coverage area'], report_prec),
                   'coverage achieved absolute': round(covered_sum, report_prec),
                   'scanpoint count': len(choice),
                   'scanpoint ids': choice}

    if report:
        print(f'greedy n={n}: {choice}')
        print(
            f"coverage goal: {strat_stats['coverage goal relative']} "
            f"({int(round(strat_stats['coverage goal absolute'], report_prec))}) reached: "
            f"coverage {round(strat_stats['coverage achieved relative'], report_prec)} "
            f"({strat_stats['coverage achieved absolute']}) "
            f"with {strat_stats['scanpoint count']} scanpoints")

    return strat_stats
