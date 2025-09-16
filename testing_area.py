import pickle

from tools import timer
from tools.utils import load_model
from tools.utils_nongeom import gather_config
from tools.direction_analysis import overlap_quality



print(f'\n[start - candidates_select ]')
t = timer.Timer()
t.start()

config = gather_config(config_list=['candidates_create.json', 'candidates_assess.json', 'candidates_select.json'])
t.report_lap(achieved='config load complete')

model = load_model(config)
t.report_lap(achieved='model load complete')

# read candidates from graph
with open('handover/1_candidates.pkl', 'rb') as f:
    candidates = pickle.load(f)
with open('handover/0_candidate_graph.pkl', 'rb') as f:
    candidate_graph = pickle.load(f)
t.report_lap(achieved='candidates load complete')

with open('handover/1_visibility_table.pkl', 'rb') as f:
    visibility_table = pickle.load(f)
t.report_lap(achieved='visibility table load complete')

with open('handover/0_candidate_graph.pkl', 'rb') as jar:
    neighborhood_graph = pickle.load(jar)
t.report_lap(achieved='Neighborhood load complete')

with open('handover/1_overlap_table_relative.pkl', 'rb') as jar:
    overlap_table_relative = pickle.load(jar)
t.report_lap(achieved='Neighborhood load complete')

ids=[]

for i in range(0,960):
    ids.append(i)


mdd,sdd,tdd=overlap_quality(model, ids)

a=0



print(f'[  end - candidates_select ]')



