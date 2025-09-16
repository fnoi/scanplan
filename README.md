# ScanPlan

* currently works in 3 separate steps
* takes input from ./config
* takes input from ./data
* handover intermediate results through ./handover
* to all: all in one: scanplan_run.py
* step by step
  * run candidates_create.py
  * run candidates_assess.py
  * run candidates_select.py (evolutionary algo & greedy)
