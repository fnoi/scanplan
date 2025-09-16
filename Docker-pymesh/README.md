# mesh subdividion (pymesh docker)

heads-up: pymesh is an ugly but useful library, therefore we use it in a standalone docker

1. mount or load obj files in running container
2. adapt pymesh_config.json according to need
3. run.
   * multiple thresholds and model files are supported in one run
   * be patient, depending on size and max. edge length might take a couple minutes
   * resulting objs are named with parameter and put with the original