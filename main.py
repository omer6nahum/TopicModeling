# TODO:
#  I did already:
#  - I created dataset of 15 records (15 revisions) from Black Lives Matter wiki page (1/2 year difference)
#  - Estimate diff between revisions (and maybe refine the time-difference)
#  __________
#  Left to do:
#  - Create 1 before and 1 after
#       * Preprocess each text
#       * Use sentence embeddings
#       * insert to clustering once
#       * perform step-pipeline (classification | merge | split)
#  - Modify classification so that it depends on threshold (if less than this threshold - open new class)
#  - Add to pipline a "cluster removal" based on number of points in this cluster
#  - Try new methods for each step in the pipline (based on papers)
#  - Extension to multiple steps
#  - Topic labeling
#  - Visualization of multiple steps merging/splitting/disappearing topics
#  - Add a user-intervention for clusters that are good
#  - Git with latest versions of code
#  - conda env export (into git also)
#  - Visualization notebook of results
#  - Report in pdf
