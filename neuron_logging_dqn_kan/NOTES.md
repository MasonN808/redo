# Logged Neurons for KAN Network on DQN in LunarLander (discrete)

## Summary

- Only fully tested on OR operator when combining the spline and basis layer masks
  - This gave more stable number of dormant neurons throughout training than MLPs but need to run it on more seeds
- The AND operator gave 0 dormant neurons for a small number of time steps (for both tau=0 and tau=.25)