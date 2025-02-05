# get disk space
getquota_zhome.sh

# get info for graphic card
nvidia-smi


# GPU debug-usage
voltash, sxm2sh (pref), a100sh


# Debugger
- use debugpy
- 

# Potential problems
- When activate the virtual environemnet using alias "mml" vscode crashes when running import torch
    - Works fine if I dont activate
    - if I unto a gpu node and tries to do "F5" it crashes
    - works using "python mnist/hyperparams" at ~/Desktop/masters
    - when going to use gpu it fails at debugging
    - not failing at debugging when on the home node
