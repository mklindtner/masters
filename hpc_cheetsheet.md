# get disk space
getquota_zhome.sh

# get info for graphic card
nvidia-smi

# GPU debug-usage
voltash, sxm2sh (pref), a100sh
- CUDA_VISIBLE_DEVICES=1,3



# Debugger
- use debugpy

# see version
- module available

# Bjobs
- bstat
- bsub < batchjob.sh
- nodestat -F hpc
- bkill <bjobid>
- showstart <jobid>
- bpeek <jobid>
- bqueues
    - look for gpus with small pending





# SCP

- scp -i ~/.ssh/gbar file1.txt s205421@transfer.gbar.dtu.dk:
    - from homedir to hpc
- scp -i ~/.ssh/gbar s205421@transfer.gbar.dtu.dk: .
    - hpc to homedir

# Python Setup debugger (NOTE! Does not work on GPU Nodes because of their security, they do not allow forward-ports)
- ctrl+shift+p: Ports: Focus on Ports View
- add port 5678
- in .vscode/launch.json write:
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach to Remote Debugger",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "/zhome/25/e/155273/masters"
                }
            ],
            "justMyCode": true
        }
    ]
}
- in vscode terminal do 
    - python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m parkinsons_telemonitoring/parkinsons_training
    - python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m experiment.experiment1_training
    - python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m experiment.experiment1_sensitivity_study
- in "run and debug" click Green arrow

# Normally
- python -m experiment.experiment1_sensitivity_study
- python -m parkinsons_telemonitoring.parkinsons_training --student_mode "mean_and_variance" --iterations 20000 --val_step=1000 --B=0

# Potential problems
- When activate the virtual environemnet using alias "mml" vscode crashes when running import torch
    - Works fine if I dont activate
    - if I unto a gpu node and tries to do "F5" it crashes
    - works using "python mnist/hyperparams" at ~/Desktop/masters
    - when going to use gpu it fails at debugging
    - not failing at debugging when on the home node

