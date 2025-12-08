# Scripts for Submitting GPU Jobs on OSG Cluster
## Workflow
- Replace the execution command in ``submit.sh`` with your executable
	- Important: either specify **absolute path** of your executable, or make sure you have a **shebang** at the top of the file telling the execution point where to find your file (e.g. ``!/usr/bin/env python``)
- In ``gpu_jobs.sub``, alter resource requests and replace your input and output files as needed
- Submit jobs from AP23 login node with ``condor_submit gpu_jobs.sub``
- Watch status with ``condor_watch_q``
