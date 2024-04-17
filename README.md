# Vision Language Models - Test Repo connection to Supplementary Materials [WIP]
Final team project for the class DL CS 7643 at Georgia Tech, Spring 2024. Team: Deep Thinkers.

To run a vlm_reasoner, salloc or sbatch on GaTech Pace clusters with a V100 with at least 32Gb memory.
Recommended to run in conda env.

The design of the code repository aims to be modular, dependencies minimized and as much async development and parallelisms between tasks and steps is desirable.

Each code change has an individual commit. A merge to main will require a PR.

While code LGTM is not necessary for a PR merge due to modular design, comments are welcome.

Current repository structure:

```
.
├── assets
├── dataset
├── modules
│   ├── ahmad_conditional_visual
│   └── denisa_vlm_reasoners
├── requirements
└── scripts

```
