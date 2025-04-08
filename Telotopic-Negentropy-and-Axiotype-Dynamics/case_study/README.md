# Case Study – Vectorial Affective Dynamics

This folder contains all data and code used for the full case study presented in the article  
**"Affective Forces and the Structure of Interaction: A Telotopic Model of Negentropic Organization"**.

## Contents

- `generate_diagrams.py`  
  Script used to generate the diagrams for each phase and actor.

- `vectors.csv`  
  Contains the vectorial data for Fc and Fi (by actor, label, intensity, and angle), sequenced across four interactional phases.

- `config.csv`  
  Contains configuration parameters: telos direction, label formatting, color codes, axis scaling, etc.

- `actor_phase_diagrams/`  
  Output folder containing 8 diagrams (4 phases × 2 actors).

## How to use

To regenerate all diagrams:

```bash
python generate_diagrams.py
