# M.tech-Thesis
LLM-Enhanced Failure Mode and Effects Analysis for Safer Multi-Agent Coverage in Unknown Environments


This project evaluates multi-robot coverage path planning under dynamically changing hazard environments using three different FMEA strategies:

1. Static FMEA
2. Dynamic FMEA without LLM
3. Dynamic FMEA with LLM-assisted updates

The framework generates randomized hazard maps, updates risk estimates over time, and compares mission performance using coverage, damage risk, catastrophic risk, and mission failure metrics.

The LLM-assisted version uses a locally hosted Qwen model through Ollama to dynamically update FMEA probabilities based on hazard evolution, movement anomalies, and sensor observations.

---

## Repository Structure

```text
.
├── coverage_fmea_evaluator.py
├── multi_update_fmea.py
└── README.md
```

### Files

#### coverage_fmea_evaluator.py

Main experiment script.

Run this file to:

* Generate randomized hazard scenarios
* Execute multi-robot coverage missions
* Compare Static FMEA, Dynamic FMEA, and Dynamic FMEA + LLM
* Produce evaluation results and plots

#### multi_update_fmea.py

Contains:

* FMEA update logic
* Dynamic hazard modeling
* Risk estimation
* Ollama/Qwen integration
* LLM prompt generation and processing

This file is imported automatically by the main evaluator and does not need to be executed separately.

---

## Requirements

Python 3.10 or newer.

Install dependencies:

```bash
pip install numpy pandas matplotlib requests
```

---

## Ollama Setup

This project uses a local Ollama server with Qwen.

### Install Ollama

Follow instructions from:

https://ollama.com

### Pull Qwen Model

```bash
ollama pull qwen2:latest
```

### Start Ollama

```bash
ollama serve
```

The code expects Ollama to be available at:

```text
http://localhost:11434
```

---

## Running the Experiment

Execute:

```bash
python coverage_fmea_evaluator.py
```

No other file needs to be run manually.

---

## Output

Results are generated inside:

```text
coverage_randomized_astar_results/
```

The output includes:

* Experiment logs
* Risk evaluation results
* Coverage statistics
* Generated plots
* Cached LLM responses

---

## Methodology

The framework evaluates multi-robot coverage planning under dynamic hazards.

Hazards evolve over time and affect:

* Terrain irregularities
* Slipping hazards
* Communication failures
* Dynamic obstacles
* Static obstacles
* Amplified hazard zones

Three approaches are compared:

### Static FMEA

Uses fixed failure probabilities throughout the mission.

### Dynamic FMEA without LLM

Updates failure probabilities using observed hazard changes and sensor information.

### Dynamic FMEA + LLM

Uses Qwen through Ollama to interpret hazard evolution and adjust FMEA failure-mode probabilities dynamically.

---

## Notes

* Ollama must be running before executing the experiment.
* The `qwen2:latest` model must be downloaded locally.
* Cached LLM outputs are stored automatically to reduce repeated model calls.
* For reproducibility, randomized scenarios use predefined seeds.

---

## Author

Saswata Maji
