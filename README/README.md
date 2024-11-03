# HGSLoc: 3DGS-based Heuristic Camera Pose Refinement

## Overview

​	We propose HGSLoc, a novel lightweight visual localization frame work, which integrates 3D reconstruction and heuristic strategy to refine further poses with higher precision. We introduce an explicit geometric map for 3D representation and rendering, which provides a new synthesis view of high-quality images for visual localization. We introduce a heuristic refinement strategy, its efficient optimization capa bility can quickly locate the target node, while we set the step level optimization step to enhance the pose accuracy in the scenarios with small errors.
<div align=center>
<img src=".\overview.png" alt="overview" style="zoom: 80%;" width="550px" />
</div>

## Perform well in Challenging Environment

​	Our method mitigates the dependence on complex neural network models while demonstrating improved robustness against noise and higher localization accuracy in challenging environments, as compared to neural network joint optimization strategies.
<div align=center>
<img src=".\noise.jpg" alt="noise" style="zoom: 80%;" width=550px />
</div>

## Dataset

### 1. 7scenes

```
https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
```

### 2. 12scenes

```
https://graphics.stanford.edu/projects/reloc/
```

### 3. Deep Blending

```
http://visual.cs.ucl.ac.uk/pubs/deepblending/datasets.html
```

## Usage

### 1. Training GS

```
python train.py -s data/7scenes/chess --iterations 30000
```

### 2. Coarse Pose Acquiring

Marepo:

```
https://github.com/nianticlabs/marepo
```

ACE:

```
https://github.com/nianticlabs/ace
```

### 3. Camera Pose Refinement

​	We set the step level optimization step to enhance the pose accuracy in the scenarios with small errors, each scene have different step parameter, and see the params file for details.

7scenes:

```
python shader_chess.py -m output/pumpkin
```

12scenes:

```
python shader_kitchen.py -m output/gates381
```

deep blending:

```
python shader_playroom.py -m output/drjohnson
```

## Qualitative Results

### 1. 7scenes
<div align=center>
<img src=".\res1.png" alt="res1" style="zoom: 60%;" width="550px" />
</div>

### 2. 12scenes
<div align=center>
<img src=".\res2.png" alt="res2" style="zoom: 60%;" width="550px" />
</div>

### 3. Deep Blending
<div align=center>
<img src=".\res3.png" alt="res3" style="zoom: 60%;" />
</div>

















