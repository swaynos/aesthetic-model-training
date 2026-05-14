"""Pytest configuration for aesthetic-model-training."""
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

# osc_human — full-body human portrait, alternating warm/cool lighting
OSC_HUMAN_SRC   = FIXTURES / "src_osc_human.png"
OSC_HUMAN_MASK  = FIXTURES / "mask_src_osc_human.png"
OSC_HUMAN_EDITS = [FIXTURES / f"edit_osc_human_{i}.png" for i in range(6)]
# even indices (0,2,4) = state A; odd (1,3,5) = state B

# osc_animal — fox-like creature, alternating orange/charcoal palette
OSC_ANIMAL_SRC   = FIXTURES / "src_osc_animal.png"
OSC_ANIMAL_MASK  = FIXTURES / "mask_src_osc_animal.png"
OSC_ANIMAL_EDITS = [FIXTURES / f"edit_osc_animal_{i}.png" for i in range(6)]

# morph_robot — robot with alternating body morphology + palette
MORPH_ROBOT_SRC   = FIXTURES / "src_morph_robot.png"
MORPH_ROBOT_MASK  = FIXTURES / "mask_src_morph_robot.png"
MORPH_ROBOT_EDITS = [FIXTURES / f"edit_morph_robot_{i}.png" for i in range(6)]

# morph_ladder — robot with progressive morphology changes (graded degradation)
MORPH_LADDER_SRC   = FIXTURES / "src_morph_ladder_robot.png"
MORPH_LADDER_MASK  = FIXTURES / "mask_src_morph_ladder_robot.png"
MORPH_LADDER_EDITS = [FIXTURES / f"edit_morph_ladder_robot_{i}.png" for i in range(5)]
