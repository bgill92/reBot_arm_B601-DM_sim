# reBot Arm B601-DM — Genesis sim

Genesis simulation of the [Seeed reBot Arm B601-DM](https://github.com/Seeed-Projects/reBot-DevArm).

```bash
pixi install
pixi run python sim.py            # viewer
pixi run python sim.py --headless # saves frame.png, asserts tracking
```

`assets/rebot_arm_dm/` is `Rebot_Arm_description/DM/` from upstream (CERN-OHL-W v2, see LICENSE there).
Local change: added `<inertial>` to both finger links (upstream omits them; Genesis' MuJoCo parser rejects massless moving bodies).

Notes:
- 8 dofs: joint1..6 + finger_left/finger_right. Genesis ignores the URDF `<mimic>`, so drive right = -left.
- joint2/joint3 limits are [-3.14, 0]; "up" is negative.
