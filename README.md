# CrashAI

**Can a neural network learn to avoid collisions by controlling each wheel's brake on its own, plus the throttle?**

> Status: round 2 trained and tested. Learned control halves crashes against scripted rules, and braking alone (no steering) avoids more crashes than a scripted swerve. With steering available, four brakes haven't beaten one pedal + ESC yet. See [Results](#results). This README changes as the project moves.

## Quick start (macOS)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

pytest                                                   # physics + environment tests
python scripts/train.py --controls full                  # 4 brakes + steering + throttle
python scripts/train.py --controls pedal                 # 1 pedal + steering + throttle (ESC on)
python scripts/train.py --controls brakes_only           # 4 brakes + throttle, steering frozen
python scripts/evaluate.py --n 250                       # all controllers on the same 1,000 emergencies
python scripts/watch.py --scenario cut_in --seed 3       # side-by-side GIF in media/
```

Training runs on the CPU (no graphics card needed). A 1.5M-step run takes about 1–2 hours on an M-series Mac.

## Project layout

| File | What it does |
|---|---|
| [crashai/params.py](crashai/params.py) | Car parameters (mass, tires, brakes, steering limits, ABS/ESC tuning) |
| [crashai/physics.py](crashai/physics.py) | Compiled (numba) vehicle dynamics: 4 wheels, combined-slip tires, load transfer, open differential, brake/steering actuator limits, ABS, ESC |
| [crashai/vehicle.py](crashai/vehicle.py) | Python wrapper around the physics |
| [crashai/world.py](crashai/world.py) | Road (lanes, guardrails, friction patches) and scripted obstacles |
| [crashai/scenarios.py](crashai/scenarios.py) | Random emergencies: stopped object, lead car braking, cut-in, pedestrian from behind parked cars |
| [crashai/driver.py](crashai/driver.py) | Simulated human driver: lane and speed keeping, late or no reaction |
| [crashai/sensors.py](crashai/sensors.py) | Simulated sensor-fusion object list: noise, delay, misses, ghosts, range, occlusion |
| [crashai/threat.py](crashai/threat.py) | Path-based time-to-collision and collision checks |
| [crashai/env.py](crashai/env.py) | Gymnasium environment: driver phase → trigger → network control → hand-back |
| [crashai/controllers.py](crashai/controllers.py) | Rule-based baselines and a wrapper for trained policies |
| [crashai/evaluation.py](crashai/evaluation.py) | Fixed test set, metrics, tables |
| [crashai/render.py](crashai/render.py) | Top-down animation, one row per controller |
| [scripts/](scripts/) | `train.py`, `evaluate.py`, `watch.py` |
| [tests/](tests/) | Physics checks (braking ≈ μg, ABS, brake-steer direction, open-diff trick), geometry, environment |

## The idea

Modern cars can brake each wheel separately. The hardware is already there: ABS and stability control (ESC) use it every day. The driver never gets direct access to it, though. One pedal sets brake force for all four wheels, and a person couldn't coordinate four separate brakes in a split second anyway.

Uneven braking does more than slow the car down. It turns it. Braking only the left wheels creates a yaw moment that pulls the nose left, the same way a tank or a differential-drive robot steers by running its sides at different speeds. So the four brakes are really a set of extra steering and stability controls that nobody uses on purpose.

**Claim:** a neural network that controls the four brakes and the throttle can learn to use that extra control, and avoid collisions that a single pedal plus a steering wheel can't.

## Why this is plausible

- **The physics is real.** If one side brakes harder than the other, the tire forces on the two sides differ, and that produces a turning moment around the car's vertical axis.
- **It's already done in a narrow way.** ESC brakes single wheels to stop a skid. Brake-based torque vectoring helps cars turn in sharper. Steer-by-brake has been studied as a fallback for when steering fails. All of these are hand-tuned rules. None of them is a learned policy aimed at avoiding collisions.
- **A neural network suits the job.** There are five continuous controls, the tire dynamics are nonlinear, and decisions take milliseconds. Reinforcement learning in simulation handles exactly this kind of problem.

## The main limit

All tire forces share one budget, the **friction circle**. Force a tire spends on braking can't also be spent on cornering. Brake-steer bends the car's path less than real steering does. Its value is likely to be:

1. **Extra control on top of steering**, giving more yaw authority and faster response in an emergency.
2. **Keeping the car stable at the limit**, so it doesn't spin when a hard swerve and hard braking happen together.
3. **A fallback** when steering is lost or the driver freezes.

A key thing to find out is *how much* collision avoidance the brakes add, compared with steering alone and with steering plus standard ESC.

## Setup

| Part | Choice |
|---|---|
| Simulator | **Our own** 2D simulator (Python + numba): 4-wheel model, Pacejka "Magic Formula" tires with combined slip, load transfer, physics at 500 Hz (see [Simulator](#simulator)) |
| Car | Ordinary front-wheel-drive car with an open differential (see [The car](#the-car)) |
| Actions | **6 continuous controls at 20 Hz:** steering + brake torque for each of the 4 wheels + **one** throttle value (see [Actions](#actions)) |
| Observations | 142 numbers: car state (30) + the 8 nearest objects from the sensor-fusion list (14 each) (see [Sensors](#sensors)) |
| Reward | See [Reward](#reward) |
| Algorithm | SAC (Stable-Baselines3), 2×256 MLP, 6 parallel environments |
| Scenarios | Stopped object, lead car braking hard, cut-in, pedestrian/cyclist from behind parked cars; dry/wet/snow/ice; 20% split-friction patches; random traffic in neighbouring lanes |

## Reward

Per control step (the total is scaled by 0.1 for training):

| Term | Value |
|---|---|
| Collision with an object | −(50 + 3 × impact speed in m/s), episode ends |
| Hitting a guardrail | −(50 + 1.5 × speed), episode ends |
| Stopped safely, or handed back (no predicted collision for 0.5 s, car stable) | +10, minus up to 3 for ending between lanes, episode ends |
| Premature hand-back (an alert driver crashes within 4 s of taking back control) | Same as a collision |
| Time | −0.02 per step (hand back sooner) |
| Smoothness | −0.02 × (change in action)² |
| Sliding | −0.5 × (sideslip above ~3°) |
| Danger shaping | Potential-based: −2 × (how close the true time-to-collision is to zero, within 3 s) |

Lower impact speed still counts when a crash can't be avoided, as it does for real AEB.

## The car

**Decision: an ordinary front-wheel-drive (FWD) car with one throttle.**

FWD is the most common layout on the road. If a learned policy can avoid crashes in a plain FWD car using only the brakes it already has, the result applies to most cars sold today. No special hardware is needed.

### Can normal cars throttle each wheel?

No. One engine drives the front wheels through an **open differential**, which always splits the engine's torque equally between left and right.

| Car type | Per-wheel throttle? |
|---|---|
| Normal cars (open differential) | No, one throttle shared equally |
| Some sporty/premium cars (torque-vectoring differential, e.g. Audi Sport differential, Acura SH-AWD) | Partly: can shift torque left/right, usually on one axle |
| EVs with a motor per wheel (e.g. Rivian quad-motor, Mercedes G 580 electric, Rimac Nevera) | Yes, fully |

**The brakes can fake it.** With an open differential, braking one front wheel pushes more of the engine's torque to the other front wheel. ESC and "brake-based torque vectoring" already rely on this. So in our FWD car, individual brakes plus a single throttle give the network some per-wheel drive control too, and the simulator must model the open differential to capture it.

### Starting parameters (typical mid-size FWD car)

| Parameter | Value |
|---|---|
| Mass | 1,500 kg |
| Weight split (front / rear) | 60 / 40 |
| Wheelbase | 2.70 m |
| Track width | 1.55 m |
| Centre-of-gravity height | 0.55 m |
| Drivetrain | FWD, open differential, single throttle |
| Brakes | Independent brake torque on all 4 wheels |

### Later extension

A quad-motor EV (true per-wheel throttle) to see how much extra avoidance per-wheel drive adds on top of per-wheel braking.

## Actions

**Decision: the network controls steering, all four brakes, and the throttle.**

It sends 6 continuous values every control step:

| # | Control | Range | Real-world limit modelled |
|---|---|---|---|
| 1 | Steering (front wheel angle) | ±35° | Max steering rate ~60°/s at the wheels (a fast emergency swerve) |
| 2–5 | Brake torque: front-left, front-right, rear-left, rear-right | 0 to max per wheel | Hydraulic pressure takes ~0.15–0.2 s to build up |
| 6 | Throttle | 0 to 1 | Engine torque split equally by the open differential |

Rate limits and delays matter. Without them the network would learn moves no real car can make, such as instant full-lock steering or instant brake pressure.

### Keeping the claim testable

With steering included, the network could avoid crashes just by swerving. The question becomes: **how much do individual brakes add on top of steering?** We answer it by training the same network with different controls and comparing them (see [Baselines](#baselines-to-beat)):

- Steering + one brake pedal + throttle (what a human has)
- Steering + four individual brakes + throttle (the full idea)
- Four individual brakes + throttle, no steering (brake-steer only, for curiosity and the steering-failure case)

## When the network steps in

**Decision: a human (simulated) drives normally. The network takes over only when a crash is about to happen**, the way automatic emergency braking (AEB) works today.

1. **Normal driving:** a scripted driver model drives the car. The network watches but does nothing.
2. **Trigger:** when a threat check fires, the network takes full control of steering, all four brakes and the throttle. To start, the check is rule-based: the **time-to-collision (TTC)** with any object drops below a threshold (around 1.5–2 s, to be tuned), or the car starts to lose grip (e.g. yaw rate far from what the steering asks for).
3. **Emergency maneuver:** the network avoids the crash and brings the car back to a stable state, ideally still in its lane or on the road.
4. **Hand-back:** once the threat is gone and the car is stable, control returns smoothly to the driver.

What this means for training:

- **Episodes start at the trigger moment**, so the network spends all its practice on near-crashes, not on boring cruising.
- The driver model is also what *causes* many dangerous moments: looking away, braking late, or over-correcting.
- Reward covers the whole intervention: no collision, stay on the road, end stable, and hand back quickly.

Later: let the network decide *when* to step in, instead of the fixed TTC rule. That's harder, because stepping in too often is also a failure.

## Simulator

**Decision: write our own lightweight simulator.**

| Option | Why not (for now) |
|---|---|
| CARLA | No official macOS support; needs a strong NVIDIA graphics card |
| BeamNG | Mainly Windows; heavy and hard to control from code |
| MuJoCo | Runs on macOS, but its tire model is too simple for braking at the grip limit |

Why our own:

- Pure Python + NumPy runs on any Mac (including Apple Silicon) with no graphics card.
- It runs thousands of times faster than real time, which RL needs (millions of practice runs).
- It models exactly what this project depends on: separate brake torque per wheel, wheel slip, tire grip limits, weight transfer, and slippery patches.
- It plugs into standard tools: **Gymnasium** (environment interface) and **Stable-Baselines3** (RL algorithms). A simple 2D top-down view can be drawn with **pygame** or **matplotlib**.

Weakness: a home-made model is simpler than reality. Plan: prove the idea here first, then check the best policies in a high-fidelity simulator (e.g. BeamNG or CARLA on a Linux/Windows machine) later.

## Sensors

The network gets the same sensor data a modern car already collects. It needs two kinds of information: **what's around the car** and **what the car is doing right now**.

### What's around the car

| Sensor | What it gives | Strengths / weaknesses |
|---|---|---|
| Cameras | Images: lanes, cars, pedestrians, signs | Detailed, cheap; weak in the dark, glare and rain |
| Radar | Distance and closing speed of objects | Works in any weather, measures speed directly; low detail |
| Lidar | 3D point cloud of the surroundings | Precise shape and distance; expensive, struggles in heavy rain or snow |
| Ultrasonic | Short-range distance (parking sensors) | Only useful within a few meters |
| V2X (optional) | Messages from other cars and infrastructure | Can "see" around corners; few cars have it today |

### What the car is doing

| Sensor | What it gives |
|---|---|
| Wheel speed sensors (×4, already used by ABS) | Speed of each wheel, which shows each wheel's slip, the key signal for individual braking |
| IMU (accelerometers + gyroscope) | Yaw rate and lateral/longitudinal acceleration: how the car is actually turning and sliding |
| Steering angle sensor | What the driver is asking for |
| Brake pressure sensors | Actual brake force at each wheel |
| GNSS (GPS) | Position and heading (coarse) |

### Target car: an ideal modern sensor set

**Decision: model a high-end 2025–2026 car with full 360° coverage**, similar to the sensor sets on cars like the Volvo EX90 or Mercedes S-Class.

| Sensor | Count | Range (approx.) | Covers |
|---|---|---|---|
| Cameras | 8 | up to ~200 m (front) | 360° |
| Front long-range lidar | 1 | ~250 m | Forward ~120° |
| Front long-range radar | 1 | ~200 m | Forward |
| Corner radars | 4 | ~80–100 m | Sides and rear, including blind spots |
| Ultrasonic | 12 | ~5 m | Close range all around |
| IMU, wheel speeds (×4), steering angle, brake pressures (×4) | — | — | The car's own motion |
| GNSS + map | — | — | Position, road layout, speed limits |

### How the data reaches the network

**Decision: the network takes what the car already produces.** Modern cars run "sensor fusion" software that combines all the sensors into one picture of the world. The network reads that output, not raw pixels or lidar points:

| Input | Contents |
|---|---|
| **Object list** (8 nearest) | Position, velocity, size, heading, type (car, truck, pedestrian, cyclist, debris), and how confident the car is in each |
| **Road** | Lane lines, road edges, drivable free space |
| **Car state** | Speed, yaw rate, lateral/longitudinal acceleration, each wheel's speed and slip, steering angle, each wheel's brake pressure |
| **Road grip estimate** | Rough friction estimate (dry / wet / snow / ice), which modern cars estimate from wheel slip and weather |
| **Driver** | Steering, throttle and brake input at the moment of takeover |

This keeps the project focused on control, the new part, and matches what a real car could actually feed such a system.

In the simulator, we don't simulate cameras or lidar physically. We generate the fused object list directly from the true world state, then **corrupt it on purpose**: position and speed noise, ~50–100 ms delay, a ~20 Hz update rate, missed detections, false "ghost" objects, and shorter range in rain or fog. A policy trained on perfect data will fail on real data.

Implemented in [crashai/sensors.py](crashai/sensors.py):

| Effect | Model |
|---|---|
| Noise | Slowly drifting (correlated) position and velocity error, growing with distance |
| Delay | 1–2 frames (50–100 ms), random per episode |
| Missed detections | 2–10% per frame depending on weather; the tracker coasts the object for up to 4 frames |
| Ghosts | Occasional false low-confidence objects 15–80 m ahead |
| Range | ~220 m ahead, ~90 m to the sides; pedestrians/cyclists ~90 m; × 0.4–0.65 in rain, snow or fog |
| Occlusion | Objects fully hidden behind nearer objects aren't reported (the pedestrian behind a parked van) |

## Baselines to beat

**Rule-based (no learning):**

1. Straight-line full braking (what most drivers do)
2. Braking with ABS
3. Braking + a scripted swerve (with and without ESC)

**Learned (same network, same training, different controls):**

4. Steering + **one** brake pedal + throttle: the "human controls" network
5. Steering + **four** brakes + throttle: the full idea
6. **Four** brakes + throttle, no steering: brake-steer only

**The main result is 5 vs 4.** If 5 avoids more crashes than 4, the gain comes from individual wheel braking and not just from having a neural network drive.

Plus a **"driver only"** reference: no system at all, the distracted or late driver keeps driving.

ABS is always on (it's the low-level hardware that keeps wheels from locking). ESC is on for the human-like controls (driver only, pedal network), because every modern car has it. In the full network, ESC is off, since the network has the four brakes itself.

## Results

The test set is 1,000 emergencies (250 per scenario type), identical for every controller. Every hand-back is checked by an alert driver driving on for 4 s. Each cell is the **crash rate**, with the mean impact speed over all episodes in brackets (0 when avoided). Full tables: [results/RESULTS.md](results/RESULTS.md).

### Summary

1. **Learned control halves the crash rate.** The trained networks crash in 13–15% of emergencies, against 29% for the best scripted rule and 44% for today's style of emergency braking. When they do crash, they hit much slower (4–5 km/h average vs 11).
2. **Braking alone, with no steering, steers the car out of trouble.** The brakes-only network crashes in 25%, fewer than straight ABS braking (44%) and even fewer than the scripted controller that can steer (29%). Uneven braking is a real steering tool, and a strong fallback if steering fails.
3. **With steering available, four brakes did *not* beat one pedal + ESC (yet).** The single-pedal network scored 13% vs 15%. Paired on the same emergencies, the pedal network avoided 31 crashes the four-brake one didn't, against 10 the other way. So the main claim isn't supported in this round.

### Round 2 test results (1,000 emergencies per controller)

| Controller | Overall | Stopped object | Lead car brakes | Cut-in | Pedestrian |
|---|---|---|---|---|---|
| Driver only (no system) | 87% (46 km/h) | 100% (78 km/h) | 100% (53 km/h) | 88% (30 km/h) | 59% (24 km/h) |
| Full braking, no ABS | 49% (17 km/h) | 59% (24 km/h) | 93% (30 km/h) | 41% (13 km/h) | 4% (1 km/h) |
| Full braking + ABS (≈ today's AEB) | 44% (14 km/h) | 48% (18 km/h) | 87% (24 km/h) | 40% (13 km/h) | 2% (0 km/h) |
| Braking + scripted swerve | 30% (11 km/h) | 30% (13 km/h) | 41% (15 km/h) | 42% (15 km/h) | 9% (1 km/h) |
| Braking + swerve + ESC | 29% (11 km/h) | 28% (13 km/h) | 40% (15 km/h) | 39% (14 km/h) | 9% (1 km/h) |
| **Learned: steering + 1 pedal (+ESC)** | **13% (4 km/h)** | 8% (4 km/h) | 18% (4 km/h) | 25% (8 km/h) | 2% (0 km/h) |
| **Learned: steering + 4 brakes** | **15% (5 km/h)** | 9% (4 km/h) | 24% (6 km/h) | 26% (9 km/h) | 1% (0 km/h) |
| **Learned: 4 brakes, no steering** | **25% (7 km/h)** | 17% (6 km/h) | 48% (12 km/h) | 33% (9 km/h) | 2% (0 km/h) |

By road surface:

| Controller | Dry | Wet | Snow | Ice | Split grip |
|---|---|---|---|---|---|
| Braking + swerve + ESC | 16% | 25% | 48% | 78% | 67% |
| Learned: 1 pedal | 5% | 9% | 21% | 54% | 30% |
| Learned: 4 brakes | 8% | 9% | 22% | 58% | 33% |
| Learned: 4 brakes, no steering | 13% | 19% | 40% | 80% | 50% |

About 12% of the emergencies (118 of 1,000) end in a crash whatever any controller does: the warning comes too late for the grip available.

### 3D replay viewer

[viewer/index.html](viewer/index.html) replays recorded emergencies in 3D (Three.js). All six controllers drive the same emergency: the one you follow is drawn in full, and the others appear as coloured ghosts once their paths split at takeover. The body is see-through, so you can watch each brake disc glow with its own brake force, the tire-force arrows, skid marks and brake lights. An instrument cluster shows speed, steering, each wheel's brake force and a **grip circle per tire**: the dot turns red when that tire uses all its grip, which is the friction-circle limit described above.

```bash
python scripts/export_viewer.py        # records showcase emergencies into viewer/episodes.js
open viewer/index.html                 # or serve the folder: python -m http.server -d viewer
```

Keys: Space play/pause, ←/→ step, 1–6 follow a controller, C camera, X x-ray, F forces, S sensor boxes (what the car's sensors report, ghosts in red).

Assets in [viewer/assets/](viewer/assets/), all from the three.js examples (r160): the car model "Ferrari 458 Italia" by vicent091036 (CC BY 4.0) with its contact-shadow texture, the "Venice sunset" HDR sky (Poly Haven, CC0) for reflections, and the JavaScript Draco decoder. The model is for looks only: the physics is still the mid-size front-wheel-drive car above.

### Animations (2D)

- [media/stopped_wet_all_controllers.gif](media/stopped_wet_all_controllers.gif): a stopped car on a wet road at 107 km/h. Driver alone crashes at 107 km/h, ABS braking at 55 km/h, and the scripted swerve hits the guardrail. All three networks get past. The brakes-only one changes lanes using uneven braking alone.
- [media/split_mu_four_brakes_vs_pedal.gif](media/split_mu_four_brakes_vs_pedal.gif): one of the 10 cases where four brakes beat one pedal, on a wet road with a slippery patch under one side.

Make your own: `python scripts/watch.py --scenario cut_in --seed 7 --controllers brake_abs learned_pedal learned_full`.

### Why four brakes didn't win (yet), and what to try next

- **The pedal network isn't "one pedal" in practice.** It has ESC, which already brakes single wheels using hand-tuned rules. So the comparison is learned four-wheel braking against ESC four-wheel braking, which is a harder test than planned.
- **Grip is the limit.** With steering available, the tires are usually already at their grip limit, so extra yaw from the brakes has little room to help. Brakes-only shows that the effect is large when steering is missing.
- **Learning is harder with more controls.** The four-brake network has 6 controls instead of 3 and started slower in both rounds. It may need more training or tuning.
- **One training run each.** RL results vary between runs. A 2-point gap needs several seeds per setup to be trusted.

Next experiments:

1. Train 3 seeds per setup, and train longer.
2. Add "pedal without ESC" to separate learned per-wheel braking from ESC.
3. Add harder emergencies where steering alone runs out: split grip, steering failure, swerving while braking on snow.
4. Give the four-brake network a head start: start from the pedal network's skills, or use one brake value plus a left/right split.

### Lesson from round 1: the networks gamed the hand-back

The first trained networks scored 21–22% crashes against 35% for the best scripted rule. But in the animations, a network would "avoid" a stopped car at 98 km/h while the car was still in its lane ahead. It had learned to twitch the car so the danger check briefly predicted a miss. The episode then ended as a success and control went back to the driver, who would crash a moment later.

Re-scored with an **alert driver driving on for 4 s after every hand-back**, where a crash in that time counts as a crash:

| Controller | Crash rate as first scored | With hand-back check |
|---|---|---|
| Braking + swerve + ESC (scripted) | 35% | 38% |
| Learned, single pedal | 21% | 44% |
| Learned, four brakes | 22% | 44% |
| Learned, brakes only | 30% | 64% |

Fixes for round 2:

1. The after-hand-back check is part of training, so a premature hand-back is punished like a crash.
2. Hand-back also needs a clear path when the car is assumed to simply follow its lane, not only along its current curve.
3. Cut-ins started too close: 152 of 250 crashed whatever any controller did. The cutting car now starts 14–35 m ahead instead of 6–22 m.

Round-1 models and results are kept in `models_v1/` and `results_v1/`.

## Open questions

- [x] Does the network control steering too? **Yes** (see [Actions](#actions))
- [x] What kind of car? **Ordinary FWD, open differential, one throttle** (see [The car](#the-car))
- [x] Simulator: ~~write our own lightweight one, or use an existing one?~~ **Our own** (see [Simulator](#simulator))
- [x] Full control or step in? **Steps in when about to crash** (see [When the network steps in](#when-the-network-steps-in))
- [x] Sensor input? **Whatever the car provides: the fused object list + car state** (see [Sensors](#sensors))
- [x] Sensor setup? **Ideal modern car: 8 cameras, lidar, 5 radars, 12 ultrasonic**
- [ ] TTC trigger threshold: what value gives the network enough time without stepping in too often?
- [ ] Scripted driver model: how realistic does it need to be?

## Status / log

- 2026-10-21: Idea written down.
- 2026-10-21: Decided the network takes input from the car's own sensors.
- 2026-10-21: Decided to write our own lightweight Python simulator (macOS-friendly, fast to train).
- 2026-10-21: Chose an ordinary FWD car with an open differential and one throttle; per-wheel throttle (quad-motor EV) is a later extension.
- 2026-10-21: The network controls steering too (6 controls in total). The main experiment compares 4 individual brakes vs 1 pedal, both with steering.
- 2026-10-21: The network steps in only when a crash is about to happen (TTC trigger, like AEB). Input is the fused object list from an ideal modern sensor set, with noise and delay added in simulation.
- 2026-10-21: Built the simulator, scenarios, sensor model, environment, baselines, training, evaluation and viewer. 24 tests pass. Physics checks: ABS braking ≈ μg, ABS stops 6 m shorter than locked wheels from 108 km/h, braking one side turns the car toward that side, and braking a spinning wheel on split grip lets the open differential drive the other wheel. Started training the three learned policies, then paused before the first checkpoint.
- 2026-10-21: Round 1 trained (1M steps each, ~2 h). Found the networks were gaming the hand-back rule (see [Lesson from round 1](#lesson-from-round-1-the-networks-gamed-the-hand-back)). Added a 4 s after-hand-back check, a stricter hand-back rule and fairer cut-ins. Round 2 training started.
- 2026-10-21: Round 2 tested on 1,000 emergencies (training stopped at ~950k of 1M steps; best checkpoints used). Learned: 13% (pedal + ESC), 15% (4 brakes), 25% (brakes only, no steering), against 29% for the best scripted rule. Only 5–8 crashes after hand-back per controller, so the gaming is fixed.
