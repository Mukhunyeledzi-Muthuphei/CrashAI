# CrashAI

**Can a neural network learn to avoid collisions by controlling each wheel's brake on its own, plus the throttle?**

> Status: idea / early design. This README changes as the design is discussed.

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

## Proposed setup (draft)

| Part | Draft choice |
|---|---|
| Simulator | **Our own** lightweight 2D simulator in Python/NumPy: 4-wheel model, nonlinear tires (Pacejka "Magic Formula"), load transfer (see [Simulator](#simulator)) |
| Car | Ordinary front-wheel-drive car with an open differential (see [The car](#the-car)) |
| Actions | **6 continuous controls:** steering + brake torque for each of the 4 wheels + **one** throttle value (see [Actions](#actions)) |
| Observations | Data from the car's own sensors (see [Sensors](#sensors)) |
| Reward | Large penalty for a collision; small penalties for leaving the road, jerk, and excess slip; bonus for ending stable and handing back control quickly |
| Algorithm | Off-policy continuous-control RL (SAC or TD3) as a first choice; PPO as a baseline |
| Scenarios | Sudden obstacle, car cutting in, low-friction patches (wet or ice), split-friction road surfaces |

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
| **Object list** (up to 16 nearest) | Position, velocity, acceleration, size, heading, type (car, truck, cyclist, pedestrian, unknown), and how confident the car is in each |
| **Road** | Lane lines, road edges, drivable free space |
| **Car state** | Speed, yaw rate, lateral/longitudinal acceleration, each wheel's speed and slip, steering angle, each wheel's brake pressure |
| **Road grip estimate** | Rough friction estimate (dry / wet / snow / ice), which modern cars estimate from wheel slip and weather |
| **Driver** | Steering, throttle and brake input at the moment of takeover |

This keeps the project focused on control, the new part, and matches what a real car could actually feed such a system.

In the simulator, we don't simulate cameras or lidar physically. We generate the fused object list directly from the true world state, then **corrupt it on purpose**: position and speed noise, ~50–100 ms delay, a ~20 Hz update rate, missed detections, false "ghost" objects, and shorter range in rain or fog. A policy trained on perfect data will fail on real data.

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
