# syslab snr rsrch 


# Multi-Agent UAV Capture and Evasion w/ Adversarial RL

snr rsrch project for Computer Systems Research Lab (SysLab) at TJHSST, May 2026. Students Owen Murphy and Rishikesh Narayana, co26, advised by Dr. Yilmaz and Dr. Gabor.

Two drones trained to play tag against each other. One tries to catch, one tries to escape. Neither was told how. We trained them against each other in simulation using PPO self-play inside a custom PyBullet environment, then deployed the tracker policy onto a real DJI Tello EDU using AprilTag detection as the perception layer. The tracker hit 95% capture rate in sim after three million timesteps, and the same policy file, no retraining, no fine-tuning, ran on the actual drone and tracked a moving target + a manually controlled evader.

The full paper is in `docs/`.

---

## what's in the repo

```
sim/            PyBullet training environment and self-play loop
deploy/         Tello deployment script with AprilTag perception pipeline
models/         Trained tracker and evader models and VecNormalize files
docs/           Final paper + poster + presentation
pybullet_train.py   main training entrypoint
deploy.py           main deployment entrypoint
```

## training

The simulator is a custom Gymnasium environment built on PyBullet. Two agents share the arena: a tracker and evader   Each one observes a nine-dimensional vector: relative position of the opponent in the observer's body frame, the observer's own velocity, and the opponent's estimated velocity. Actions are continuous 3D velocity commands in the range [-1, +1] on each axis.

Training alternates between the two agents. One trains with PPO while the other sits frozen as the opponent, then they swap. This is standard self-play. It means the reward curve oscillates by design,  whatever agent is currently learning climbs, then the roles flip and it drops back. 

A few things turned out to be load-bearing for getting non-trivial behavior:

**Vision-limited observation.** Early runs gave both agents full state. The tracker became effectively omniscient, the evader never developed any real strategy, and training stalled fast. Restricting the tracker to a cone-shaped FOV matching the forward camera geometry on the Tello fixed this. Suddenly the evader had something to actually optimize for, which was just breaking line of sight, and both policies got interesting.

**Curriculum.** Without it both agents locked onto trivial solutions. The evader sprinted straight away, the tracker trailed at a fixed offset, neither one ever discovered a feint or a dodge. The curriculum shrinks the starting separation progressively across rounds, forcing close-range engagements where constant-velocity doesn't cut it and it forces both drones to learn stronger behaivors because of the closened region.  Around round six it also increases the evader's max speed by 10% and drops a two-meter cylindrical obstacle at the arena center. The evader figures out how to transit behind the cylinder to break line of sight, and then reverses heading sharply once it does. This was done in an effort to induce more advanced behaviors from each agent.

**Reward shaping.** The tracker gets rewarded on a negative-tanh of distance (so the gradient stays nonzero at long range), a per-step closing-rate bonus, a term for keeping the evader near the center of its FOV, and a big terminal bonus for captures inside 0.7m. The evader gets rewarded for distance, for opening distance step over step, and specifically for not holding a constant heading across consecutive steps, which is what kills the straight-line collapse. Both have a smoothness penalty on velocity commands, which is important for transfer because it discourages high-frequency oscillation that the Tello's flight controller cannot execute.

 The loop saves to one directory: tracker.zip, evader.zip, the matching VecNorm pickle files, timestamped checkpoints, the SB3 log directory, and a `training_state.json` with the current round and step count. If you kill it and restart it picks up from that JSON file w/ rounds and everything

---

## Running training

Install dependencies:

```bash
pip install stable-baselines3 pybullet gymnasium opencv-python numpy
```

Then:

```bash
python pybullet_train.py
```

That starts the sim from scratch completely;  to resume a previous run, the script checks for `training_state.json` automatically and continues from whatever round it was on. To watch the trained agents run in the PyBullet GUI with the FOV cone drawn:

```bash
python pybullet_train.py --demo
```

The demo loads the latest checkpoints, disables training, and renders the scene. Useful for checking whether the evader is actually doing interesting things or just sprinting in a straight line (if it's the latter, the curriculum probably needs another round or two).

---

## How the sim-to-real transfer works

The policy file doesn't change at all between simulation and hardware. The gap gets closed on the perception side instead, with three components added between the raw camera output and the policy input.

**Inter-frame velocity prediction.** The H.264 stream from the Tello arrives with nontrivial jitter and the AprilTag detection pipeline runs at roughly 15 Hz against a 20 Hz control loop, which means the observation feeding the network is frequently 50 to 100 ms stale. We extrapolate the evader's relative position forward in time between detections using the most recent smoothed velocity estimate. The prediction is capped at 500 ms; past that it falls back to the last confirmed detection rather than guessing further.

**Action smoothing.** A one-pole low-pass filter with coefficient α=0.15 sits on the policy output before the velocity command goes to the drone. Lower values cause visible jitter in the Tello's attitude. Higher values add enough lag that the tracker falls behind on sharp turns. 0.15 was found by hand across a few test flights.

**Yaw controller.** The Tello has only a forward-facing camera. The trained policy doesn't know to rotate to keep the target in frame, so an auxiliary proportional yaw controller runs in parallel with the velocity command, rotating the drone in body-frame yaw to keep the AprilTag near the image center. Gain 90 deg/s per radian of horizontal bearing error, 0.04 radian deadband. It runs alongside the network output, not through it, so the policy itself never had to learn yaw control.

One thing that helped a lot during training was injecting uniform latency jitter of ±30 ms into the simulator step time. The policy never saw a perfectly synchronous observation during training, which probably explains why it held up reasonably well against the actual Wi-Fi-induced delays on the real drone.

---

## Running deployment

You need a DJI Tello EDU, a printed AprilTag (family tag36h11, 10 cm side length) mounted on whatever is serving as the evader, and a laptop that can connect to the Tello over Wi-Fi.

Additional dependencies on top of the training stack:

```bash
pip install djitellopy pupil-apriltags pyav
```

Basic flight against a moving target:

```bash
python deploy.py
```

The script loads `models/tracker.zip` and `models/tracker_vecnorm.pkl` by default. To point it at a different checkpoint, pass `--model path/to/tracker.zip --vecnorm path/to/tracker_vecnorm.pkl`.

Useful flags:

```bash
--display          show the annotated camera feed while flying
--dry-run          use a laptop webcam instead of the Tello (for testing the perception pipeline on the ground)
--motors-off       run the policy and print commands without actually taking off
--record file.avi  save the annotated video stream
```

`--dry-run` and `--motors-off` are worth using before you put it in the air. `--dry-run` lets you verify that AprilTag detection is working and that the observation vector being assembled looks sane. `--motors-off` lets you see what velocity commands the policy would issue in response to actual detections, without the drone going anywhere.

When the tag goes out of frame (the dominant real-world failure mode), the drone rotates slowly toward the last-known bearing until detection re-acquires, which usually takes one to two seconds if the target is still in the room.

---

## Future directions

The most obvious next step is replacing the AprilTag with a learned onboard detector. A YOLOv5-class network on a companion computer would let the tracker follow an actual drone without a marker, which is the main thing separating this from a fielded system. After that, flying both the trained tracker and the trained evader on separate physical drones simultaneously is the experiment the project was originally designed around but didn't reach.

Systematic domain randomization across observation noise, wider latency jitter, and randomized vehicle dynamics would make the policy more portable across different hardware without re-tuning the deployment components. And a properly sized quantitative evaluation, enough flights to report tag rate and time-to-capture with confidence intervals rather than just qualitative behavior, would make the real-world claims much stronger.

---

## Citation / reference

If you use or build on this, the full paper is in `docs/finalpaper.pdf`.

Murphy, O. and Narayana, R. "Multi-Agent UAV Capture and Evasion with Adversarial Reinforcement Learning: A Vision-Only Sim-to-Real Pipeline for Low-Cost Quadrotors." TJHSST Computer Systems Research Laboratory, May 2026.

---

## Dependencies

| Package | Purpose |
|---|---|
| stable-baselines3 | PPO implementation |
| pybullet | Physics simulation |
| gymnasium | Environment interface |
| opencv-python | Camera feed processing |
| djitellopy | Tello UDP interface |
| pupil-apriltags | AprilTag detection |
| pyav | H.264 video stream decoding |
| numpy | Everything else |

Python 3.11 was used throughout. Other 3.x versions will probably work but haven't been tested.
