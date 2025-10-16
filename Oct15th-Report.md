# October 15th Report
# Team Progress Update

## What has been done during the past two weeks?

* **Hanyoung Chung**: Helped write MDP specifications for States and Action parameters. Theorized possible implementations for agent behaviour.
* **Jaden Chang**: Helped finalize the MDP specifications, researched possible metrics/evaluations during further development, read through the gym environment documentation, and planned the TA report with group mates.
* **Nicholas Nicolaev**:  Experimented with the gym environment and created a branch in the GitHub repo. Implemented a tiny heuristic driver to demonstrate interaction and added detailed logging for each step.
* **Ziyang Ling**: Imported the Gym environment into the GitHub repo and started the modified environment (implemented `__init__()` and `step()` functions) with the MDP specification.
* **Mike Lin**: Set up the GitHub repo, tested and became familiar with the Gym environment to help modify, and helped finalize the MDP and formatting of report.

## What are you planning to do in the next two weeks?

* **Hanyoung Chung**: Start implementation on custom parameters for the agent. Research the different algorithms in order to choose the best one for this project.
* **Jaden Chang**: Integrate our new features to the environment and prepare for the env demo such as working on presentations slides.
* **Nicholas Nicolaev**: Finalize the exact wear/pit formulas, progress term, and reward equation. Run multiple episodes and save a CSV with episode reward, pit count, and time stuck. Average results over 3 fixed seeds (0,1,2) to avoid flukes, and add the summary table/plot to the environment demo slides.
* **Ziyang Ling**: Do more research on the algorithms(DQN) and start initial implementation for the project and prepare for env demo slides and video.
* **Mike Lin**: Finalize the environment and reward structure; create environment demo slides, record the demo; begin initial algorithm testing and experiments.


# MDP Specification

## State \(s_t\)
| Component | Range / Domain | Type | Notes |
|--------------------------|----------------|------|-------|
| **Visual frame (RGB image)** — | Box(0, 255, (96, 96, 3), uint8) | image | Each frame is a 96 × 96 RGB top-down view of the car and racetrack (as in CarRacing Gymnasium). |
| **Lateral offset from track center (\(d_t\))** | \([-5, 5]\) m | float | Negative = left; positive = right. |
| **Speed (\(v_t\))** | \([0, 70]\) m/s | float | Forward speed. |
| **Infield flag (\(\text{infield}_t\))** | \(\{0, 1\}\) | int (0/1) | 1 if in infield. |
| **Pit-road flag (\(\text{pitroad}_t\))** | \(\{0, 1\}\) | int (0/1) | 1 if on pit road. |
| **Lap progress (\(\ell_t\))** | \([0, 1]\) | float | Fraction of current lap. |
| **Tire wear (\(w_t\))** | \([0, 1]\) | float | 0.0 = new, 1.0 = fully worn. |
| **Fuel level (\(f_t\))** | \([0, 1]\) | float | 1.0 = full, 0.0 = empty. |
| **Lookahead curvature (\(\kappa_t\))** | \([-0.05, 0.05]\) m\(^{-1}\) | float | Negative = right turn; positive = left. |

---

## Action \(a_t\)

### Continuous control

* Throttle: \([0,,1]\)
* Brake: \([0,,1]\)
* Steering: \([-1,,1]\)
* Pit command: \({0,1}\) *(only valid when `infield = 1` **and** `pitroad = 1`)*

### Discrete control
* 0: do nothing
* 1: steer right
* 2: steer left
* 3: gas
* 4: brake
* 5: pit *(only executes when eligible as above)*

---

## Transition \(P(s_{t+1}\mid s_t, a_t)\)

* Dynamics follow Box2D vehicle physics (deterministic given (s_t, a_t)).
* Track generation/tiles may vary stochastically across episodes.
* Tire wear increases over time; **accelerated** by harsh braking, slip, and over-revving.
* Fuel decreases with time and throttle usage.
* Entering pit resets **tire wear** and **fuel** to optimal levels but incurs a **time cost** (lost progress / frames).

---

## Rewards \(R_t\)

**Base environment (Gymnasium CarRacing):**

* Per-frame penalty: \(-0.1\)
* Tile visitation: \(+\frac{1000}{N}\) for each unique track tile visited (where (N) = total tiles on the track).
* Example: finishing in 732 frames -> \(1000 - 0.1 \times 732 = 926.8\).

**Possible extensions (experiments):**

* Incorporate tire health and fuel efficiency (maybe regularization terms for wear and consumption) to balance pace vs. resource management.

---

## Algorithms (compare 2–3)

* **Q-Learning**

  * [Paper 1 (IEEE)](https://ieeexplore.ieee.org/abstract/document/8441797)
  * [Thesis/Report](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1763095&dswid=-8323)
  * [Paper 2 (IEEE)](https://ieeexplore.ieee.org/abstract/document/10328086)

* **Deep Q-Network (DQN)**

  * [NIPS 2016](https://proceedings.neurips.cc/paper/2016/hash/8d8818c8e140c64c743113f563cf750f-Abstract.html)
  * [IEEE 2019a](https://ieeexplore.ieee.org/abstract/document/8721655)
  * [IEEE 2019b](https://ieeexplore.ieee.org/abstract/document/8946332)

* **PPO (Proximal Policy Optimization)** *(simple, strong baseline for continuous control)*

  * [arXiv:2410.22766](https://arxiv.org/abs/2410.22766)

**Things to Consider:**

* **Fuel**

  * Continuous depletion over time / throttle.
  * Forces long-horizon planning (pit timing over hundreds of steps).

* **Pit Stop**

  * **Delayed reward**: short-term time loss for long-term gains (fresh tires, fuel).
  * Creates a **sparse reward problem**.

* **Tire Wear**

  * Trade-off between aggressive driving (lap time) and degradation (grip loss, more mistakes).
