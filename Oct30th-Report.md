# October 15th Report
# Team Progress Update

## What has been done during the past two weeks?

* **Hanyoung Chung**: Helped create the environment demo video. Researched wrapper method for the custom gym environment to add extra UI that displays the added tire wear and fuel.
* **Jaden Chang**: Prepared the env demo slides and presentation. Had discussions about modifications of the gym environment. Researched off-road boundary detection reward system.
* **Nicholas Nicolaev**: Revised step() and reset() essential function for demo. Implemented CLI controls for episode count and deterministic seeding in the test harness, and added periodic telemetry to reduce console spam and aid in debugging.
* **Ziyang Ling**: Modified the custom gym environment and discussed how to get a working agent. Helped create the env demo slides and video. 
* **Mike Lin**: Modified state representation and added a helper function to map actions in the env, added to-dos/tasks we need to do to finish the environment, and finished env demo slides and video.

## What are you planning to do in the next two weeks?

* **Hanyoung Chung**: Start code implementation on adding UI overlay to the preexisting display window.
* **Jaden Chang**: Planning and working on the agent through the implementation of the algorithms and designing the controller.
* **Nicholas Nicolaev**: Add minimal pit-lane detection and wire termination rules (fuel=0, wear=1, lap complete), finalize reward shaping per our proposal, and run baseline random vs. simple heuristic agents to generate initial comparisons for the result demo.
* **Ziyang Ling**: Start implementing the controller function to get a working agent and prepare for the algos.
* **Mike Lin**: Finish the env by implementing pit-stop rendering and helper functions to calculate the numerical features, and help test and finalize the env for training. Look into the PPO algorithm further, start implementing it, and plan experiments.
