# LinkedIn Post -- Engram Launch

---

What if your AI could learn from experience the way you do -- without forgetting everything it knew yesterday?

I just open-sourced Engram.

It's a brain-inspired AI framework where spiking neurons learn continuously from experience -- no retraining, no fine-tuning, no catastrophic forgetting.

Here's what makes it different from standard ML:

Standard reinforcement learning memorizes exact positions. Drop it in a new environment? It starts from zero. Every. Single. Time.

Engram learns transferable features -- "wall ahead + goal to the right = turn right" -- that work in ANY environment, not just the one it trained on.

We tested this head-to-head:

Same maze. Same rules. Same algorithm (Q-learning).
The only difference? How they represent what they see.

Tabular RL (the standard): memorizes coordinates. Learns fast in one maze. Completely useless in a new one.

Engram: encodes local patterns -- walls, corridors, goal direction, hazards. Knowledge compounds across hundreds of random mazes.

The result: after enough mazes, Engram's success rate keeps climbing while the standard agent resets to zero every time.

(Watch the video -- the teal line is Engram, the red dashed line is standard RL. Same maze, different strategies, diverging outcomes.)

The tech stack is kind of insane:
- Rust core runtime (3,955 lines) with real LIF spiking neurons
- Three-factor STDP with eligibility traces and neuromodulatory signals (dopamine, acetylcholine, norepinephrine, serotonin)
- Spiking DQN trained with surrogate gradients through PyTorch + snnTorch
- 6 brain-inspired modules: sensory cortex, associative memory, predictive error, episodic memory, motor cortex, safety kernel
- Real-time observatory dashboard with 3D brain topology, spike rasters, and live learning visualization
- 87KB WebAssembly binary for browser demos
- 17/17 Rust tests passing
- The whole thing compiles to run on neuromorphic hardware (Loihi, Akida)

Why does this matter?

Because the next wave of AI isn't just bigger models. It's AI that:
- Learns on-device without cloud retraining
- Adapts to new environments without forgetting old ones
- Runs at 100x lower energy on neuromorphic chips
- Has a safety kernel that can veto dangerous actions before they execute

Think: robots that adapt to new warehouses. Drones that learn obstacle patterns in real-time. IoT sensors that detect anomalies without phone-home. Game NPCs that actually learn your playstyle.

The entire spiking RL + cross-environment generalization space is basically unexplored in the literature. Every existing spiking RL paper evaluates on fixed environments only. Nobody has benchmarked spiking networks on procedurally generated environments for transfer learning.

We're targeting ICONS 2026 (ACM International Conference on Neuromorphic Systems) with a paper on this.

Fun fact: the dashboard was originally just "boxes on a screen." It took 4 complete visual redesigns to get it to the point where it actually looks like neurotechnology instead of a generic SaaS dashboard. The final version has CRT scanlines and holographic brain nodes. I regret nothing.

Also fun fact: the first version of the spiking network had a 0% success rate on maze navigation. Zero. The random baseline beat it. Turns out, three-factor STDP alone can't do credit assignment at scale. We had to add surrogate gradient training, non-spiking output neurons for Q-values, experience replay, and optimistic initialization before it started learning. Science is humbling.

The entire project is open source. Apache 2.0.

If you're working on neuromorphic computing, continual learning, adaptive agents, robotics, or edge AI -- I'd love to hear what you'd build with this.

Video: the dashboard running live -- you can see both agents navigating the same random maze simultaneously, with the reward curve and success rate diverging in real-time. The 3D brain topology shows spiking activity across 6 neural regions. The spike raster at the bottom shows 672 neurons firing across all modules.

https://github.com/tejasnaladala/engram

#NeuromorphicComputing #SpikingNeuralNetworks #ReinforcementLearning #OpenSource #BrainInspiredAI #ContinualLearning #EdgeAI #Robotics
