# Outreach Email Templates

## To: Korneel Van den Berghe (TU Delft)

Subject: Spiking RL cross-environment generalization -- building on your NeurIPS 2025 work

Dear Korneel,

I read your NeurIPS 2025 oral on adaptive surrogate gradients for spiking RL with great interest. Your finding that slope scheduling yields 2.1x improvement is directly relevant to a project I'm working on.

I'm investigating whether spiking neural networks exhibit different generalization properties than ANNs when tested on procedurally generated environments -- specifically, whether a spiking DQN trained on N random mazes generalizes better to unseen mazes than an equivalent MLP DQN. Surprisingly, I cannot find any prior work that benchmarks spiking RL generalization across environment distributions.

I have a working implementation (Rust core + Python/snnTorch training, open-source at github.com/tejasnaladala/engram) and initial benchmark results comparing Spiking DQN vs MLP DQN vs Tabular Q-Learning with proper seed variance.

I'm targeting ICONS 2026 (April deadline) and would value your perspective on:
1. Whether adaptive surrogate gradient slopes interact with distribution shift across environments
2. Any methodological pitfalls in comparing SNN vs ANN generalization
3. Whether this gap in the literature is genuinely underexplored or if I'm missing relevant work

Happy to share the draft and experiment results. No obligation of course.

Best regards,
Tejas Naladala


## To: Jason Eshraghian (UC Santa Cruz, snnTorch)

Subject: Cross-environment RL benchmark for spiking networks -- potential NeuroBench contribution

Dear Prof. Eshraghian,

I'm building on snnTorch for a project investigating spiking neural network generalization in reinforcement learning -- specifically, whether spiking DQN agents transfer learned navigation policies to unseen procedurally generated environments better than equivalent ANNs.

The project (github.com/tejasnaladala/engram) includes:
- Spiking DQN with surrogate gradients via snnTorch (LIF + non-spiking LI output)
- Procedurally generated maze environments with ego-centric observations
- Head-to-head comparison: Spiking DQN vs MLP DQN vs Tabular Q-Learning
- Zero-shot transfer evaluation on held-out environment distributions

I'm targeting ICONS 2026 and believe this could also be useful as a cross-environment generalization benchmark for NeuroBench. No existing spiking RL benchmark tests generalization across procedural environments.

Would you be open to a brief conversation about:
1. Whether this benchmark gap aligns with NeuroBench priorities
2. Implementation best practices for fair SNN vs ANN comparison
3. Feedback on the experimental design

Best regards,
Tejas Naladala
