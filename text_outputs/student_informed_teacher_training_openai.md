Below is an audio-narration–ready version of the paper. I have structured it into clear, digestible sections with natural transitions and added [pause] markers between major parts.

–––––––––––––––––––––––––––––––––––––––––
Title and Authors

The title of this paper is “Student-Informed Teacher Training.” The work is authored by Nico Messikommer, Jiaxu Xing, Elie Aljalbout, and Davide Scaramuzza. [pause]

–––––––––––––––––––––––––––––––––––––––––
Introduction

In this work we address a key challenge in imitation learning. Reinforcement learning, which teaches an agent to maximize accumulated rewards by interacting with an environment, can be incredibly effective when the agent learns from high-dimensional inputs such as images. However, when an expert or teacher policy trains with privileged task information, and a student policy is trained to mimic the teacher from more limited observations, a problem emerges due to information asymmetry. In short, even if the teacher is highly capable by virtue of having extra sensory or state information, the student may struggle because it only sees part of the picture. For example, in a mobile robot navigation task, the teacher might observe distances to obstacles in all directions while the student is limited to a forward-facing camera. This asymmetry can lead the teacher to learn behaviors that the student simply cannot imitate.

To overcome this, we propose a novel teacher-student training framework in which the teacher is not trained solely on the task objective, but also by encouraging behaviors that the student can successfully mimic. We modify the teacher’s reward by incorporating a penalty term that increases when there is a significant difference between the teacher’s chosen actions and the actions the student—or a proxy of it—would predict. In doing so, we “nudge” the teacher toward learning strategies that work well even under more limited observation. [pause]

–––––––––––––––––––––––––––––––––––––––––
Background

Reinforcement learning is built around the concept of a Markov Decision Process, where an agent in a given state takes an action according to a policy and subsequently receives rewards. The goal is to optimize the expected cumulative reward over time. Traditional methods rely on extensive exploration. By contrast, imitation learning takes a different approach: a student policy directly learns to mimic an expert policy using a dataset of interactions. However, if the input provided to the student is not as rich as that available to the teacher, the performance gap between the two policies can become problematic.

In our approach, we start with established bounds on the performance gap between teacher and student policies. In simplified terms, the closer the actions predicted by the student are in distribution to those of the teacher, the smaller the expected performance difference. Key equations in this area relate the difference in cumulative rewards to the average divergence between the teacher’s and student’s action distributions. Although the exact formulas are mathematically involved, the main idea is that by minimizing the action divergence – as measured by information-theoretic distances such as Kullback-Leibler divergence – we can improve the student’s performance. [pause]

–––––––––––––––––––––––––––––––––––––––––
Methodology – Student-Informed Teacher Training

At the heart of our work lies a reformulated objective for the teacher policy. Rather than simply maximizing the task reward, the teacher’s objective is extended with a term that accounts for the mismatch between its own action predictions and those of the student. In plain language, the teacher is rewarded when its actions are both effective in the environment and aligned with actions the student could infer. This is achieved by subtracting a penalty – derived from the measure of divergence between the teacher’s action distribution and that of the student – from the teacher’s reward. Additionally, during updates, a gradient term based on this divergence directly adjusts the teacher’s weights so that the teacher is encouraged to reduce its misalignment with the student.

To put it simply, imagine a teacher policy that not only cares about reaching the goal in a maze but also about choosing paths that the student, with its limited view, can follow. This dual objective ensures that the teacher does not over-rely on privileged information, but instead learns strategies that are realistically imitable by the student. [pause]

–––––––––––––––––––––––––––––––––––––––––
Joint Learning Framework

Our framework is implemented using three networks that work in concert:

1. The Teacher Network – This network is responsible for making decisions using privileged observations.
2. The Student Network – This network learns to mimic the teacher but is limited by lower-quality inputs such as images or reduced state information.
3. A Proxy Student Network – This auxiliary network receives the teacher’s observations to approximate what the student might do. It provides a computationally efficient way to estimate the student’s action distribution without the cost of simulating high-dimensional student inputs.

These networks share a common action decoder. By sharing layers that translate high-level features into actions, both the teacher and student are encouraged to develop similar representations. This shared architecture is particularly important because it helps bridge the gap between the two policies. [pause]

The training process is broken down into three alternating phases:

• Roll-out Phase: The teacher interacts with the environment and collects trajectories. In addition to the normal task rewards, a penalty based on the divergence between teacher and proxy student actions is subtracted. This phase is crucial, as it drives the teacher to explore and avoid states where the student is likely to perform badly. A key aspect of this stage is that it stores a subset of expert states that will later be used to align the student and proxy representations.

• Policy Update Phase: Using the popular Proximal Policy Optimization algorithm, the teacher’s network weights are updated. The loss function in this phase consists of two parts: the standard policy gradient for task reward and an additional term related to the divergence between teacher and proxy student actions. In continuous action spaces, the divergence simplifies to a term that reflects the weighted difference between the predicted mean actions. This adjustment can even influence how confident the teacher is in its actions. For example, if the teacher is very certain (a low spread in its Gaussian output) but still misaligned with the student, the update will increase exploration by reducing that certainty.

• Alignment Phase: In this phase, we directly align the student network with the teacher network, using paired observations from the environments. The student’s features are encouraged to match those of the teacher through a simple L-one loss on the activations, with gradients flowing only into the student encoder. Similarly, the proxy student network is aligned with the student. This step does not change the teacher, preserving the benefits of the previously learned task-related behavior while refining the student’s ability to imitate it.

Together, these phases ensure that both task performance and policy alignment are optimized simultaneously. [pause]

–––––––––––––––––––––––––––––––––––––––––
Experimental Setup and Results

We evaluated our approach on three distinct scenarios to showcase its broad applicability:

1. Color Maze Navigation:
   In this setting, the agent’s objective is to navigate a maze from a start point to a goal. The maze is represented on a grid with cells annotated as either empty, a dangerous "lava" cell, or a viable path. The teacher has access to rich observations, including the type of each neighboring cell, whereas the student receives a simplified version where both lava and paths are viewed as occupied. Visual diagrams in the paper illustrate a comparison between a teacher that finds an optimal maze path and a corresponding student policy that may fail when the teacher’s behavior is too complex to imitate. Importantly, our method trains the teacher to avoid overly relying on shortcuts that the student cannot see, resulting in both policies converging to a solution where the teacher deliberately navigates around the maze. [pause]

2. Vision-Based Obstacle Avoidance with a Quadrotor:
   This experiment is designed around agile quadrotor flight through an environment with static obstacles. The teacher policy, which receives state-based inputs including velocity, orientation, and relative distances to obstacles, is compared against a vision-based student that receives images from a limited field-of-view camera. Figures accompanying the text show teacher trajectories where the quadrotor adjusts its camera direction during flight to capture critical environmental information. Our approach, particularly with alignment, resulted in a substantially higher obstacle avoidance success rate. For instance, compared to behavior cloning and other imitation learning baselines, our method achieved a success rate that was nearly five to six times higher. Graphs in the paper display how both teacher and student returns improve over training epochs with proper alignment. [pause]

3. Vision-Based Manipulation:
   In a more complex task, a robot arm is trained to open a drawer. The teacher has full access to the precise relative positions of the drawer and the arm, whereas the student relies on a camera view that may show the robot arm partially occluding the drawer handle. The visual materials in this section include images where one can see how a teacher trained without alignment tends to block the view, while our method encourages a camera-aware behavior. The teacher learns to adjust its configuration such that critical visual cues, like the red drawer handle, remain clearly visible. Success rates reported indicate that students trained under the aligned framework can achieve significantly higher task success compared to standard imitation learning methods. [pause]

Across these tasks, tables provided in the paper underscore that, compared to baselines such as Behavior Cloning, DAgger, and several hybrid approaches, our method consistently improves the success rates as well as overall returns achieved by the student policy. Ablation studies further confirm that both the additional divergence penalty and the shared action decoder play critical roles in the observed performance improvements. We also measure perception-related metrics – such as the angle between the quadrotor’s heading and its velocity direction, as well as the number of obstacles visible in the camera’s view – and show that our approach leads to more perception-aware behavior. [pause]

–––––––––––––––––––––––––––––––––––––––––
Related Work

This work builds on a rich legacy of research in both imitation learning and reinforcement learning. Traditional approaches like Behavior Cloning focus on supervised learning to mimic expert actions, while methods such as DAgger incorporate iterative expert feedback using the student’s own roll-outs. Other recent studies have addressed the problem of asymmetric observability by fusing reinforcement learning objectives with imitation learning cues.

Our approach differs by actively regularizing the teacher policy so that it learns behaviors that the student can replicate even under partial observability. In our framework, the teacher not only learns through standard reinforcement learning but also integrates a form of direct supervision based on the divergence between its own action distribution and that of the student. This influence from the student is incorporated through policy gradients and explicit alignment steps, leading to improved cooperation between the networks. [pause]

–––––––––––––––––––––––––––––––––––––––––
Conclusion

To conclude, our paper tackles a fundamental issue in privileged imitation learning – that of teacher-student asymmetry. We show that by modifying the teacher’s training objective to include a penalty for producing actions that the student cannot reliably infer, the teacher is encouraged to adopt behaviors that are easier for the student to imitate. With a joint training framework that alternates between standard roll-outs, gradient-based policy updates, and careful alignment of the student representations, we achieve markedly higher success rates on tasks ranging from maze navigation to complex vision-based control in quadrotor flight and manipulation.

This approach not only demonstrates strong empirical performance but also opens up new avenues for research where teaching and imitation are more tightly integrated. Overall, our method suggests a promising path forward for scenarios where the learner’s perceptual capabilities are limited relative to those available during training.

–––––––––––––––––––––––––––––––––––––––––
Closing Remarks

In summary, this work represents a substantial step towards reducing the gap between expert and learner policies when the available observation spaces differ significantly. By carefully designing teacher training to incorporate student feedback and alignment, we pave the way for more robust and deployable imitation learning systems in complex real-world environments. [pause]

Thank you for listening. This concludes our summary of the paper “Student-Informed Teacher Training.”