# Step 12 --- Building a Realistic VLA Robot in Simulation

So far, your VLA robot has operated in a 2D grid world. Now we'll move
toward a physically simulated robot that sees its environment through a
camera and controls its movements through robot actions.

The goal is to bridge the gap between your miniature VLA and modern
systems such as OpenVLA.

## Step 12 curriculum

  -----------------------------------------------------------------------
  Lesson                  Topic                   What you'll build
  ----------------------- ----------------------- -----------------------
  12.1                    Introduction to robot   Understand the robot,
                          simulation              world, physics, and
                                                  sensors

  12.2                    Robot models and URDF   Load a robot arm into
                                                  simulation

  12.3                    Camera observations     Capture RGB images from
                                                  the robot's camera

  12.4                    Coordinate systems      Understand world,
                                                  robot, and camera
                                                  coordinates

  12.5                    Robot control           Control joints and
                                                  end-effector positions

  12.6                    Grasping objects        Pick up and move
                                                  objects

  12.7                    Language-conditioned    Execute instructions
                          tasks                   such as "pick up the
                                                  red cube"

  12.8                    Vision-to-action        Connect camera
                          pipeline                observations to robot
                                                  actions

  12.9                    Closed-loop control     Observe, predict,
                                                  execute, and observe
                                                  again

  12.10                   Mini VLA robot project  Build a simulated
                                                  language-controlled
                                                  robot arm
  -----------------------------------------------------------------------

## Recommended tools

We'll use:

-   **PyBullet** --- physics-based robot simulation.
-   **URDF** --- robot model descriptions.
-   **PyTorch** --- neural networks.
-   **OpenCV** --- camera image processing.
-   **Pygame** --- optional visualization and interaction.

We'll continue with PyBullet initially because you've already worked
with it in Step 10.

## Final project

Build a simulated robot arm that receives instructions such as:

-   "Pick up the red cube."
-   "Move the blue object to the left."
-   "Place the red cube on the table."

The complete pipeline:

``` text
       Human instruction
              |
              v
        Language Encoder
              |
              v
Robot Camera --> Vision Encoder
              |
              v
       Multimodal Fusion
              |
              v
        Action Prediction
              |
              v
        Robot Controller
              |
              v
        Physics Simulator
              |
              v
       New Camera Observation
              |
              +------> Repeat
```

## Where this leads

After Step 12, continue with:

  Course    Topic
  --------- ------------------------------------------
  Step 13   Learning from robot demonstrations
  Step 14   Action chunking and Transformer policies
  Step 15   Fine-tuning pretrained VLA models
  Step 16   OpenVLA and LeRobot
  Step 17   Real robot deployment

## Recommended starting point

Start Step 12.1 with a simulated Franka Panda robot arm, then gradually
teach it to see, understand, grasp, and manipulate objects.

This is where the VLA course starts becoming much closer to real robot
learning.

## The connection to the rest of VLA

In the examples here, the model's input is a simple numeric description of the
robot and its target, and its output is a movement. In a later VLA system, the
input can include camera images and a language instruction, while the output
is still an action the robot can take. The scale and inputs become more
complex, but the basic question introduced in Step 1 remains: **how can a
model use what it is given to choose an appropriate action?**


# Step 2: Computer Vision — Teaching a Robot to See

In Step 1, our robot received numerical information about its position and target. However, real robots usually don't know their surroundings directly. They need sensors, especially cameras, to understand the environment.

Step 2 introduces computer vision: the process of extracting useful information from images.

## Why does a VLA course need computer vision?

A VLA system needs to understand what a robot sees before deciding what action to take.

For example, consider a robot receiving the instruction:

"Pick up the red cup on the table."

Before performing this task, the robot must identify objects, recognize their positions, and understand which object matches the instruction.

This step introduces the fundamental image-processing and deep-learning techniques used in robotic perception.

## Step 2 learning path

1. **Understand images as numerical data.**

   Introduce `image_basics.py`. Load an image using OpenCV or PIL and inspect its dimensions, channels, and pixel values.

   Learn how computers represent images as arrays and tensors.

2. **Learn basic image processing.**

   Introduce `image_processing.py`. Experiment with resizing, cropping, normalization, color conversion, and image transformations.

   Understand why images need preprocessing before being passed into neural networks.

3. **Train a simple image classifier.**

   Introduce `image_classification.py`. Train a small CNN using a dataset such as MNIST or CIFAR-10.

   Learn how a model identifies visual patterns and classifies images.

4. **Understand convolutional neural networks.**

   Introduce `cnn_exercise.py`. Experiment with convolution, pooling, feature maps, and classification layers.

   Understand how neural networks extract visual features.

5. **Introduce object detection.**

   Introduce `object_detection.py`. Use a pretrained YOLO model to detect objects in an image.

   Learn the difference between image classification and object detection.

6. **Connect perception to robot environments.**

   Introduce `vision_simulator.py`. Display a simulated robot environment and use a vision model to identify objects.

   The goal is to move from numerical robot states to visual observations.

By the end of Step 2, you should understand how a robot can process camera images, extract visual features, classify objects, and detect objects in its environment.

# Step 3: Natural Language Processing — Teaching a Robot to Understand Instructions

In Step 2, we taught the robot to recognize objects. However, recognizing objects is not enough. A robot also needs to understand what a human wants it to do.

Step 3 introduces Natural Language Processing (NLP) and language representations.

## Why does a VLA course need language understanding?

Consider these instructions:

- Pick up the red cup.
- Move the cup to the left.
- Place the cup on the table.

A robot needs to understand the meaning of these instructions and extract the information required to perform the task.

This step introduces how computers represent and process human language.

## Step 3 learning path

1. **Understand how text becomes numbers.**

   Introduce `text_basics.py`.

   Learn tokenization, vocabulary, token IDs, and embeddings.

   Understand why neural networks cannot directly process human-readable sentences.

2. **Learn word embeddings.**

   Introduce `word_embeddings.py`.

   Explore how words can be represented as numerical vectors and how similar words can have related representations.

3. **Understand sentence embeddings.**

   Introduce `sentence_embeddings.py`.

   Use a pretrained sentence-transformer model to convert instructions into vectors.

   Compare the embeddings of different robot instructions.

4. **Introduce attention mechanisms.**

   Introduce `attention.py`.

   Learn how attention allows a model to focus on relevant parts of an input sequence.

   Understand why attention is important for modern language models.

5. **Understand Transformers.**

   Introduce `transformer_basics.py`.

   Explore positional embeddings, self-attention, feed-forward networks, and Transformer blocks.

6. **Build a simple instruction classifier.**

   Introduce `instruction_classifier.py`.

   Train a small model to classify instructions into actions such as:

   - Pick
   - Place
   - Move
   - Rotate

7. **Connect language to the robot simulator.**

   Introduce `language_controller.py`.

   Allow a user to provide a text instruction and convert it into a corresponding robot command.

By the end of Step 3, you should understand how language is tokenized, embedded, processed by Transformers, and converted into information that can guide robot behavior.

# Step 4: Multimodal Learning — Connecting Vision and Language

In Step 2, the robot learned to see.

In Step 3, the robot learned to process language.

Now we combine these two capabilities.

## Why does a VLA course need multimodal learning?

A robot should not understand images and language independently. It needs to connect them.

For example:

Input image: A table containing a red cup, blue cup, and green bottle.

Instruction: "Pick up the blue cup."

The robot needs to connect the word "blue cup" with the correct object in its visual observation.

This is the foundation of vision-language models.

## Step 4 learning path

1. **Understand multimodal representations.**

   Introduce `multimodal_basics.py`.

   Learn how images and text can be converted into numerical representations.

2. **Explore CLIP.**

   Introduce `clip_basics.py`.

   Use a pretrained CLIP model to compare image and text embeddings.

   Experiment with matching an image against several text descriptions.

3. **Understand image-text alignment.**

   Introduce `image_text_alignment.py`.

   Learn how contrastive learning brings matching image-text representations closer together while separating unrelated pairs.

4. **Build a simple image-text matching system.**

   Introduce `image_text_matching.py`.

   Given an image and multiple instructions, select the instruction that best matches the image.

5. **Understand visual-language models.**

   Introduce `vlm_basics.py`.

   Explore how vision encoders and language models work together.

   Introduce models such as LLaVA and Qwen-VL.

6. **Build a visual instruction understanding demo.**

   Introduce `visual_instruction.py`.

   Provide an image and a natural-language question, then use a pretrained vision-language model to interpret the scene.

By the end of Step 4, you should understand how visual information and language can be represented in a shared multimodal system.

# Step 5: Robot Actions — From Understanding to Control

So far, our robot can recognize objects and understand instructions.

However, understanding a task does not automatically mean the robot knows how to physically perform it.

Step 5 introduces robotic action representations and control.

## Why does a VLA course need robot control?

Consider the instruction:

"Pick up the red cup."

A robot needs to determine:

- Where is the cup?
- Where should the robot move?
- What position should the end effector reach?
- When should the gripper close?
- How should the robot move the object?

These decisions require converting task understanding into physical actions.

## Step 5 learning path

1. **Understand robot coordinate systems.**

   Introduce `coordinate_systems.py`.

   Learn world coordinates, robot coordinates, local coordinates, and transformations.

2. **Understand robot kinematics.**

   Introduce `kinematics.py`.

   Explore forward kinematics and inverse kinematics.

   Understand how joint positions determine end-effector positions.

3. **Understand robot action representations.**

   Introduce `action_representations.py`.

   Explore different ways of representing actions:

   - Joint positions
   - Joint velocities
   - End-effector positions
   - End-effector rotations
   - Gripper commands

4. **Build a simple robot controller.**

   Introduce `robot_controller.py`.

   Use numerical commands to control a simulated robotic arm.

5. **Understand feedback control.**

   Introduce `feedback_control.py`.

   Learn how a robot observes its current state, compares it with a target state, and adjusts its movements.

6. **Build a pick-and-place task.**

   Introduce `pick_and_place.py`.

   Combine object positions, robot movements, and gripper commands to complete a basic manipulation task.

By the end of Step 5, you should understand how robot actions are represented and how controllers translate desired movements into physical robot commands.

# Step 6: Imitation Learning — Learning Robot Skills from Demonstrations

In Step 1, we trained models using manually generated numerical examples.

Now we introduce a more realistic robotics learning approach: imitation learning.

## Why does a VLA course need imitation learning?

Writing a rule for every possible robot situation is difficult.

Instead, we can demonstrate a task to a robot and teach it to reproduce the demonstrated behavior.

For example:

A human demonstrates how to pick up a cup and place it inside a box.

The robot records the observations and actions during the demonstration. It then learns to reproduce similar behavior.

This is one of the fundamental approaches used to train robotic manipulation policies.

## Step 6 learning path

1. **Understand supervised learning for robot actions.**

   Introduce `supervised_robot_learning.py`.

   Learn how observations and corresponding actions become training examples.

2. **Understand behavior cloning.**

   Introduce `behavior_cloning.py`.

   Train a neural network to predict robot actions from robot observations.

3. **Create a demonstration dataset.**

   Introduce `collect_demonstrations.py`.

   Record observations, robot states, and actions while performing a simulated task.

4. **Build a robot dataset.**

   Introduce `robot_dataset.py`.

   Organize collected demonstrations into training samples.

   Understand observations, actions, timestamps, and episode boundaries.

5. **Train a manipulation policy.**

   Introduce `train_policy.py`.

   Train a model using collected demonstrations and evaluate its performance in simulation.

6. **Evaluate the learned policy.**

   Introduce `evaluate_policy.py`.

   Compare the model's predictions with expert demonstrations and measure task success rates.

By the end of Step 6, you should understand how robots can learn skills from demonstrations instead of relying entirely on manually programmed rules.

# Step 7: Action Chunking and Transformers — Learning Sequences of Robot Movements

In Step 6, the robot learned to predict actions from observations.

However, robot manipulation often requires a sequence of coordinated movements rather than a single action.

Step 7 introduces sequence modeling and action chunking.

## Why does a VLA course need action chunking?

Imagine a robot picking up a cup.

The task involves several movements:

1. Move toward the cup.
2. Align the gripper.
3. Open the gripper.
4. Reach the cup.
5. Close the gripper.
6. Lift the cup.

Predicting one action at a time can be inefficient and may produce unstable behavior.

Instead, a model can predict a sequence of future actions.

This idea is important in modern robot learning architectures such as Action Chunking with Transformers (ACT).

## Step 7 learning path

1. **Understand sequential robot data.**

   Introduce `sequential_data.py`.

   Learn how observations and actions are organized over time.

2. **Understand temporal dependencies.**

   Introduce `temporal_learning.py`.

   Explore why previous observations and actions can help predict future movements.

3. **Learn action chunking.**

   Introduce `action_chunking.py`.

   Train a model to predict multiple future robot actions from one observation.

4. **Understand Transformer-based action prediction.**

   Introduce `action_transformer.py`.

   Explore how Transformers can model sequences of robot actions.

5. **Implement a simplified ACT model.**

   Introduce `act_basics.py`.

   Build a small educational implementation of Action Chunking with Transformers.

6. **Evaluate action prediction.**

   Introduce `evaluate_action_chunks.py`.

   Compare single-step prediction with action-chunk prediction in a simulated task.

By the end of Step 7, you should understand temporal modeling, action sequences, and how Transformers can predict multiple future robot movements.

# Step 8: Building a Vision-Language-Action System

We have now introduced the three essential components of a VLA system:

- Vision: understanding the robot's surroundings.
- Language: understanding human instructions.
- Action: generating robot movements.

Step 8 brings these components together.

## Why does a VLA course need this step?

A complete VLA system connects perception, instruction understanding, and action prediction.

For example:

Camera image: A table containing several objects.

Language instruction: "Pick up the red block and place it inside the container."

The system must interpret the image, understand the instruction, and generate robot actions that accomplish the task.

## Step 8 learning path

1. **Understand VLA architecture.**

   Introduce `vla_basics.py`.

   Explore the overall architecture of a VLA model.

   Understand the roles of the vision encoder, language model, multimodal fusion, and action decoder.

2. **Understand the VLA data format.**

   Introduce `vla_dataset.py`.

   Learn how robot demonstrations can contain:

   - Camera observations
   - Language instructions
   - Robot states
   - Robot actions

3. **Build a simple vision-language-action pipeline.**

   Introduce `simple_vla.py`.

   Combine a pretrained vision encoder, language embeddings, and an action prediction network.

   Start with a small dataset and a simple simulated task.

4. **Train a multimodal action policy.**

   Introduce `train_vla.py`.

   Train a model using image observations, language instructions, and corresponding robot actions.

5. **Run the VLA policy in simulation.**

   Introduce `vla_simulator.py`.

   Connect the trained model to a simulated robotic environment.

   Test whether the robot can perform different tasks based on natural-language instructions.

6. **Evaluate the VLA system.**

   Introduce `evaluate_vla.py`.

   Measure task success rates, action prediction errors, and generalization to different scenes and instructions.

By the end of Step 8, you should understand the fundamental architecture and training process behind a simple Vision-Language-Action system.

# Step 9: Advanced VLA — Understanding Modern Architectures

After building a basic VLA system, the next step is understanding how modern research systems work.

This step introduces existing VLA architectures and the engineering techniques used to train them.

## Step 9 learning path

1. **Explore pretrained vision-language models.**

   Introduce `pretrained_vlm.py`.

   Experiment with existing vision-language models and understand their visual and language capabilities.

2. **Understand parameter-efficient fine-tuning.**

   Introduce `lora.py`.

   Learn how LoRA and other parameter-efficient methods reduce the computational requirements of model adaptation.

3. **Explore diffusion-based action generation.**

   Introduce `diffusion_policy.py`.

   Understand how diffusion models can generate robot action trajectories.

4. **Explore Action Chunking with Transformers.**

   Introduce `act_advanced.py`.

   Study the architecture and training strategy of ACT in greater detail.

5. **Explore OpenVLA.**

   Introduce `openvla_basics.py`.

   Understand how an open-source VLA model combines vision, language, and robot action prediction.

6. **Explore model fine-tuning.**

   Introduce `finetune_vla.py`.

   Learn how pretrained VLA models can be adapted to specific robots and manipulation tasks.

7. **Compare different VLA architectures.**

   Introduce `vla_comparison.py`.

   Compare different approaches to visual encoding, language processing, action representation, and policy learning.

By the end of Step 9, you should understand the major architectural ideas behind modern VLA research and how to experiment with existing models.

# Step 10: Real Robot Deployment — From Simulation to Reality

The final step moves from educational simulations to real robotic systems.

A model that works in simulation does not automatically work on a physical robot.

This step introduces the practical challenges of deploying learned policies.

## Why does a VLA course need real-world deployment?

A physical robot introduces challenges that are difficult to reproduce perfectly in simulation:

- Camera calibration
- Sensor noise
- Robot communication delays
- Motor limitations
- Unexpected object positions
- Gripper failures
- Real-world environmental changes

A successful VLA system must handle these conditions safely and reliably.

## Step 10 learning path

1. **Understand the robot software interface.**

   Introduce `robot_interface.py`.

   Learn how a trained policy communicates with a robot controller.

2. **Understand camera integration.**

   Introduce `camera_interface.py`.

   Capture images from real robot cameras and prepare them for model inference.

3. **Connect the policy to a robot.**

   Introduce `deploy_policy.py`.

   Connect model predictions to the robot's control interface.

4. **Understand real-time inference.**

   Introduce `realtime_inference.py`.

   Explore inference latency, observation frequency, action frequency, and control loops.

5. **Introduce safety mechanisms.**

   Introduce `robot_safety.py`.

   Implement workspace limits, action limits, emergency stopping, and basic execution checks.

6. **Perform a real-world manipulation task.**

   Introduce `real_robot_demo.py`.

   Run a simple pick-and-place task using a trained policy and a physical robot.

7. **Evaluate real-world performance.**

   Introduce `deployment_evaluation.py`.

   Compare simulation and real-world performance and identify possible causes of failure.

By the end of Step 10, you should understand the process of deploying and evaluating a learned VLA policy on a physical robot.

# Final Learning Goal

The VLA Crash Course follows this progression:

| Step | Topic | Main capability |
|---|---|---|
| 1 | Deep Learning and Robot Control | Learn how models predict actions |
| 2 | Computer Vision | Teach a robot to see |
| 3 | Natural Language Processing | Teach a robot to process instructions |
| 4 | Multimodal Learning | Connect vision and language |
| 5 | Robot Actions and Control | Convert decisions into movements |
| 6 | Imitation Learning | Learn from robot demonstrations |
| 7 | Action Chunking and Transformers | Predict sequences of movements |
| 8 | Building a VLA System | Connect vision, language, and action |
| 9 | Advanced VLA Architectures | Understand and fine-tune modern models |
| 10 | Real Robot Deployment | Execute learned policies on physical robots |

The ultimate objective is to understand the complete learning pipeline:

**Camera Images + Language Instructions → Multimodal Understanding → Action Prediction → Robot Execution**

By completing these lessons, you will progress from training a small neural network to understanding, building, training, and deploying a Vision-Language-Action system.
