import subprocess
import os

# Base configuration
base_path = "/Users/barisozkan/PycharmProjects/CMPE489/exp2/savepath/"
script_dir = "/Users/barisozkan/PycharmProjects/CMPE489/exp2"
gamma = "0.99"
reward_weight = "0.1"

# Task ranges
task_ids = [7]      # 0 to 11
task_idxs = [0]  # 0

for task_id in task_ids:
    for task_idx in task_idxs:
        print(f"\n▶️ Running for TaskID={task_id}, TaskIdx={task_idx}, Gamma={gamma}")

        # Create save folder
        folder_path = os.path.join(base_path, str(task_id), str(task_idx))
        os.makedirs(folder_path, exist_ok=True)

        # Step 1: Run inference.py
        try:
            subprocess.run([
                "python3",
                os.path.join(script_dir, "inference.py"),
                base_path,
                str(task_id),
                str(task_idx),
                gamma
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Error running inference.py for TaskID={task_id}, TaskIdx={task_idx}")
            continue

        # Step 2: Run count_traj.py
        try:
            subprocess.run([
                "python3",
                os.path.join(script_dir, "count_traj.py"),
                base_path,
                str(task_id),
                str(task_idx),
                reward_weight,
                gamma
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Error running count_traj.py for TaskID={task_id}, TaskIdx={task_idx}")
            continue

        # Step 3: Run collect_and_build_csv_file.py
        try:
            subprocess.run([
                "python3",
                os.path.join(script_dir, "collect_and_build_csv_file.py"),
                base_path,
                str(task_id),
                str(task_idx),
                reward_weight,
                gamma
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Error running collect_and_build_csv_file.py for TaskID={task_id}, TaskIdx={task_idx}")
            continue

        print(f"✅ Finished TaskID={task_id}, TaskIdx={task_idx}")
