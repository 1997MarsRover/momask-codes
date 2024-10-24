import os
import numpy as np
from pyquaternion import Quaternion
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BVHProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.joint_names = []
        self.joint_parents = []
        self.joint_offsets = []
        self.channels = []
        self.motion_data = None
        self.frame_time = None
        self.num_frames = 0
        self.bracket_count = 0

    def euler_to_quaternion(self, euler_angles):
        """Convert euler angles to quaternion using intrinsic XYZ rotation order."""
        # Convert to radians
        x, y, z = np.radians(euler_angles)
        
        # Calculate rotation matrix components
        cx, sx = np.cos(x/2), np.sin(x/2)
        cy, sy = np.cos(y/2), np.sin(y/2)
        cz, sz = np.cos(z/2), np.sin(z/2)

        # Calculate quaternion components
        w = cx*cy*cz + sx*sy*sz
        x = sx*cy*cz - cx*sy*sz
        y = cx*sy*cz + sx*cy*sz
        z = cx*cy*sz - sx*sy*cz

        return [w, x, y, z]

    def parse_bvh(self):
        with open(self.file_path, 'r') as file:
            content = file.readlines()
        
        hierarchy_end = next(i for i, line in enumerate(content) if 'MOTION' in line.strip())
        self._parse_hierarchy(content[:hierarchy_end])
        self._parse_motion(content[hierarchy_end:])

    def _parse_hierarchy(self, hierarchy_lines):
        stack = []
        current_joint = None
        
        for line in hierarchy_lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()

            if 'ROOT' in line or 'JOINT' in line:
                joint_name = parts[-1]
                self.joint_names.append(joint_name)
                self.joint_parents.append(len(stack) - 1 if stack else -1)
                stack.append(joint_name)
                current_joint = joint_name
                self.bracket_count += 1

            elif 'End Site' in line:
                self.bracket_count += 1
                continue

            elif 'OFFSET' in line:
                if current_joint is not None or len(stack) > 0:
                    offset = [float(x) for x in parts[1:4]]
                    self.joint_offsets.append(offset)

            elif 'CHANNELS' in line:
                num_channels = int(parts[1])
                channels = parts[2:2+num_channels]
                self.channels.append(channels)

            elif '}' in line:
                self.bracket_count -= 1
                if stack:
                    stack.pop()
                    if stack:
                        current_joint = stack[-1]
                    else:
                        current_joint = None

        if self.bracket_count != 0:
            logging.warning(f"Bracket mismatch in hierarchy: {self.bracket_count}")

    def _parse_motion(self, motion_lines):
        motion_lines = [line for line in motion_lines if line.strip()]
        
        self.num_frames = int(motion_lines[1].split()[-1])
        self.frame_time = float(motion_lines[2].split()[-1])
        
        motion_data = []
        for line in motion_lines[3:]:
            if line.strip():
                values = [float(x) for x in line.split()]
                motion_data.append(values)
        self.motion_data = np.array(motion_data)

    def preprocess_data(self):
        processed_data = []
        
        for frame in self.motion_data:
            frame_data = []
            index = 0
            
            for i, channels in enumerate(self.channels):
                # Handle root joint position
                if i == 0:  # Root joint
                    position = frame[index:index+3].tolist()
                    frame_data.extend(position)
                    index += 3
                
                # Handle rotations
                rot_channels = [ch for ch in channels if 'rotation' in ch.lower()]
                if len(rot_channels) == 3:
                    euler_angles = frame[index:index+3]
                    try:
                        # Use our custom euler to quaternion conversion
                        quaternion = self.euler_to_quaternion(euler_angles)
                        frame_data.extend(quaternion)
                    except Exception as e:
                        logging.warning(f"Quaternion conversion failed: {str(e)}. Using identity quaternion.")
                        frame_data.extend([1, 0, 0, 0])
                    index += 3
                else:
                    frame_data.extend([1, 0, 0, 0])
            
            processed_data.append(frame_data)
        
        return np.array(processed_data)

    def save_processed_data(self, output_path):
        processed_data = self.preprocess_data()
        np.save(output_path, processed_data)

    def get_joint_hierarchy(self):
        return {
            'names': self.joint_names,
            'parents': self.joint_parents,
            'offsets': self.joint_offsets
        }

def process_bvh_files(bvh_dir, output_base_dir):
    motion_dir = os.path.join(output_base_dir, "new_joint_vecs")
    os.makedirs(motion_dir, exist_ok=True)

    all_processed_data = []

    for filename in os.listdir(bvh_dir):
        if filename.endswith(".bvh"):
            file_path = os.path.join(bvh_dir, filename)
            logging.info(f"Processing {filename}...")

            try:
                processor = BVHProcessor(file_path)
                processor.parse_bvh()
                processed_data = processor.preprocess_data()

                output_filename = os.path.splitext(filename)[0] + ".npy"
                output_path = os.path.join(motion_dir, output_filename)
                np.save(output_path, processed_data)

                all_processed_data.append(processed_data)
                logging.info(f"Successfully processed {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                continue

    return all_processed_data, motion_dir

def calculate_stats(all_processed_data, output_base_dir):
    all_data = np.concatenate(all_processed_data, axis=0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)

    np.save(os.path.join(output_base_dir, "Mean.npy"), mean)
    np.save(os.path.join(output_base_dir, "Std.npy"), std)

def create_train_val_split(motion_dir, output_base_dir, train_ratio=0.8):
    all_files = [f for f in os.listdir(motion_dir) if f.endswith('.npy')]
    np.random.shuffle(all_files)
    split = int(train_ratio * len(all_files))

    train_files = all_files[:split]
    val_files = all_files[split:]

    with open(os.path.join(output_base_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_files))

    with open(os.path.join(output_base_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_files))

if __name__ == "__main__":
    bvh_dir = "/mnt/c/Users/Signvrse/Desktop/bvh"
    output_base_dir = "./dataset/ProcessedMotion"

    all_processed_data, motion_dir = process_bvh_files(bvh_dir, output_base_dir)

    if not all_processed_data:
        logging.error("No files were successfully processed. Exiting.")
        exit(1)

    calculate_stats(all_processed_data, output_base_dir)
    create_train_val_split(motion_dir, output_base_dir)

    logging.info("Processing complete. Dataset structure created.")
