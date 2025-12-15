import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Optional, Tuple, List


class MovingMNIST(Dataset):
    
    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        num_videos: int = 10000,
        sequence_length: int = 20,
        frame_size: int = 64,
        num_digits: int = 2,
        digit_size: int = 28,
        velocity_range: Tuple[float, float] = (2.0, 5.0),
        transform: Optional[transforms.Compose] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
        download: bool = True
    ):
        """
        Initialize Moving MNIST dataset.
        
        Args:
            root: Root directory for MNIST data
            train: If True, use training set; else use test set
            num_videos: Number of videos to generate
            sequence_length: Number of frames per video
            frame_size: Size of the canvas (frame_size x frame_size)
            num_digits: Number of digits per video
            digit_size: Size of each digit (should match MNIST)
            velocity_range: (min_velocity, max_velocity) for digit movement
            transform: Optional transforms to apply to videos
            seed: Random seed for reproducibility
            deterministic: If True, same index always gives same video
            download: If True, download MNIST if not found
        """
        super().__init__()
        
        self.num_videos = num_videos
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.num_digits = num_digits
        self.digit_size = digit_size
        self.velocity_range = velocity_range
        self.transform = transform
        self.deterministic = deterministic
        self.seed = seed
        
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )
        
        # Pre-compute digit images for efficiency
        self.digit_images = []
        self.digit_labels = []
        for img, label in self.mnist:
            self.digit_images.append(img.squeeze(0).numpy())  # [28, 28]
            self.digit_labels.append(label)
        self.digit_images = np.array(self.digit_images)
        self.digit_labels = np.array(self.digit_labels)
        
        # Create label-to-indices mapping for controlled generation
        self.label_to_indices = {i: np.where(self.digit_labels == i)[0] 
                                  for i in range(10)}
        
        # Set random state
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
            
    def __len__(self) -> int:
        return self.num_videos
    
    def _get_random_digit(self, rng: np.random.RandomState, 
                          label: Optional[int] = None) -> Tuple[np.ndarray, int]:
        if label is not None:
            idx = rng.choice(self.label_to_indices[label])
        else:
            idx = rng.randint(len(self.digit_images))
        return self.digit_images[idx], self.digit_labels[idx]
    
    def _get_random_position(self, rng: np.random.RandomState) -> np.ndarray:
        max_pos = self.frame_size - self.digit_size
        x = rng.uniform(0, max_pos)
        y = rng.uniform(0, max_pos)
        return np.array([x, y])
    
    def _get_random_velocity(self, rng: np.random.RandomState) -> np.ndarray:
        min_v, max_v = self.velocity_range
        magnitude = rng.uniform(min_v, max_v)
        angle = rng.uniform(0, 2 * np.pi)
        vx = magnitude * np.cos(angle)
        vy = magnitude * np.sin(angle)
        return np.array([vx, vy])
    
    def _bounce(self, position: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        max_pos = self.frame_size - self.digit_size
        
        new_position = position + velocity
        new_velocity = velocity.copy()
        
        if new_position[0] < 0:
            new_position[0] = -new_position[0]
            new_velocity[0] = -new_velocity[0]
        elif new_position[0] > max_pos:
            new_position[0] = 2 * max_pos - new_position[0]
            new_velocity[0] = -new_velocity[0]
        
        if new_position[1] < 0:
            new_position[1] = -new_position[1]
            new_velocity[1] = -new_velocity[1]
        elif new_position[1] > max_pos:
            new_position[1] = 2 * max_pos - new_position[1]
            new_velocity[1] = -new_velocity[1]
        
        new_position = np.clip(new_position, 0, max_pos)
        
        return new_position, new_velocity
    
    def _render_frame(self, digit_images: List[np.ndarray], 
                      positions: List[np.ndarray]) -> np.ndarray:
        frame = np.zeros((self.frame_size, self.frame_size), dtype=np.float32)
        
        for digit_img, pos in zip(digit_images, positions):
            x, y = int(pos[0]), int(pos[1])
            
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(self.frame_size, x + self.digit_size)
            y_end = min(self.frame_size, y + self.digit_size)
            
            dx_start = x_start - x
            dy_start = y_start - y
            dx_end = dx_start + (x_end - x_start)
            dy_end = dy_start + (y_end - y_start)
            
            digit_region = digit_img[dy_start:dy_end, dx_start:dx_end]
            frame[y_start:y_end, x_start:x_end] = np.maximum(
                frame[y_start:y_end, x_start:x_end],
                digit_region
            )
        
        return frame
    
    def generate_video(
        self, 
        rng: np.random.RandomState,
        digit_labels: Optional[List[int]] = None,
        initial_positions: Optional[List[np.ndarray]] = None,
        initial_velocities: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, dict]:
        
        digit_images = []
        labels = []
        for i in range(self.num_digits):
            label = digit_labels[i] if digit_labels else None
            img, lbl = self._get_random_digit(rng, label)
            digit_images.append(img)
            labels.append(lbl)
        
        if initial_positions is not None:
            positions = [pos.copy() for pos in initial_positions]
        else:
            positions = [self._get_random_position(rng) for _ in range(self.num_digits)]
            
        if initial_velocities is not None:
            velocities = [vel.copy() for vel in initial_velocities]
        else:
            velocities = [self._get_random_velocity(rng) for _ in range(self.num_digits)]
        
        frames = []
        position_history = []
        velocity_history = []
        
        for t in range(self.sequence_length):
            frame = self._render_frame(digit_images, positions)
            frames.append(frame)
            
            position_history.append([pos.copy() for pos in positions])
            velocity_history.append([vel.copy() for vel in velocities])
            
            for i in range(self.num_digits):
                positions[i], velocities[i] = self._bounce(positions[i], velocities[i])
        
        video = np.stack(frames, axis=0)
        video = np.expand_dims(video, axis=1)
        
        metadata = {
            'labels': labels,
            'initial_positions': [pos.copy() for pos in position_history[0]],
            'initial_velocities': [vel.copy() for vel in velocity_history[0]],
            'position_history': position_history,
            'velocity_history': velocity_history
        }
        
        return video, metadata
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        if self.deterministic:
            rng = np.random.RandomState(self.seed + idx if self.seed else idx)
        else:
            rng = self.rng
        
        video, metadata = self.generate_video(rng)
        
        video = torch.from_numpy(video).float()
        
        if self.transform is not None:
            video = self.transform(video)
        
        return video, metadata
    
    def generate_with_control(
        self,
        digit_labels: List[int],
        motion_type: str = 'random',
        velocity_magnitude: float = 3.0
    ) -> Tuple[torch.Tensor, dict]:
        rng = np.random.RandomState()
        
        velocities = []
        for i in range(len(digit_labels)):
            if motion_type == 'horizontal':
                vx = velocity_magnitude * (1 if rng.random() > 0.5 else -1)
                vy = 0
            elif motion_type == 'vertical':
                vx = 0
                vy = velocity_magnitude * (1 if rng.random() > 0.5 else -1)
            elif motion_type == 'diagonal':
                vx = velocity_magnitude * 0.707 * (1 if rng.random() > 0.5 else -1)
                vy = velocity_magnitude * 0.707 * (1 if rng.random() > 0.5 else -1)
            else:
                angle = rng.uniform(0, 2 * np.pi)
                vx = velocity_magnitude * np.cos(angle)
                vy = velocity_magnitude * np.sin(angle)
            velocities.append(np.array([vx, vy]))
        
        video, metadata = self.generate_video(
            rng,
            digit_labels=digit_labels,
            initial_velocities=velocities
        )
        
        metadata['motion_type'] = motion_type
        video = torch.from_numpy(video).float()
        
        return video, metadata


class MovingMNISTWithLabels(MovingMNIST):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_directions = 8
        self.num_speeds = 3
        
    def _get_motion_label(self, velocity: np.ndarray) -> dict:
        vx, vy = velocity
        magnitude = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vy, vx)
        
        direction = int((angle + np.pi) / (2 * np.pi) * self.num_directions) % self.num_directions
        
        min_v, max_v = self.velocity_range
        speed_range = max_v - min_v
        if magnitude < min_v + speed_range / 3:
            speed = 0
        elif magnitude < min_v + 2 * speed_range / 3:
            speed = 1
        else:
            speed = 2
            
        return {'direction': direction, 'speed': speed, 'angle': angle, 'magnitude': magnitude}
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        video, metadata = super().__getitem__(idx)
        
        motion_labels = []
        for vel in metadata['initial_velocities']:
            motion_labels.append(self._get_motion_label(vel))
        metadata['motion_labels'] = motion_labels
        metadata['content_labels'] = metadata['labels']
        
        return video, metadata


def _collate_fn(batch):
    videos = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return videos, metadata


def get_moving_mnist_dataloaders(
    root: str = './data',
    batch_size: int = 32,
    train_size: int = 10000,
    val_size: int = 1000,
    test_size: int = 1000,
    sequence_length: int = 20,
    frame_size: int = 64,
    num_digits: int = 2,
    num_workers: int = 4,
    seed: Optional[int] = 42,
    deterministic: bool = True,
    with_labels: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    DatasetClass = MovingMNISTWithLabels if with_labels else MovingMNIST
    
    def _make_seed(offset: int) -> Optional[int]:
        return None if seed is None else seed + offset
    
    train_dataset = DatasetClass(
        root=root,
        train=True,
        num_videos=train_size,
        sequence_length=sequence_length,
        frame_size=frame_size,
        num_digits=num_digits,
        seed=_make_seed(0),
        deterministic=deterministic
    )
    
    val_dataset = DatasetClass(
        root=root,
        train=True,
        num_videos=val_size,
        sequence_length=sequence_length,
        frame_size=frame_size,
        num_digits=num_digits,
        seed=_make_seed(1000),
        deterministic=deterministic
    )
    
    test_dataset = DatasetClass(
        root=root,
        train=False,
        num_videos=test_size,
        sequence_length=sequence_length,
        frame_size=frame_size,
        num_digits=num_digits,
        seed=_make_seed(2000),
        deterministic=deterministic
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    print("Testing Moving MNIST Dataset")
    
    dataset = MovingMNIST(
        num_videos=100,
        sequence_length=20,
        frame_size=64,
        num_digits=2,
        deterministic=True,
        seed=42
    )
    
    video, metadata = dataset[0]
    print(f"Video shape: {video.shape}")  # [20, 1, 64, 64]
    print(f"Video dtype: {video.dtype}")
    print(f"Video range: [{video.min():.3f}, {video.max():.3f}]")
    print(f"Metadata: {metadata['labels']}")
    
    train_loader, val_loader, test_loader = get_moving_mnist_dataloaders(
        batch_size=8,
        train_size=100,
        val_size=10,
        test_size=10,
        num_workers=0
    )
    
    batch, metadata = next(iter(train_loader))
    print(f"\nBatch shape: {batch.shape}")  # [8, 20, 1, 64, 64]
    print("Dataset test passed!")


