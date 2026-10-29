# This file is for mask generating
# The masks are lines generated through two directions
# The mask for training is random
# The mask for constructing the 3D rock should have some laws

import torch

INPUT_SIZE = 128   # masks are generated at this size and resized to the judge window (train_diffusion.resize_masks)


def training_mask(batch_size=16, bias = 0.08):
    # bias is the parameter controlling the region oof masks
    img_size = INPUT_SIZE
    masks = torch.ones(batch_size, 1, img_size, img_size)

    for i in range(batch_size):
    # sample on x and y directions
        point_x = random_sampling(bias)
        point_y = random_sampling(bias)

        # generate the mask
        masks[i, :, point_x, :] = 0
        masks[i, :, :, point_y] = 0

    return masks

def random_sampling(bias):
    #  higher probability near 0
    # Controls the bias (higher = stronger bias towards 0)
    num_samples = int(torch.distributions.Exponential(bias).sample())
    # num_samples = torch.randint(0, INPUT_SIZE, (1,)).item()
    # print(num_samples)

    points = torch.randperm(INPUT_SIZE)[0:num_samples]
    return points

def gen_masks():
    img_size = INPUT_SIZE
    masks = []

    fake_masks = torch.full((1, 1, img_size, img_size, img_size), float('nan'))

    points = hierarchical_binary_layers(0, INPUT_SIZE)
    for _, idx in enumerate(points):
        masks.append(torch.isnan(fake_masks[0, 0, idx, :, :]))
        fake_masks[0, 0, idx, :, :] = 1
        masks.append(torch.isnan(fake_masks[0, 0, :, idx, :]))
        fake_masks[0, 0, :, idx, :] = 1
        masks.append(torch.isnan(fake_masks[0, 0, :, :, idx]))
        fake_masks[0, 0, :, :, idx] = 1

    masks = torch.stack(masks).to(torch.float32)
    masks = masks.unsqueeze(1)
    return masks


def hierarchical_binary_layers(start, end):
    """Traverse [start, end) using a hierarchical binary order."""
    queue = [(start, end)]  # BFS queue storing (start, end) pairs
    result = []

    while queue:
        new_queue = []
        for s, e in queue:
            if s >= e:
                continue
            mid = (s + e) // 2
            result.append(mid)
            new_queue.append((s, mid))  # Left half
            new_queue.append((mid + 1, e))  # Right half

        queue = new_queue  # Move to next layer

    return result


if __name__ == '__main__':
    mask = gen_masks()

    # Define grid size
    num_images = mask.shape[0]

    # Create grid of images
    grid_images = vutils.make_grid(mask, nrow=8, normalize=True)

    # Plot the grid
    plt.figure(figsize=(12, 16))
    plt.imshow(grid_images.permute(1, 2, 0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()
