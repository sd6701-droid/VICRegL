
import torch
from PIL import Image
import numpy as np
from transforms import MultiCropTrainDataTransform

def test_transform_grid_size():
    # Test case 1: Default configuration
    print("Testing default configuration: size_crops=[224, 96], num_crops=[2, 6]")
    transform = MultiCropTrainDataTransform(
        size_crops=[224, 96],
        num_crops=[2, 6],
        return_location_masks=True
    )
    
    img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    crops, locations = transform(img)
    
    # Check first 2 crops (224x224) -> expect grid size 7 (224//32)
    for i in range(2):
        loc = locations[i]
        N = loc.shape[0]
        print(f"Crop {i} (size 224): Grid size N={N}")
        assert N == 7, f"Expected N=7 for crop size 224, got {N}"
        
    # Check next 6 crops (96x96) -> expect grid size 3 (96//32)
    for i in range(2, 8):
        loc = locations[i]
        N = loc.shape[0]
        print(f"Crop {i} (size 96): Grid size N={N}")
        assert N == 3, f"Expected N=3 for crop size 96, got {N}"

    print("Default configuration test passed!")
    
    # Test case 2: Custom configuration (e.g. all 96x96)
    print("\nTesting custom configuration: size_crops=[96, 96], num_crops=[2, 6]")
    transform = MultiCropTrainDataTransform(
        size_crops=[96, 96],
        num_crops=[2, 6],
        return_location_masks=True
    )
    
    crops, locations = transform(img)
    
    # All crops should have grid size 3
    for i in range(8):
        loc = locations[i]
        N = loc.shape[0]
        print(f"Crop {i} (size 96): Grid size N={N}")
        assert N == 3, f"Expected N=3 for crop size 96, got {N}"
        
    print("Custom configuration test passed!")

    # Test case 3: Mixed configuration with non-standard sizes
    print("\nTesting mixed configuration: size_crops=[160, 64], num_crops=[1, 1]")
    transform = MultiCropTrainDataTransform(
        size_crops=[160, 64],
        num_crops=[1, 1],
        return_location_masks=True
    )
    
    crops, locations = transform(img)
    
    # Crop 0: 160 // 32 = 5
    loc = locations[0]
    N = loc.shape[0]
    print(f"Crop 0 (size 160): Grid size N={N}")
    assert N == 5, f"Expected N=5 for crop size 160, got {N}"
    
    # Crop 1: 64 // 32 = 2
    loc = locations[1]
    N = loc.shape[0]
    print(f"Crop 1 (size 64): Grid size N={N}")
    assert N == 2, f"Expected N=2 for crop size 64, got {N}"
    
    print("Mixed configuration test passed!")

if __name__ == "__main__":
    test_transform_grid_size()
