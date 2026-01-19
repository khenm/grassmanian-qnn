
import os
import tempfile

# Placeholder for verification logic
def verify_pipeline():
    print("Starting verification...")
    
    # Create temp dirs
    with tempfile.TemporaryDirectory() as temp_tar_dir, tempfile.TemporaryDirectory() as temp_root_dir:
        print(f"Temp Tar Dir: {temp_tar_dir}")
        print(f"Temp Root Dir: {temp_root_dir}")
        
        # Create dummy tar file
        import tarfile
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        
        # Create dummy TIFF
        dummy_tif_path = "SPECTRAL_IMAGE.TIF"
        dummy_ql_path = "QL_QUALITY_CLASSES.TIF"
        
        # Create dummy spectral image
        with rasterio.open(
            dummy_tif_path,
            'w',
            driver='GTiff',
            height=32,
            width=32,
            count=224, # HyspecNet bands
            dtype=rasterio.uint16
        ) as dst:
            dst.write(np.random.randint(0, 1000, (224, 32, 32), dtype=np.uint16))
            
        # Create dummy label image
        with rasterio.open(
            dummy_ql_path,
            'w',
            driver='GTiff',
            height=32,
            width=32,
            count=1,
            dtype=rasterio.uint8
        ) as dst:
            dst.write(np.random.randint(0, 5, (1, 32, 32), dtype=np.uint8))
            
        # Tar them up
        tar_path = os.path.join(temp_tar_dir, "sample.tar")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(dummy_tif_path)
            tar.add(dummy_ql_path)
            
        # Cleanup local tifs
        os.remove(dummy_tif_path)
        os.remove(dummy_ql_path)
        
        # Import pipeline function
        from src.data.preprocessing import prepare_hyspecnet_data
        
        # Run preparation
        try:
            prepare_hyspecnet_data(temp_tar_dir, temp_root_dir, num_workers=2)
            print("Preparation executed successfully.")
            
            # Verify output
            expected_data = os.path.join(temp_root_dir, "DATA.npy")
            expected_label = os.path.join(temp_root_dir, "QL_QUALITY_CLASSES.TIF")
            
            if os.path.exists(expected_data):
                print(f"[PASS] DATA.npy generated at {expected_data}")
            else:
                print(f"[FAIL] DATA.npy misting at {expected_data}")
                
            if os.path.exists(expected_label):
                 print(f"[PASS] QL_QUALITY_CLASSES.TIF extracted at {expected_label}")
            else:
                 print(f"[FAIL] QL_QUALITY_CLASSES.TIF missing at {expected_label}")
                 
        except Exception as e:
            print(f"[FAIL] Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()

    print("Verification complete.")

if __name__ == "__main__":
    verify_pipeline()
