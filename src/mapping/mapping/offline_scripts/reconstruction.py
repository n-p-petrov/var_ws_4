from pathlib import Path

import pycolmap

output_path = Path("./colmap_output1/")
image_dir = Path("./rosbag_images_cleaned/")

output_path.mkdir()
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

extraction_options = pycolmap.FeatureExtractionOptions()
extraction_options.sift.max_num_features = 4000
pycolmap.extract_features(
    str(database_path), str(image_dir), extraction_options=extraction_options
)
pycolmap.match_sequential(str(database_path))
maps = pycolmap.incremental_mapping(
    str(database_path), str(image_dir), str(output_path)
)
maps[0].write(str(output_path))
# dense reconstruction
# pycolmap.undistort_images(str(mvs_path), str(output_path), str(image_dir))
# pycolmap.patch_match_stereo(str(mvs_path))  # requires compilation with CUDA
# pycolmap.stereo_fusion(str(mvs_path / "dense.ply"), str(mvs_path))
