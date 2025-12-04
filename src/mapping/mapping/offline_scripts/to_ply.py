import pycolmap

reconstruction = pycolmap.Reconstruction("./colmap_output/")
reconstruction.export_PLY("./colmap_output/out.ply")
