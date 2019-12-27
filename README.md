# Stitch-Panorama
Generate a panoramic image by stitching together multiple images of a scene.

# Task
# Extracting Feature Points
  •You may use any descriptor of your choice available in OpenCV.
# Homography Estimation
  •Compute the pair-wise image transformation matrix by applying RANSAC on the setof extracted feature points.
  •Develop some notion for scoring mechanism to choose which transformation pairs toconsider for the panorama construction.
# Stitch and Blend images
  •With the obtained transformations, first, estimate the size of the overall picture thatwould be formed upon stitching.
  •For estimating the pixel values in regions where multiple images are superimposed,apply a blending technique of your choice.     
   For a start, checkout Alpha blending,Laplacian Pyramid Blending, 2-band blending, Graph cut blending
