"""MediaPipe Face Mesh landmark index groups.

Replaces face_recognition/dlib: MediaPipe ships as a pip wheel (no cmake /
compiled dlib required), runs faster on CPU, and gives 468 3D landmarks per
face instead of dlib's 68.

Indices below are the canonical MediaPipe Face Mesh points, ordered to match
the 6-point convention expected by geometry.eye_aspect_ratio /
mouth_aspect_ratio: [corner, top, top, corner, bottom, bottom].
"""

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Outer mouth contour, 6 points spanning left corner -> right corner.
MOUTH = [61, 81, 311, 291, 178, 402]

LEFT_EYE_OUTER_CORNER = 33
RIGHT_EYE_OUTER_CORNER = 263
