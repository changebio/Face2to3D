eos: A lightweight header-only 3D Morphable Model fitting library in modern C++11/14
=========

Files in this directory:

- ibug_to_sfm.txt:
	Mappings from the popular ibug 68-point 2D facial landmarks markup to
	Surrey Face Model indices.

- sfm_shape_3448.bin:
	The public shape-only Surrey 3D Morphable Face Model.
	To obtain a full 3DMM and higher resolution levels, follow the instructions
	at cvssp.org/facemodel.
	Details about the different models can be found in:
	"A Multiresolution 3D Morphable Face Model and Fitting Framework",
	P. Huber, G. Hu, R. Tena, P. Mortazavian, W. Koppen, W. Christmas, M. Rätsch, J. Kittler,
	VISAPP 2016, Rome, Italy.

- expression_blendshapes_3448.bin:
	6 expression blendshapes for the sfm_shape_3448 model. Contains the expressions anger,
	disgust, fear, happiness, sadness and surprise.

- sfm_3448_edge_topology.json:
	Contains a precomputed list of the model's edges, and the two faces and vertices that are
	adjacent to each edge. Uses 1-based indexing ("0" has a special meaning of "no adjacent
	vertex/edge") - this may change to 0-based in the future to be consistent with the rest of
	the library. The file is used in the edge-fitting.

- sfm_model_contours.json:
	Definition of the SFM's contour vertices of the right and left side of the face.

- sfm_reference.obj:
	The reference 3D shape used to built the Surrey Face Model. We make it available so
	that new user-defined landmark points can be marked in this lowest-resolution
	model, if the points exist here.
	
- sfm_reference_annotated.obj:
	Visualisation of the landmark points defined in the ibug_to_sfm.txt mapping file.
	* Red: Annotated ibug points that are defined on the reference shape.
	* Green: Contour vertices from the file model_contours.json.
	The file ibug_to_sfm.txt contains a few more mappings of landmarks that are not present
	in the reference, for example the middle-inner eyebrow points - they are not visualised.

- sfm_reference_symmetry.txt:
	Contains a list of vertex symmetries of the reference shape, i.e. each
	vertex's symmetric counterpart. See the top of the file for more information.
