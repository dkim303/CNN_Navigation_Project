# CNN_Navigation_Project

Description:

    An image geolocation model that estimates a drone’s coordinates by mapping aerial photographs to georeferenced satellite images.

    The project divides large satellite maps into overlapping tiles and calculates the coordinate boundaries of each tile. A dual encoder CNN then processes the drone images and satellite tiles through separate CNN branches which will produce embeddings in a shared vector space. During training, the 2 CNN models are trained to product simlar embeddings between the correct pairs of drone images and sattelite tiles.

    In order to preserve local details while lowering computational costs, each drone image is divided into 4 quadrant. The quadrant embeddings are combined into a single representation for similarity based retrieval. In deployment, satellite tile embeddings are intended to be computed and stored in memroy in advance, which allows the new drone image embeddings to be localized through nearest neighbor search.
    
    Planned project features include:
        Geospatial satellite tiling and coordinate bound calculation
        Drone-to-satellite tile label generation using GPS metadata
        Separate PyTorch CNN encoders for drone and satellite imagery
        Contrastive similarity training and top-k tile retrieval
        Configurable model architecture, preprocessing, tiling, and training parameters
        Training diagnostics, evaluation metrics, and model export
        A planned C++ inference pipeline for deployment

Docker Setup:

    1.
    2.
    3.
    4.

Activate Virtual Environment:

    source .env/bin/activate

Instructions:

    Create and train new models, then export to C++:

        1: python -m training_program.train\ --model-name <???> --config configs/<???.yaml>
        2: python -m training_program.evaluate\ --model-name<???>

        3: make -C cpp_program
        4: ./cpp_program/program models/<???> models/<???>

    Compare different models:
        1: python -m training_program.compare\ --model-name-1 <???> --model-name-2 <???>