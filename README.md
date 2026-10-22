# CNN_Navigation_Project

Description:

    An image geolocation model that estimates a drone’s coordinates by mapping aerial 
    photographs to georeferenced satellite images.

    The project divides large satellite maps into overlapping tiles and calculates the coordinate 
    boundaries of each tile. A dual encoder CNN then processes the drone images and satellite tiles 
    through separate CNN branches which will produce embeddings in a shared vector space. 
    During training, the 2 CNN models are trained to product simlar embeddings between the correct 
    pairs of drone images and sattelite tiles.

    In order to preserve local details while lowering computational costs, each drone image is divided 
    into 4 quadrant. The quadrant embeddings are combined into a single representation for similarity based retrieval. 
    In deployment, satellite tile embeddings are intended to be computed and stored in memroy in advance, 
    which allows the new drone image embeddings to be localized through nearest neighbor search.
    
    Combined model contains seperate models for satellite and drone data. Given a trained model pair, 
    it can ingest satellite data and estimate locations given new drone images.

    Planned project features include:
        1. Geospatial satellite tiling and coordinate bound calculation
        2. Drone-to-satellite tile label generation using GPS metadata
        3. Separate PyTorch CNN encoders for drone and satellite imagery
        4. Contrastive similarity training and top-k tile retrieval
        5. Configurable model architecture, preprocessing, tiling, and training parameters
        6. Training diagnostics, evaluation metrics, and model export
        7. A planned C++ inference pipeline for deployment

Docker Setup:

    1.
    2.
    3.
    4.

Create and Activate Virtual Environment:
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Instructions:

    Create and train new models and save as .pt files. Then export as C++ executables.

        1: python -m training_program.train\ --model-name <???> --config configs/<???.yaml>
        2: python -m training_program.evaluate\ --model-name<???>

        3: make -C cpp_program
        4: ./cpp_program/program models/<???>

    Compare different models:
        1: python -m training_program.compare\ --model-name-1 <???> --model-name-2 <???>