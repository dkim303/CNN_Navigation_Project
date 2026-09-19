# CNN_Navigation_Project

Description:

    This program

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