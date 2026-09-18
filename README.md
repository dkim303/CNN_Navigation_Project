# CNN_Navigation_Project

source .env/bin/activate

Instructions: 
    python -m training_program.train \ --model-name <???> --config configs/<???.yaml>
    python -m training_program.evaluate \ --model-name<???>

    make -C cpp_program
    ./cpp_program/program models/<???> models/<???>