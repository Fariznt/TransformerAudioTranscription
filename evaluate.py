"""
Main entry point for evaluating a trained MT3 checkpoint, saving results.
"""
import argparse

def main():
    """
    1. Load Model and load weights from checkpoint.
    2. Initialize test dataloader.
    4. Run evaluate_test_set from src/evaluation/evaluate.py
    5. Print/Log the results.
    """
    # configuration
    save = True # save the results
    name = "..." # name of the run to load the model from
    evaluation_dataset_path = "./datasets/..."
    # .. other relevant configuration that could be arguments to evaluate_test_set
    # ... goes here

    # load model from ./runs/<name>/model.pth

    # run

    # evaluate with evaluate_test_set from src/evaluation/evaluate.py

    # print results

    # save all results (images and metrics) to ./runs/<name>/... 


    pass

if __name__ == "__main__":
    main()
