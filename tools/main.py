from fire import Fire
from train import main as train_main
from test import main as test_main

if __name__ == "__main__":
    Fire(dict(
        train=train_main,
        test=test_main,
    ))
