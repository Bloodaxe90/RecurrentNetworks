import os

from src.engine.test import test
from src.layers.gru_layer import GRULayer
from src.layers.lstm_layer import LSTMLayer
from src.layers.recurrent_layer import RecurrentLayer
from src.models.general_rnn import GeneralRNN
from src.utils.create_mnist_dataloaders import create_mnist_dataloaders
from src.utils.other import set_seed


def main():
    print("sup")
    import torch.optim
    from torch import nn

    from src.engine.train import train

    def main():
        DATASET = "Fashion_MNIST"
        BATCH_SIZE = 64
        SHUFFLE = True
        CHUNK_SIZE = 14
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        SEED = 32
        set_seed(SEED)
        WORKERS = (
            min(max(1, BATCH_SIZE // 16), torch.cuda.device_count())
            if DEVICE == "cuda"
            else torch.cpu.device_count()
        )

        mnist_dir_path = f"{os.path.dirname(os.getcwd())}/resources/{DATASET}"

        train_dataloader, test_dataloader = create_mnist_dataloaders(
            mnist_dir_path=mnist_dir_path,
            batch_size=BATCH_SIZE,
            workers=WORKERS,
            shuffle=SHUFFLE,
            chunk_size=CHUNK_SIZE,
        )

        dataset = train_dataloader.dataset

        LAYER_TYPE = GRULayer
        NUM_RECURRENT_LAYER = 2
        INPUT_DIM = CHUNK_SIZE
        HIDDEN_DIM = 64
        OUTPUT_DIM = len(dataset.classes)
        DROPOUT_PROB = 0.2

        model = GeneralRNN(
            recurrent_layer=LAYER_TYPE,
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            num_recurrent_layers=NUM_RECURRENT_LAYER,
            dropout_prob=DROPOUT_PROB,
        )

        if DEVICE == "cuda":
            model = nn.DataParallel(model, device_ids=list(range(WORKERS)))
        model.to(DEVICE)

        print(f"Device: {DEVICE}")
        print(f"Workers: {WORKERS}")

        ALPHA = 1e-4
        EPOCHS = 2
        EXP_NAME = (
            f"frfr_"
            f"{DATASET}_"
            f"{LAYER_TYPE.__name__}_"
            f"NL{NUM_RECURRENT_LAYER}_"
            f"LR{ALPHA}_"
            f"E{EPOCHS}_"
            f"B{BATCH_SIZE}_"
            f"ID{INPUT_DIM}_"
            f"HD{HIDDEN_DIM}_"
            f"OD{OUTPUT_DIM}_"
            f"SD{SEED}_"
            f"S{SHUFFLE}_"
            f"DP{DROPOUT_PROB}"
        )
        MODEL_NAME = f"{EXP_NAME}"
        TRAIN_EVAL_INTERVAL = 10  # Every x batches
        TRAIN_SAVE_INTERVAL = 100  # Every x batches

        train_results = train(
            model=model,
            dataloader=train_dataloader,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(params=model.parameters(), lr=ALPHA),
            epochs=EPOCHS,
            eval_interval=TRAIN_EVAL_INTERVAL,
            save_interval=TRAIN_SAVE_INTERVAL,
            device=DEVICE,
            exp_name=EXP_NAME,
            model_name=MODEL_NAME,
        )

        print(train_results)

        TEST_EVAL_INTERVAL = 1  # Every x batches
        TEST_SAVE_INTERVAL = 50  # Every x batches

        test_results = test(
            model=model,
            dataloader=test_dataloader,
            loss_fn=nn.CrossEntropyLoss(),
            eval_interval=TEST_EVAL_INTERVAL,
            save_interval=TEST_SAVE_INTERVAL,
            device=DEVICE,
            exp_name=f"TEST_{MODEL_NAME}",
        )

        print(test_results)

    if __name__ == "__main__":
        main()


if __name__ == "__main__":
    main()
