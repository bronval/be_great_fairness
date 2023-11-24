
##################################################################################################
#
# use this file to reproduce the results from original paper (https://arxiv.org/abs/2210.06280)
#
##################################################################################################


from be_great import GReaT
from sklearn import datasets
import logging
from examples.utils import set_logging_level
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from torch import cuda
from sklearn.model_selection import train_test_split


logger = set_logging_level(logging.INFO)


def train_model(dataset_name, model_name, save_folder_checkpoint, from_csv=False, epochs=200):
    
    if from_csv:
        dataset = pd.read_csv(dataset_name)
    else:
        dataset = load_dataset(dataset_name, split="train").to_pandas()
        
    dataset, _ = train_test_split(dataset, test_size=0.2, random_state=42) # random_state added after exp on housing adult and heloc (before sick)

    great = GReaT("distilgpt2",
                epochs=epochs,
                save_steps=20000,
                logging_steps=1000,
                batch_size=16,
                experiment_dir=save_folder_checkpoint,
                learning_rate=5e-5,
                use_cpu=False,
                )

    trainer = great.fit(dataset)
    great.save(model_name)

    # loss_hist = trainer.state.log_history.copy()
    # loss = [x["loss"] for x in loss_hist]
    # epochs = [x["epochs"] for x in loss_hist]
    # plt.plot(epochs, loss)
    # plt.savefig("loss_adult.png", bbox_inches="tight")


def generate_sample(saved_folder, save_sample, n_generate=1000):
    great = GReaT.load_from_dir(saved_folder)
    samples = great.sample(n_generate, temperature=0.7, max_length=2000)
    samples.to_csv(save_sample, index=False)



if __name__ == "__main__":
    
    print(f"CUDA available: {cuda.is_available()}")
    
    # dataset_name = 'scikit-learn/adult-census-income'
    
    ## 200 epochs except for housing and diabetes
    # train_model("real_case_D.csv", "rule_D", "trainer_rule_D", from_csv=True, epochs=100)
    # train_model("heloc_renamed.csv", "heloc_renamed", "trainer_heloc_renamed", from_csv=True, epochs=200)
    train_model("adult_test.csv", "adult_test", "trainer_adult_test", from_csv=True, epochs=200)


    # generate_sample("housing_renamed", "housing_renamed_synthetic.csv", n_generate=20500)




