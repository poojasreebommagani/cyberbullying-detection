import os
import pandas as pd

# Gets the directory where the script is running
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'cyberbullying_tweets.csv')
df = pd.read_csv(file_path)