# Named Entity Recognition (NER) Training Code

This repository contains code for training a Named Entity Recognition (NER) model using DistilBERT and PyTorch. The model is trained to extract specific entities from text data, such as banking alerts.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Transformers library (Hugging Face)
- Other dependencies specified in requirements.txt

### Installation

1. Clone the repository:

```
git clone git@github.com:shekolla/banking-alerts-ner.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

### Usage

1. Prepare the training data:
   - Modify the `texts` and `tags` lists in `train.py` to match your training data.
   - Ensure that the tags follow the B-I-O tagging scheme.

2. Run the training script:

```
python train.py
```

3. Evaluate and fine-tune the model as needed:
   - Modify the hyperparameters in `TrainingArguments` to adjust the training configuration.
   - You can explore other training techniques like data augmentation, active learning, or transfer learning to improve performance.

4. Save the trained model:
   - Use the provided code snippet to save the model after training.
   - Modify the path to save the model in your desired location.

5. Inference using the trained model:
   - Load the saved model using the provided code snippet in `inference.py`.
   - Modify the code in `inference.py` to fit your specific use case.
   - Run `inference.py` to extract entities from new text data.

### Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. You can also fork this repository and adapt the code and labels to your specific use case.

### Fork and Improve Labels

If you want to improve the labels or adapt the code for your own use case, feel free to fork this repository. You can modify the training data, add new entity types, or enhance the model's performance based on your specific requirements. We encourage you to contribute back to the community by sharing your improvements and modifications through pull requests.

### Special Thanks

- Special thanks to OpenAI for providing the ChatGPT model which powers the AI assistance in this conversation. I'm loving it, and I highly recommend using it for your AI projects!

### License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

### Acknowledgments

- The code in this repository is based on examples from the Hugging Face Transformers library and the PyTorch documentation.