# Few Shot Text Classification

Few shot learning is common in Computer Vision, and I wondered how effective it would be to take a common CV algorithm and apply it to text classification.

In this repo I've converted Prototypical Networks for use in NLP.

### Running

- `git clone https://github.com/E-Renshaw/few-shot-text.git`
- change the sentence variables to be your sentences
- `python3 main.py`

### Results

We evaluate using the [Multi-Domain Sentiment Dataset](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/).

With 5 examples the algorithm gets 75.11% accuracy.
