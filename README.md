# CogniSort – Machine Learning-based Content Recommendation System

CogniSort is an advanced content recommendation system that leverages Python, Machine Learning, [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), Natural Language Processing (NLP), and the [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark). 

This system provides intelligent content recommendations for various applications, optimizing both **_space performance and time complexity_**. :partying_face:

## Features

- **Machine Learning Powered:** CogniSort employs state-of-the-art machine learning techniques to deliver personalized content recommendations.

- **Optimized Performance:** By selecting the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4) over [BERT](https://sease.io/2021/12/using-bert-to-improve-search-relevance.html), we have significantly improved the efficiency of the recommendation system, reducing time complexity from **_O(P ** 2) to O(P)_**, where P represents the number of available items.

- **Customizable:** CogniSort offers flexibility for easy integration into different platforms and applications.

## Getting Started

Follow these steps to get CogniSort up and running:

1. **Installation:** Start by cloning this repository to your local machine.

   ```shell
   git clone https://github.com/your-username/CogniSort.git
   ```

2. **Dependencies:** Make sure to install the required Python packages. You can do this via pip:

   ```shell
   pip install -r requirements.txt
   ```

3. **Data Preparation:** Prepare your data according to the specifications in the documentation. 

4. **Training:** Train the recommendation model on your data. You can refer to the documentation for guidance on this process.

5. **Integration:** Integrate CogniSort's recommendations into your application, platform, or website.

## Usage

CogniSort provides an easy-to-use API for incorporating content recommendations into your project. Refer to the documentation for comprehensive usage instructions.

## Documentation

For detailed information, please refer to the [documentation](https://arxiv.org/pdf/1803.11175.pdf).


We would like to express our gratitude to the contributors, the open-source community, and the authors of the STS Benchmark for their invaluable work.

