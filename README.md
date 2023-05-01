<!-- PROJECT LOGO -->
<a name="readme-top"></a>
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://user-images.githubusercontent.com/47918966/235481977-d0f1269c-a353-4153-935d-9aa24af71538.png" alt="Logo" width="450" height="450">
  </a>
  
<h3 align="center">Graph4StackOverflow</h3>

  <p align="center">
    Graph-based Text Classification of Stack Overflow Expertise
    <br />
    <a href="https://www.linkedin.com/in/liam-h-byrne/"><strong>Liam Byrne</strong></a>
    <br />
    <br />
    <a href="https://github.com/liamhbyrne/Graph4StackOverflow/issues">Report Bug</a>
    ·
    <a href="https://github.com/liamhbyrne/Graph4StackOverflow/issues">Request Feature</a>
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

Stack Overflow is a widely-used platform for programmers to ask and find answers to their questions. This project presents two Graph Neural Network (GNN) models, one for identifying expert users who can provide high-quality answers to questions and another for ranking the answers based on the expertise of each user. This project creates the “user graph”; which is a novel graph construction method for representing a user’s contributions to the platform that captures their technical expertise. Heterogeneous GAT and GraphSAGE models are developed to operate directly on the graph structure to predict whether a user will provide a high-quality answer. Experimental results show that the GNN models outperform conventional deep learning methods for upvote classification and achieve an impressive weighted-F1 score of **0.781**. Moreover, the GraphSAGE model achieves a P@1 score of **0.395** for accepted answer prediction, which is competitive withthe state-of-the-art approaches. Analysis into the performance-overhead tradeoff was carried out and it has been concluded feasible for Stack Exchange to implement recommendation engines that utilize graph-based models. These outcomeshave the potential to make a significant contribution towards improving the quality of answers on Stack Overflow and hold the potential to further enhance the platform.


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* [Anaconda](https://www.anaconda.com/)
* [Java](https://www.java.com/en/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
   
2. Create [Anaconda](https://www.anaconda.com/) environment from YAML file.
   ```sh
   conda env create --file=env.yml
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Liam Byrne - [LinkedIn](https://www.linkedin.com/in/liam-h-byrne/) - lhb1g20@soton.ac.uk

Project Link: [https://github.com/liamhbyrne/Graph4StackOverflow/](https://github.com/liamhbyrne/Graph4StackOverflow/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
