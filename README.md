# American Express: Attentive Recommendation

This repository contains the source code and supporting datasets for a team project developed during the Break Through Tech AI program. The project's objective is to build a recommendation engine utilizing a two-tower model with attention mechanisms to enhance recommendation relevance across various applications.

## Project Overview

The primary goal of this project is to develop a recommendation system that leverages a two-tower architecture integrated with attention mechanisms. This approach aims to improve the relevance and accuracy of recommendations by effectively capturing user-item interactions.

## Table of Contents

- [Datasets](#datasets)
- [Data Cleaning and Exploratory Data Analysis (EDA)](#data-cleaning-and-exploratory-data-analysis-eda)
- [Model Development](#model-development)
- [Results and Evaluation](#results-and-evaluation)
- [Contributors](#contributors)
- [License](#license)

## Datasets

The project utilizes publicly available datasets to train and evaluate the recommendation engine. These datasets include user interaction data, item metadata, and contextual information relevant to the recommendation task.

## Data Cleaning and Exploratory Data Analysis (EDA)

Data preprocessing steps involve cleaning the datasets to handle missing values, outliers, and inconsistencies. Exploratory Data Analysis is conducted to understand data distributions, identify patterns, and inform feature engineering decisions.

## Model Development

The recommendation engine is built using a two-tower model architecture:

- **User Tower**: Processes user-related features to generate user embeddings.
- **Item Tower**: Processes item-related features to generate item embeddings.

An attention mechanism is incorporated to weigh the importance of different features, enhancing the model's ability to capture complex user-item interactions.

## Results and Evaluation

The model's performance is evaluated using metrics such as precision, recall, and mean reciprocal rank (MRR). Comparative analyses with baseline models are conducted to demonstrate the effectiveness of the proposed approach.

## Contributors
This project is a collaborative effort by the following team members:
<!-- readme: contributors -start -->
<table>
	<tbody>
		<tr>
	    <td align="center">
                <a href="https://github.com/ardahk">
                    <img src="https://avatars.githubusercontent.com/u/73215056?v=4" width="100;" alt="ardahk"/>
                    <br />
                    <sub><b>Arda Hoke</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/asmik12">
                    <img src="https://avatars.githubusercontent.com/u/168493757?v=4" width="100;" alt="asmik12"/>
                    <br />
                    <sub><b>Asmi K</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/paigecaskey">
                    <img src="https://avatars.githubusercontent.com/u/120995805?v=4" width="100;" alt="paigecaskey"/>
                    <br />
                    <sub><b>Paige Caskey</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Rebeccals">
                    <img src="https://avatars.githubusercontent.com/u/2145912?v=4" width="100;" alt="Rebeccals"/>
                    <br />
                    <sub><b>Rebecca Smith</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/tanishaad">
                    <img src="https://avatars.githubusercontent.com/u/100120733?v=4" width="100;" alt="tanishaad"/>
                    <br />
                    <sub><b>tanisha dutta</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/vaishnavi-rama">
                    <img src="https://avatars.githubusercontent.com/u/156384098?v=4" width="100;" alt="vaishnavi-rama"/>
                    <br />
                    <sub><b>vaishnavi-rama</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors -end -->

## License

This project is licensed under the [MIT License](LICENSE).

For detailed information on the implementation and usage, please refer to the individual module documentation within the repository. 

