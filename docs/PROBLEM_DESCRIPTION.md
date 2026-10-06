# Case Study I: Telecom Service Provisioning & Billing Prediction

Have you ever been charged for something you did not subscribe to? How hard could it be to get my invoice right?

One of the most important processes for telecommunications operators is service provisioning to a customer. This is the process responsible for providing you with the services you contracted and charging you for them.

A simple 4 Play subscription—Internet, TV, mobile phone, and fixed line—involves configuring multiple systems:

* **CRM (Customer Relationship Management):** The master system that keeps information about the offers that the customer subscribed to.
* **TV platform:** Defines the channels that the customer can watch.
* **Internet platform:** Defines the internet speed.
* **Billing system:** Responsible for billing the services and the usage that the customer makes of them.

It is a complex process as there are hundreds of options, thousands of possible combinations, and millions of customers. It is subject to errors, both human and automatic. For this reason, operators have different "audit" processes that try to ensure that all intervening systems are consistent with each other.

In this competition, we challenge you to predict the right bill for a customer given the services they contracted. The goal is to build a model that receives the configuration of the CRM system and predicts the correct configuration of the Billing system.

As mentioned, CRM is normally the master of the information of the services that the customer has subscribed to, so it will be possible to infer which configuration to expect in the Billing system. Note, however, that the configurations between these two systems may not be one-to-one; in many cases, there are many-to-one configurations, meaning two different CRM configurations may point to the same Billing configuration.

## Project Guidelines & Structure

* The project will be carried out in teams of 5 or 6 students.
* Each team must appoint a team leader.
* The team leader will be responsible for defining the project activities and assigning team members to each task within Microsoft Planner.
* The project has a duration of 3 weeks, structured into three distinct sprints:
  * **Sprint 1:** Pre-processing
  * **Sprint 2:** Modeling
  * **Sprint 3:** Optimization and Explainability
* The final grade for the project will be calculated based on the following weights for each sprint:
  * **Sprint 1:** 40% — Submission date: 22/09/2026 23:59
  * **Sprint 2:** 30% — Submission date: 29/09/2026 23:59
  * **Sprint 3:** 30% — Submission date: 06/10/2026 23:59
* At the end of each week, all teams are required to present the activities they have completed and the results obtained during that sprint.
* The notebook corresponding to each sprint must be submitted on Moodle.

## Characteristics of the Problem

* The dataset contains several provisioning errors, i.e., in the dataset it may happen that for a CRM configuration there are different configurations in Billing.
* The dimension of the output ($731$) is very large.
* Different CRM configurations can correspond to only one Billing configuration.
* The individual variables of the CRM configuration may not have a direct correspondence to the individual variables of the Billing configuration.

## Evaluation

Submissions are evaluated according to the **Exact Matching Ratio (EMR)**: the percentage of samples that have all their labels classified correctly.

$$
\text{EMR} = \frac{1}{n} \sum_{i=1}^{n} I(Y_i = Z_i)
$$

Where $I$ is the indicator function, $n$ is the number of samples, $k$ is the size of the labelset, $Y_i \in \{0,1\}^k$ represents the true label, and $Z_i \in \{0,1\}^k$ represents the prediction.

Only samples where all labels are correct will be considered correct. In order to have a perfect score of $\text{EMR} = 1$, all the labels for all samples need to be correct.

## Dataset Description

The configurations in both systems have been pre-processed in order to obtain only binary variables. This way, each column corresponds to an option in the configuration of the respective system (CRM / BILLING), where the value `0` indicates that the option is inactive and the value `1` indicates that it is active.

The training dataset contains one line for each client, and given the set of CRM system settings, the objective is to return the Billing system configuration. The training dataset contains errors in the configuration of CRM/BILLING, however when testing, only pairs of CRM-BILLING configurations that are deemed correct will be used.

### File Descriptions

* `train.csv`: The training set.
* `test.csv`: The test set.
* `sampleSubmission.csv`: A sample submission file in the correct format.

### Data Fields

The training dataset is structured as follows: 1 column to identify the customer, 745 columns for the CRM configuration, and 731 columns for the BILLING configuration.

* `MSISDN`: Customer's identifier (encrypted field).
* **CRM Columns:** Columns starting with `CRM` (2nd – 746th inclusive) correspond to the input configuration (CRM).
* **Billing Columns:** Columns starting with `BIL` (747th – 1477th inclusive) correspond to the output configuration (BILLING).

The test dataset is structured similarly, containing 1 column to identify the customer and 745 columns for the CRM configuration.