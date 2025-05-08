# Credit-Risk-Management-Model

### [train-data](https://drive.google.com/drive/folders/1-pwcPyKBsz49qDfn42VHirtutonfr_Mt?usp=sharing)

## Introduction

Banks use credit risk management models to assess how trustworthy a client is in fulfilling obligations under loan agreements. When you, as a client, fill out an application for a loan or mortgage, you are evaluated using a credit risk management model. A bank may use various types of information—for example, your place of work, age, and history of previous repayments for other loans from banks and credit institutions. Based on this data, a machine learning model helps the credit manager decide whether the requested loan amount can be trusted to the applicant.

With such automation, banks save time for their specialists, so they don’t have to search for and aggregate information for each client to make a credit decision. This speeds up the approval process for loan applications. However, in certain cases, specialists may still review the model’s decision to audit it and identify potential weaknesses.

This example covers only one model used in credit risk management. In addition to it, banks also use models to predict the amount of credit or loan a client can afford to repay and to determine the current creditworthiness rating of a client who already has a loan. These models are needed to forecast which clients may fall into default and to take preventive actions before issuing the loan.

## The Problem to Be Solved

As part of the final project, it tackles a highly relevant task — assessing the risk of a client defaulting on a loan.

**Default** refers to the failure to pay interest on a loan or bond, or to repay a loan within a certain time period *t*. A default is typically considered to have occurred if the client has not made a loan payment within 90 days.

An effective model allows a bank or other credit organization to assess the current risk associated with any issued loans or credit products and, with higher probability, prevent the client from defaulting on their obligations. This way, the bank reduces its risk of incurring losses.

## Brief Description of the Task

It needs to create one of the models for credit risk assessment — predicting whether a client will default on a loan.

## Description of the Data

The data contains information about various attributes of borrowers and credit products: clients who already have loans, their credit history, and financial indicators. Each record in the dataset represents a single credit product issued to a specific borrower.

**Attributes are described in the file**: [description_eng.xlsx](description_eng.xlsx)

## Results

Successfully implemented a service that can accept new data and make predictions.

The application includes:

- **Final_eng.ipynb** — a notebook with data exploration, preprocessing steps, and model building;
- **main.py** — a module with the full training and model saving pipeline;
- **api.py** — an implementation of the API that serves the model as a service;
- **default_prediction_model.pkl** — the serialized model in Pickle format;
- **autoencoder_model_weights.pt** — a file containing the weights of the neural network used during the preprocessing stage;
- **classes.py** — a file defining the two classes involved in building and running the neural network. Separating these classes was necessary for the correct loading of the model and the operation of the service.

The project structure is quite extensive, but the entire system runs stably: model training (in **main.py**) takes about an hour, while the service itself handles requests very quickly.

## How to Run the Application

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your_username/Credit-Risk-Management-Model.git


