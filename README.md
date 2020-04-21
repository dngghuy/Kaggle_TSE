# Kaggle_TSE
Here stores my code and experiments for this Kaggle Twitter Sentiment Extraction competition.

In this branch, I will do some experiments with Electra.

Experiments:

- Train Electra 5-fold with some custom head.
    
    - [x] Head 1:
    
        - Linear (768, 2)
        
    - [x] Head 2: 
    
        - Linear (768, 128)
        
        - ReLU
        
        - Dropout
        
        - Linear (128, 2)
        
- Train Electra with custom freezing/ weight decay strategy.

    - [x] AdamW with Linear scheduling with Warm up
    
    - [x] SGD with OneCyclic Scheduling, 2 stages
    
        - In stage 1 we freeze all of the BERT layers, only train custom head
        
        - In stage 2 we unfreeze all the weights and train with different learning rate for each group.
        
- Initialize scheme.

    - [x] Normal init for header
    
    - [ ] Xavier init for header
    
- Taking advantages from sentiment (which appears in both train and test set as feature)

    - [x] Include the class into input tweets as extra token.

    - [x] Fine-tuning BERT and custom head through sentiment analysis first
    
    - [ ] Multi-task learning scheme - one for sentiment classification and one for sentiment extraction.
    
- Tricks (?!)

    - [ ] Train only the positive and negative tweets, taking advantages of the fact that
     extracted neutral tweet usually the tweet itself 
     