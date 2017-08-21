library("tidyverse")
library("text2vec")
library("caret")
library("glmnet")


sentiment_prob = function(train_dat,new_dat,wd){

setwd(wd)
  
#Read training dataset
  #Create training data
  tweets_dat = read.csv(train_dat,stringsAsFactors = FALSE); names(tweets_dat) = c("sentiment","id","date","query","user","text")
  tweets_dat$sentiment = ifelse(tweets_dat$sentiment == 0,0,1)
  
  # data splitting on train and test
  set.seed(2340)
  trainIndex <- createDataPartition(tweets_dat$sentiment, p = 0.8, 
                                    list = FALSE, 
                                    times = 1)
  tweets_train <- tweets_dat[trainIndex, ]
  tweets_test <- tweets_dat[-trainIndex, ]  

  ##### doc2vec #####
  # define preprocessing function and tokenization function
  prep_fun <- tolower
  tok_fun <- word_tokenizer

  it_train <- itoken(tweets_train$text, 
                     preprocessor = prep_fun, 
                     tokenizer = tok_fun,
                     ids = tweets_train$id,
                     progressbar = TRUE)
  
  # creating vocabulary and document-term matrix
  vocab <- create_vocabulary(it_train)
  vectorizer <- vocab_vectorizer(vocab)
  
  #read in csv
  #Change this to get the dataframe instead
  # assign("df_tweets",new_dat)
  df_tweets = get(new_dat)
  # df_tweets = read.csv(new_dat,stringsAsFactors = FALSE)
  names(df_tweets)[which(names(df_tweets) == "id.x")] = "id"
  
  
  # preprocessing and tokenization
  it_tweets <- itoken(df_tweets$text,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = df_tweets$id,
                      progressbar = TRUE)
  
  # creating vocabulary and document-term matrix
  dtm_tweets <- create_dtm(it_tweets, vectorizer)
  
  # transforming data with tf-idf
  # define tf-idf model
  tfidf <- TfIdf$new()
  dtm_tweets_tfidf <- fit_transform(dtm_tweets, tfidf)
  
  # loading classification model
  glmnet_classifier <- readRDS(trained_rds_file)
  
  # predict probabilities of positiveness
  # Can vary the thresholds of what's positive
  preds_tweets <- predict(glmnet_classifier, dtm_tweets_tfidf, type = 'response')[ ,1]
  
  # adding rates to initial dataset
  df_tweets$sentiment <- preds_tweets
  
  #Return dataset
  return(df_tweets)
  
}
