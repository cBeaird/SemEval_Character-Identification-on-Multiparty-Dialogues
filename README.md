# SemEval2018: Task 4 Character Identification on Multiparty Dialogues

## Introduction
Character identification is the task of linking mentions within a dialog to the characters that they reference. 
Mentions are in the form of words referencing a person (eg. _he_, _Dad_, _Joey_) which then must be linked to an 
"entity" or character within the dialog. What can make this task challenging is that the entity which a mention 
references may not be a direct participant of the dialog, meaning that information regarding references must be stored 
and cataloged across multiple dialogs. This task has several applications as part of a larger NLP pipeline such as 
building a question answering system or text summarization. The SemEval 2018 Task 4: _Character Identification on 
Multiparty Dialogues_ focuses on this task and is the objective of our work. The task is presented as identifying 
mentions of characters within the scripts of the first two seasons of the popular television show "Friends".
The scripts have been provided pre-annotated in Conll format as part of the materials for the task.
The objective then, is to devise a system that can correctly identify mentions of not only the main characters of the 
show, but any and all characters which make appearances.  For example, given the piece of dialog "See! He's her 
lobster!", the goal is to correctly match the two references "he" and "she" to the corresponding characters that they 
refer to, in this case Ross Geller and Rachel.  When evaluating the system, special emphasis is placed on accuracy and 
F-1 measures on the main characters as the system will be examined across all entities, as well as across the main 
characters specifically.  

Our motivation behind our approach to this task was to first begin with a simple probabilistic model which would select 
the most likely entity a mention refers to based solely on the character speaking the line of dialog and the mention 
word they used.  We then use this model as a baseline with which we compare against more sophisticated machine learning 
approaches, where much of the focus of contemporary research lies. Finally, based on our preliminary results, we propose 
utilizing neural networks as a possible method to achieve higher performance than previously attempted approaches.


## Getting Started
The general idea behind this task is to identify the entity being referred to by a speaker. For example we'll take the 
sentence "See! He's her lobster!" we have here two references that need to be resolved {he's, her} in this case if we 
look at the final slide on presentation we have the visual context clues for resolving these mentions. In this case He's 
is referring to Ross Geller and her to Rachel.

This data for the SemEval task is provided in the Conll format which is tab or space delimited data. The column order 
for the data is specified to the parser. 

an example of our data is: 

\#begin document (/friends-s01e01) <br />
/friends-s01e01 0 0 There EX (TOP(S(NP*) there - - Monica_Geller * - <br />
/friends-s01e01 0 1 's VBZ (VP* be - - Monica_Geller * - <br />
/friends-s01e01 0 2 nothing NN (NP* nothing - - Monica_Geller * - <br />
/friends-s01e01 0 3 to TO (S(VP* to - - Monica_Geller * - <br />
/friends-s01e01 0 4 tell VB (VP*))))) tell - - Monica_Geller * - <br />
/friends-s01e01 0 5 ! . *)) ! - - Monica_Geller * - <br /> <br />

/friends-s01e01 0 0 He PRP (TOP(S(NP*) he - - Monica_Geller * (284) <br />
/friends-s01e01 0 1 's VBZ (VP* be - - Monica_Geller * - <br />
/friends-s01e01 0 2 just RB (ADVP*) just - - Monica_Geller * - <br />
/friends-s01e01 0 3 some DT (NP(NP* some - - Monica_Geller * - <br />
/friends-s01e01 0 4 guy NN *) guy - - Monica_Geller * (284) <br />
/friends-s01e01 0 5 I PRP (SBAR(S(NP*) I - - Monica_Geller * (248) <br />
/friends-s01e01 0 6 work VBP (VP* work - - Monica_Geller * - <br />
/friends-s01e01 0 7 with IN (PP*)))))) with - - Monica_Geller * - <br />
/friends-s01e01 0 8 ! . *)) ! - - Monica_Geller * - <br />

## Methods

### Most Likely Baseline
The most likely tag baseline is a fairly simple and straight forward method to determine the entity being referenced. 
When we look at more complicated models we see that we might want to encode the corpus tokens to better understand the 
relationship between words. Additionally we might look at modeling the relationship between the speakers. However, with 
the most likely reference baseline these tasks are unnecessary. Simply we need to capture the speaker making the 
reference, and the word the speaker uses to reference the entity. In our case we can look and at the simplified example: 

    {sentence: Mike said he likes boating, speaker: Tom, reference: Mike}
when we parse this sentence out into the conll format we get:

| Word         | Speaker  | Entity |
|:-------------|:---------|:------:|
| Mike         | Tom      | Mike   |
| said         | Tom      | -      |
| he           | Tom      | Mike   |
| likes        | Tom      | -      |
| boating      | Tom      | -      |

Given we are not interested in the relationships between words and are only interested in identifying the referred 
entity we can reduce our data to:  

| Word         | Speaker  | Entity |
|:-------------|:---------|:------:|
| Mike         | Tom      | Mike   |
| he           | Tom      | Mike   |

We develop a simple algorithm to capture these references, words and speakers into a dictionary. 
This dictionary is comprised of a speaker, followed by a dictionary of words the speaker used to refer to an entity, 
and finally a dictionary to count the number of times the speaker used word w to refer to entity e. This looks like: 

    {Tom: {Mike: {Mike, 1}, he: {Mike, 1}}}

Once this information has been extracted from the training data we simply evaluate by lookup. We find a reference 
that needs to be tagged. We read the dictionary of speakers to find the correct speaker; further read that speakers 
word list for the word used to reference the entity and find the entity with the highest count. To follow our example 
we come across the sentence: 

    {sentence: Mike is a really nice guy, speaker: Tom, reference: Mike}
From this sentence we know we need to identify the following entities being referred to.

| Word         | Speaker  | Entity |
|:-------------|:---------|:------:|
| Mike         | Tom      | ?      |
| guy          | Tom      | ?      |

If we follow through the evaluation algorithm we will retrieve Tom, the speaker, looking to see if we have seen the word 
Mike. 
Given that Mike is in our list we will simply find the entity that satisfies the highest probability given the speaker 
and the word, then return that answer. 
In this case we will return Mike and be correct. The process is repeated for the word guy which we have not seen before 
so we will simply guess the answer from the entities provided.

### Machine Learning Approach
We then explored machine learning approaches to compare against our most likely tag baseline. 
To begin we use gensim to map  the mentions to vectors using a skip-gram model and the training data 
provided as the corpora. We added additional orthographic features to our word2vec vector in an attempt 
to combine lexical and orthographic information. Feature Vectors have the following structure: 
[Season, Episode, Scene ID, Speaker, Word2Vec Representation, Entity ID].

#### Deep Neural Networks

#### Decision Tree

#### Random Forest

We then examined the performance of three machine learning algorithms, Naïve Bayes, SVM, and C.45, utilizing these 
feature vectors.  The run configurations of each algorithm can be seen in the "Running Example" section below.  To 
evaluate the performance we compared the accuracies of each algorithm when trained and tested on the entire training set
and when utilizing 10 fold cross-validation against the accuracy of our most likely tag baseline. 

### Prerequisites
#### Python 2.7
packages `conllu` and `gensim`
```
pip install conllu
pip install gensim
pip install tensorflow
```
#### Weka  
The latest stable versions of weka are available here https://www.cs.waikato.ac.nz/ml/weka/downloading.html  
Download latest stable version of weka for your operating system  
Before attempting to run the project add weka to your java classpath  
```
export CLASSPATH=/path/to/weka/weka-3-8-1/weka.jar:$CLASSPATH
```

### Running Example
Once all the required packages are installed the base line can be trained with the following command.<br />
```
python vcu_cmsc516_semeval4.py -m <model_name> --train -mf <entity_map_file> -df <data_file>
python vcu_cmsc516_semeval4.py -m <model_name> --evaluate -df <data_file>
```


The Machine learning can be executed with the following commands from within the Machine learning directory
```
python create_model.py <training_data>
python start.py <data_file> <map_py>
java weka.core.converters.CSVLoader weka.csv -B 50000 > weka.arff
java weka.classifiers.bayes.NaiveBayes -t weka.arff
java weka.classifiers.trees.J48 -C 0.25 -M 2 -t weka.arff
java -Xmx6g weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007" -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4" -t weka.arff
java weka.attributeSelection.GainRatioAttributeEval -i weka.arff
```

## Built With
* [Python](https://www.python.org/)
* [Genism](https://radimrehurek.com/gensim/)
* [Conllu](https://github.com/EmilStenstrom/conllu)
* [Weka](https://www.cs.waikato.ac.nz/ml/weka/)
* [TensorFlow](https://www.tensorflow.org)

## Authors
* **Casey Beaird** Construction of basic Python framework for the intial parsing of data and training of the simple probabilistic model
* **Chase Greco** Team coordination and development of machine learing models in Weka
* **Brandon Watts** Development of additional data parsing methods, exctracting of base features into feature vectors, and feature vector manipulation in Word2Vec

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/cBeaird/SemEval_Character-Identification-on-Multiparty-Dialogues/blob/master/LICENSE) file for details
