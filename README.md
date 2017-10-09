# SemEval2018: Task 4 Character Identification on Multiparty Dialogues
Entity linking task that identifies each mention as a certain character in multiparty dialogue using cross- document entity resolution.

## Getting Started
The general idea behind this task is to identify the entity being referred to by a speaker. For example we'll take the sentence "See! He's her lobster!" we have here two references that need to be resolved {he's, her} in this case if we look at the final slide on presentation we have the visual context clues for resolving these mentions. In this case He's is referring to Ross Geller and her to Rachel.

This data for the SemEval task is provided in the Conll format which is tab or space delimited data. The column order for the data is specified to the parser. 

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

### Methods
We started by establishing a simple baseline by choosing a speakers most likely tag given a word. From the papers we read, this method seemed to be widely regarded as the method of choice, so we thought it would make a nice baseline for our machine learning algorithms.

We used information from the mentions to create custom feature vectors incorporating both lexical and orthographic properties. We tested a variety of machine learning algorithms in WEKA including Naïve Bayes, SVM, and C.45.
### Prerequisites
What things you need to install the software and how to install them
Weka is installed and the path to Weka has been added to the path
```
pip install conllu
pip install gensim
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

## Authors
* **Casey Beaird** (Baseline implementation)
* **Chase Greco** (Weka execution)
* **Brandon Watts** (Feature extraction)

See also the list of [contributors](https://github.com/cBeaird/SemEval_Character-Identification-on-Multiparty-Dialogues/graphs/contributors) who participated in this project.

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/cBeaird/SemEval_Character-Identification-on-Multiparty-Dialogues/blob/master/LICENSE) file for details