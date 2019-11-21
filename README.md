# News Analysis with pyspark

# Table of Contents
1. [Context](#context)
2. [Workflow](#workflow)
3. [Honor Code](#code)

## 1. Context <a name="context"></a>
We were tasked with performing big data analitycs to a set of more than 100,000 news articles written in english.
The dataset can be found in https://www.kaggle.com/snapcrack/all-the-news. It consists of three .csv files each with around 50,000 articles, though to ease the burden, we chose to blend the files into a single articles.csv file that was used for the analysis.

## 2. Workflow <a name="workflow"></a>
We worked in two environments, the first was AWS using notebooks with pyspark on top of a EMR cluster, there the data was stored in s3. Our second environment was the DCA, again using notebooks with pyspark, but this time the data was stored in the cluster's hdfs.

## 3. Honor Code <a name="code"></a>

### Juan Gonzalo Quiroz

### Valentino Malvarmo
I, Valentino Malvarmo, declare that all the content aported by me is of my own authorship. I contributed in the phase of data preparation, documentation and with topic modeling.

### William Cadenas
