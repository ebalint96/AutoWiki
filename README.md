# AutoWiki
Using deep autoencoders to gain insight into Wikipedia articles

## Data
I used Simple English Wikipedia data dump ([available here](https://meta.wikimedia.org/wiki/Data_dump_torrents#Simple_English_Wikipedia)).
Exported each article into separate files, then extracted features using the bag of words model (BoW) with TF-IDF.

## Model
The autoencoder consists of 8 dense layers (4 for the encoder, 4 for the decoder). The output of the encoder is a 2-dimensoinal vector, so it's easy to visualize.

## Results
I trained the network on 10,000 examples, then tested on additional 5,000 and plotted them in a [nice interactive graph](http://people.inf.elte.hu/ebalint96/wiki/wiki.html).




