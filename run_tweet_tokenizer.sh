#!/bin/bash
tweetPath="HIVTweets/cities"
for file in $( ls $tweetPath ); do
	echo file: "HIVTweets/cities/$file"
	./ark-tweet-nlp-0.3.2/runTagger.sh --input-format text "$tweetPath/$file" > "$tweetPath/${file%.txt}.tsv"
	if [ $? -ne 0 ]; then
		exit 1
	fi
done