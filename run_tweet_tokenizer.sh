#!/bin/bash
tweetPath="HIVRepo/pennData/countiesRates"
outputPath="HIVRepo/pennData/countiesRatesTSV"
for file in $( ls $tweetPath ); do
	echo file: "HIVRepo/pennData/countiesRates/$file"
	./ark-tweet-nlp-0.3.2/runTagger.sh --input-format text "$tweetPath/$file" > "$outputPath/${file%.txt}.tsv"
	if [ $? -ne 0 ]; then
		echo "ERROR in file $file"
		exit 1
	fi
done
